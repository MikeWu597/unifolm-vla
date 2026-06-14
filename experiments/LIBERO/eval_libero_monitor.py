"""
VLA evaluation with MiniCPM-o-4.5 monitoring.

VLA runs normally on LIBERO. Every step pushes a video frame to the
HTTP server. Every N steps, MiniCPM-o-4.5 analyzes the latest frames
and outputs text observations. No tool calling — observation only.

Usage:
  python experiments/LIBERO/eval_libero_monitor.py \
      --args.pretrained-path /path/to/checkpoint.pt \
      --args.vlm-pretrained-path /path/to/UnifoLM-VLM-Base \
      --args.task-suite-name libero_spatial \
      --args.num-trials-per-task 5 \
      --args.minicpm-interval 50
"""

import base64
import dataclasses
import io
import json
import logging
import math
import os
import pathlib
import time
from collections import deque
from pathlib import Path
from typing import Any

import imageio
import numpy as np
import requests
import tqdm
import tyro

# torch 2.6+ compat
import torch
_orig_load = torch.load
torch.load = lambda *a, **kw: _orig_load(*a, **{'weights_only': False, **kw})

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from unifolm_vla.rlds_dataloader.constants import NUM_ACTIONS_CHUNK

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from experiments.LIBERO.libero_utils import (
    DATE, DATE_TIME,
    get_libero_image, get_libero_wrist_image,
    quat2axisangle, resize_image_for_policy,
)
from experiments.LIBERO.unifolm_vla_inference import Unifolm_VLA_Inference

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 512
VIDEO_FPS = 30
MONITOR_PORT = 8778
JPG_QUALITY = 75


# ── Config ────────────────────────────────────────────────────────────
@dataclasses.dataclass
class Args:
    task_suite_name: str = "libero_spatial"
    num_steps_wait: int = 10
    num_trials_per_task: int = 3
    window_size: int = 2
    minicpm_interval: int = 50  # steps between MiniCPM checks
    video_out_path: str = "results"
    local_log_dir: str = "./experiments/logs"
    seed: int = 42
    pretrained_path: str = ""
    post_process_action: bool = True
    unnorm_key: str = "libero_spatial_no_noops"
    vlm_pretrained_path: str = None


# ── Helpers ───────────────────────────────────────────────────────────
def prepare_observation(obs, resize_size=224):
    img = get_libero_image(obs)
    wrist_img = get_libero_wrist_image(obs)
    img_resized = resize_image_for_policy(img, resize_size)
    wrist_img_resized = resize_image_for_policy(wrist_img, resize_size)
    observation = {
        "full_image": img_resized,
        "wrist_image": wrist_img_resized,
        "state": np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        ),
    }
    return observation, img  # img is the full-resolution render


def send_frame(img: np.ndarray, status: str = "", step: int = 0, interval: int = 50):
    """Push a frame to the MiniCPM HTTP server."""
    try:
        buf = io.BytesIO()
        from PIL import Image
        Image.fromarray(img).save(buf, format="JPEG", quality=JPG_QUALITY)
        b64 = base64.b64encode(buf.getvalue()).decode()
        requests.post(f"http://127.0.0.1:{MONITOR_PORT}/frame",
                      json={"image": b64, "status": status,
                            "step": step, "interval": interval}, timeout=1)
    except Exception:
        pass


def send_text(text: str):
    """Push MiniCPM observation text to the HTTP server."""
    try:
        requests.post(f"http://127.0.0.1:{MONITOR_PORT}/text",
                      json={"text": text}, timeout=1)
    except Exception:
        pass


def normalize_gripper_action(action, binarize=True):
    a = action.copy()
    a[..., -1] = 2 * (a[..., -1] - 0.0) / (1.0 - 0.0) - 1
    if binarize:
        a[..., -1] = np.sign(a[..., -1])
    return a


def invert_gripper_action(action):
    a = action.copy()
    a[..., -1] *= -1.0
    return a


def process_action(action):
    return invert_gripper_action(normalize_gripper_action(action))


def _get_libero_env(task, resolution, seed):
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=task_bddl_file,
                             camera_heights=resolution, camera_widths=resolution)
    env.seed(seed)
    return env, task.language


def get_action_state(observations, instruction, model):
    from qwen_vl_utils import process_vision_info
    from experiments.LIBERO.libero_utils import prepare_images_for_vla

    all_images = []
    for obs in observations:
        all_images.append(obs["full_image"])
    for obs in observations:
        all_images.extend([obs[k] for k in obs.keys() if "wrist" in k])
    all_images = prepare_images_for_vla(all_images)

    text = (f'You are a robot using joint control. '
            f'The task is "{instruction.lower()}". '
            f'Please predict up to 10 key trajectory points to complete the task.')

    messages = [{"role": "user", "content": [
        *[{"type": "image", "image": img} for img in all_images],
        {"type": "text", "text": text},
    ]}]

    proc = model.vla.qwen_vl_interface.processor
    text_p = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    img_in, vid_in = process_vision_info(messages)
    q_in = proc(text=text_p, images=img_in, videos=vid_in, padding=True, return_tensors="pt")
    q_in["state"] = np.stack([o["state"] for o in observations], axis=0)
    return model.step(q_in)


# ── Main ──────────────────────────────────────────────────────────────
def eval_libero_monitor(args: Args):
    logging.info(f"Arguments: {json.dumps(dataclasses.asdict(args), indent=4)}")
    np.random.seed(args.seed)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}, {num_tasks} tasks")

    pathlib.Path(args.video_out_path + "/" + DATE).mkdir(parents=True, exist_ok=True)

    # Load VLA
    model = Unifolm_VLA_Inference(
        policy_ckpt_path=args.pretrained_path,
        image_size=[224, 224],
        unnorm_key=args.unnorm_key,
        vlm_pretrained_path=args.vlm_pretrained_path,
    )

    # MiniCPM runs in separate conda env — already started via run_monitor.sh

    total_episodes, total_successes = 0, 0

    for task_id in tqdm.tqdm(range(num_tasks)):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            model.reset(task_description=task_description)
            env.reset()

            t = 0
            step = 0
            replay_images = []
            action_queue = deque(maxlen=NUM_ACTIONS_CHUNK)
            obs_queue = deque(maxlen=args.window_size)
            success = False

            while t < 10000:  # no step limit
                if t < args.num_steps_wait:
                    obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
                    t += 1
                    continue

                while len(obs_queue) < args.window_size:
                    observation, img_full = prepare_observation(obs, resize_size=224)
                    obs_queue.append(observation)
                replay_images.append(img_full)

                if len(action_queue) == 0:
                    actions = get_action_state(obs_queue, task_description, model)
                    action_queue.extend(actions)

                obs_queue.popleft()
                action = action_queue.popleft()
                obs, _, done_flag, _ = env.step(process_action(action).tolist())

                # Push frame to monitor
                send_frame(img_full, f"VLA | Step {step+1} | {task_description[:60]}",
                           step=step + 1, interval=args.minicpm_interval)

                if done_flag:
                    success = True
                    break

                t += 1
                step += 1

            # Episode done
            total_episodes += 1
            if success:
                total_successes += 1

            suffix = "success" if success else "failure"
            seg = task_description.replace(" ", "_")
            if replay_images:
                imageio.mimwrite(
                    pathlib.Path(args.video_out_path) / f"monitor_{seg}_ep{episode_idx}_{suffix}.mp4",
                    [np.asarray(x) for x in replay_images], fps=VIDEO_FPS,
                )

            logging.info(
                f"Ep {total_episodes}: {suffix.upper()} "
                f"({total_successes}/{total_episodes} = {total_successes/total_episodes*100:.1f}%)"
            )

    rate = total_successes / total_episodes if total_episodes else 0
    logging.info(f"FINAL: {total_successes}/{total_episodes} = {rate:.1%}")


if __name__ == "__main__":
    tyro.cli(eval_libero_monitor)
