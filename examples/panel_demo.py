"""
UnifoLM-VLA-Panel: browser-controlled VLA demo.

Start: python examples/panel_demo.py
Open:  http://localhost:8778
"""

import argparse
import logging
import os
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from qwen_vl_utils import process_vision_info
from PIL import Image

# torch 2.6+ compat
_orig_load = torch.load
torch.load = lambda *a, **kw: _orig_load(*a, **{'weights_only': False, **kw})

from unifolm_vla.rlds_dataloader.constants import NUM_ACTIONS_CHUNK
from unifolm_vla.panel.scene import create_scene, get_image, get_state
from unifolm_vla.model.framework.unifolm_vla import Unifolm_VLA
from unifolm_vla.model.framework.share_tools import read_mode_config, dict_to_namespace
from unifolm_vla.model.framework import build_framework

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
DEVICE = torch.device("cuda:0")


def load_vla(checkpoint_path: str, vlm_path: str) -> Unifolm_VLA:
    """Load VLA model from checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    model_config, norm_stats = read_mode_config(checkpoint_path)
    config = dict_to_namespace(model_config)
    config.framework.qwenvl.base_vlm = vlm_path
    m = build_framework(cfg=config)
    m.norm_stats = norm_stats
    sd = torch.load(checkpoint_path, map_location="cpu")
    m.load_state_dict(sd, strict=False)
    m = m.to(torch.bfloat16).to(DEVICE).eval()
    return m, norm_stats


def build_inputs(images: list, instruction: str, state: np.ndarray, model, norm_stats):
    """Build qwen_inputs for predict_action."""
    from unifolm_vla.rlds_dataloader.constants import (
        ACTION_PROPRIO_NORMALIZATION_TYPE, NormalizationType,
    )

    # Normalize proprio state
    ns = norm_stats.get("libero_spatial_no_noops", norm_stats.get(list(norm_stats.keys())[0]))
    action_ns = ns["action"]
    proprio_ns = ns["proprio"]

    if ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS_Q99:
        mask = proprio_ns.get("mask", np.ones_like(proprio_ns["q01"], dtype=bool))
        lo, hi = np.array(proprio_ns["q99"]), np.array(proprio_ns["q01"])
    else:
        mask = proprio_ns.get("mask", np.ones_like(proprio_ns["min"], dtype=bool))
        lo, hi = np.array(proprio_ns["max"]), np.array(proprio_ns["min"])
    norm_state = np.clip(np.where(mask, 2 * (state - lo) / (hi - lo + 1e-8) - 1, state), -1, 1)

    # Resize images
    pil_images = [Image.fromarray(img).resize((224, 224)).convert("RGB") for img in images]

    text = f'The task is "{instruction.lower()}".'
    messages = [{"role": "user", "content": [
        *[{"type": "image", "image": img} for img in pil_images],
        {"type": "text", "text": text},
    ]}]

    processor = model.processor
    txt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    img_in, vid_in = process_vision_info(messages)
    q_in = processor(text=txt, images=img_in, videos=vid_in, padding=True, return_tensors="pt")

    q_in["state"] = torch.from_numpy(norm_state).unsqueeze(0).unsqueeze(0).to(DEVICE)
    for k in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw"):
        if k in q_in:
            q_in[k] = q_in[k].to(DEVICE)
    return q_in


def unnorm_actions(actions, norm_stats, unnorm_key=None):
    """Un-normalize predicted actions."""
    from unifolm_vla.rlds_dataloader.constants import (
        ACTION_PROPRIO_NORMALIZATION_TYPE, NormalizationType,
    )
    ns = norm_stats.get(unnorm_key, norm_stats.get(list(norm_stats.keys())[0]))
    action_ns = ns["action"]
    if ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS_Q99:
        mask = action_ns.get("mask", np.ones_like(action_ns["q01"], dtype=bool))
        hi, lo = np.array(action_ns["q99"]), np.array(action_ns["q01"])
    else:
        mask = action_ns.get("mask", np.ones_like(action_ns["min"], dtype=bool))
        hi, lo = np.array(action_ns["max"]), np.array(action_ns["min"])
    return np.where(mask, 0.5 * (actions + 1) * (hi - lo + 1e-8) + lo, actions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="./models/UnifoLM-VLA-LIBERO/checkpoints/pytorch_model.pt")
    parser.add_argument("--vlm", default="./models/UnifoLM-VLM-Base")
    parser.add_argument("--port", type=int, default=8778)
    parser.add_argument("--window", type=int, default=2)
    args = parser.parse_args()

    # ── Load VLA ──────────────────────────────────────────────────────
    logger.info("Loading VLA...")
    model, norm_stats = load_vla(args.ckpt, args.vlm)
    logger.info("VLA loaded.")

    # ── Create scene ───────────────────────────────────────────────────
    logger.info("Creating robosuite scene...")
    env = create_scene(render_resolution=512)
    env.reset()

    # ── Start server ──────────────────────────────────────────────────
    from unifolm_vla.panel.server import app as server_app, push_frame, \
        next_command, set_status, get_status
    import threading, uvicorn

    def run_server():
        uvicorn.run(server_app, host="0.0.0.0", port=args.port, log_level="info")

    t = threading.Thread(target=run_server, daemon=True)
    t.start()
    time.sleep(2)
    logger.info(f"Server running on http://localhost:{args.port}")

    # ── Main loop ─────────────────────────────────────────────────────
    obs_queue = deque(maxlen=args.window)
    action_queue = deque(maxlen=NUM_ACTIONS_CHUNK)
    current_instruction = None
    max_steps_per_cmd = 500

    # Dummy action for observation collection
    DUMMY = [0.0] * 6 + [-1.0]

    while True:
        # Check for new command
        cmd = next_command()
        if cmd is not None:
            if cmd == "__STOP__":
                current_instruction = None
                action_queue.clear()
                obs_queue.clear()
                set_status(running=False, instruction="")
                continue
            if cmd == "__RESET__":
                env.reset()
                action_queue.clear()
                obs_queue.clear()
                set_status(running=False, instruction="")
                logger.info("Scene reset")
                continue
            # Same instruction → skip, don't reset context
            if cmd == current_instruction:
                continue
            current_instruction = cmd
            action_queue.clear()
            obs_queue.clear()
            set_status(running=True, instruction=cmd, step=0, active_since=time.time())
            logger.info(f"New command: {cmd}")

        # Execute VLA step if we have an instruction
        if current_instruction:
            s = get_status()
            if s["step"] >= max_steps_per_cmd:
                set_status(running=False, instruction="")
                current_instruction = None
                action_queue.clear()
                continue

            # Execute: VLA action if available, else dummy
            if len(action_queue) > 0 and len(obs_queue) >= args.window:
                act = action_queue.popleft()
                act[..., -1] *= -1.0
                try:
                    obs, _, done, _ = env.step(act.tolist())
                except ValueError:
                    env.reset()
                    obs, _, done, _ = env.step(act.tolist())
                if done:
                    env.reset()
            else:
                try:
                    obs, _, _, _ = env.step(np.array(DUMMY))
                except ValueError:
                    env.reset()
                    obs, _, _, _ = env.step(np.array(DUMMY))

            img = get_image(env, obs)
            wrist_img = obs["robot0_eye_in_hand_image"][::-1, ::-1]
            state = get_state(env, obs)
            push_frame(img)

            obs_queue.append({"full_image": img, "wrist_image": wrist_img, "state": state})

            if len(obs_queue) >= args.window and len(action_queue) == 0:
                images = []
                for o in obs_queue:
                    images.append(o["full_image"])
                for o in obs_queue:
                    images.extend([o[k] for k in o.keys() if "wrist" in k])
                q_in = build_inputs(images, current_instruction, state, model, norm_stats)
                raw = model.predict_action(q_in)
                actions = unnorm_actions(raw["normalized_actions"][0], norm_stats)
                action_queue.extend(actions)

            set_status(step=s["step"] + 1)
            time.sleep(0.02)
        else:
            # Idle — still push frames with dummy steps
            obs, _, _, _ = env.step(np.array(DUMMY))
            img = get_image(env, obs)
            push_frame(img)
            time.sleep(0.1)


if __name__ == "__main__":
    main()
