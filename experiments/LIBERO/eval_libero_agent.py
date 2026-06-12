"""
Agentic VLA evaluation on LIBERO benchmark.

Extends eval_libero.py with a ToolCallAgent that monitors VLA execution
and can intervene with new instructions or early termination.

Dual-head architecture (shared Qwen2.5-VL backbone):
  VLA Head:  predict_action() → 7D actions, every step
  LLM Head:  predict_text()   → agent reasoning + tool calls, every N steps

Usage:
  python experiments/LIBERO/eval_libero_agent.py \
      --args.pretrained-path /path/to/checkpoint.pt \
      --args.vlm-pretrained-path /path/to/UnifoLM-VLM-Base \
      --args.task-suite-name libero_spatial \
      --args.num-trials-per-task 5 \
      --args.vla-interval 10
"""

import dataclasses
import json
import logging
import math
import os
import pathlib
from pathlib import Path
import time
from collections import deque
from typing import Any, Dict, List, Optional

import imageio
import numpy as np
import tqdm
import tyro
import torch

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from qwen_vl_utils import process_vision_info

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from unifolm_vla.rlds_dataloader.constants import NUM_ACTIONS_CHUNK
from unifolm_vla.agent.agent_loop import ToolCallAgent, AgentDecision

from experiments.LIBERO.libero_utils import DATE_TIME, DATE
from experiments.LIBERO.unifolm_vla_inference import Unifolm_VLA_Inference
from experiments.LIBERO.libero_utils import (
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
    resize_image_for_policy,
    prepare_images_for_vla,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class Args:
    resize_size = [224, 224]

    # LIBERO environment
    task_suite_name: str = "libero_spatial"
    num_steps_wait: int = 10
    num_trials_per_task: int = 5
    window_size: int = 2

    # Agent
    vla_interval: int = 10  # VLA steps between agent checks
    max_agent_rounds: int = 30

    # Paths
    video_out_path: str = "results"
    local_log_dir: str = "./experiments/logs"
    seed: int = 42
    pretrained_path: str = ""
    post_process_action: bool = True
    unnorm_key: str = "libero_spatial_no_noops"
    vlm_pretrained_path: str = None


# ---------------------------------------------------------------------------
# Helpers (same as eval_libero.py)
# ---------------------------------------------------------------------------
def prepare_observation(obs, resize_size):
    img = get_libero_image(obs)
    wrist_img = get_libero_wrist_image(obs)
    img_resized = resize_image_for_policy(img, resize_size)
    wrist_img_resized = resize_image_for_policy(wrist_img, resize_size)
    observation = {
        "full_image": img_resized,
        "wrist_image": wrist_img_resized,
        "state": np.concatenate(
            (
                obs["robot0_eef_pos"],
                quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"],
            )
        ),
    }
    return observation, img


def setup_logging(args: Args):
    run_id = f"EVAL-AGENT-{args.task_suite_name}-{DATE_TIME}"
    os.makedirs(args.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(args.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    logger.info(f"Logging to: {local_log_filepath}")
    return log_file, local_log_filepath


def log_message(message: str, log_file=None):
    logger.info(message)
    if log_file:
        log_file.write(message + "\n")
        log_file.flush()


def get_action_state(
    observations: deque,
    task_description: str,
    model: Unifolm_VLA_Inference,
):
    """Build qwen_inputs and run VLA inference. Returns action chunk."""
    all_images = []
    for obs in observations:
        all_images.append(obs["full_image"])
    for obs in observations:
        all_images.extend([obs[k] for k in obs.keys() if "wrist" in k])

    all_images = prepare_images_for_vla(all_images)

    text = (
        f"You are a robot using joint control. "
        f'The task is "{task_description.lower()}". '
        f"Please predict up to 10 key trajectory points to complete the task."
    )

    messages = [
        {
            "role": "user",
            "content": [
                *[{"type": "image", "image": img} for img in all_images],
                {"type": "text", "text": text},
            ],
        },
    ]

    text_processed = model.vla.qwen_vl_interface.processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    qwen_inputs = model.vla.qwen_vl_interface.processor(
        text=text_processed,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    qwen_inputs["state"] = np.stack([obs["state"] for obs in observations], axis=0)

    actions = model.step(qwen_inputs)
    return actions


def get_agent_qwen_inputs(
    observations: deque,
    task_description: str,
    model: Unifolm_VLA_Inference,
    agent_prompt: str,
):
    """Build qwen_inputs for the LLM Head (agent check)."""
    all_images = []
    for obs in observations:
        all_images.append(obs["full_image"])
    for obs in observations:
        all_images.extend([obs[k] for k in obs.keys() if "wrist" in k])

    all_images = prepare_images_for_vla(all_images)

    messages = [
        {
            "role": "user",
            "content": [
                *[{"type": "image", "image": img} for img in all_images],
                {"type": "text", "text": agent_prompt},
            ],
        },
    ]

    text_processed = model.vla.qwen_vl_interface.processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    qwen_inputs = model.vla.qwen_vl_interface.processor(
        text=text_processed,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    return qwen_inputs


def normalize_gripper_action(action: np.ndarray, binarize: bool = True) -> np.ndarray:
    normalized_action = action.copy()
    orig_low, orig_high = 0.0, 1.0
    normalized_action[..., -1] = (
        2 * (normalized_action[..., -1] - orig_low) / (orig_high - orig_low) - 1
    )
    if binarize:
        normalized_action[..., -1] = np.sign(normalized_action[..., -1])
    return normalized_action


def invert_gripper_action(action: np.ndarray) -> np.ndarray:
    inverted_action = action.copy()
    inverted_action[..., -1] *= -1.0
    return inverted_action


def process_action(action):
    action = normalize_gripper_action(action, binarize=True)
    action = invert_gripper_action(action)
    return action


def _get_libero_env(task, resolution, seed):
    task_description = task.language
    task_bddl_file = (
        pathlib.Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


# ---------------------------------------------------------------------------
# Agent check function
# ---------------------------------------------------------------------------
def agent_check(
    model: Unifolm_VLA_Inference,
    agent: ToolCallAgent,
    obs_queue: deque,
    task_description: str,
    step_counter: int,
    max_steps: int,
    log_file=None,
) -> AgentDecision:
    """
    Run the LLM Head to check task progress.

    Returns an AgentDecision: continue / new_instruction / terminate.
    """
    # Build agent-specific prompt
    agent_prompt = (
        f"You are a robot task supervisor monitoring a VLA model.\n\n"
        f'Current task: "{task_description}"\n'
        f"Progress: {step_counter} / {max_steps} steps executed.\n\n"
        f"Your role:\n"
        f"1. Assess whether the VLA is making progress.\n"
        f"2. If progress is normal, output your observation. Do NOT call any tool.\n"
        f"3. If the task is stuck or failing, call call_vla with a corrected instruction.\n"
        f"4. If the task appears completed, call terminate with success=true.\n"
        f"5. If the task has irrecoverably failed, call terminate with success=false.\n\n"
        f"Available tools:\n"
        f"- call_vla: Give the VLA a new sub-task instruction. "
        f'Parameters: {{"instruction": "new task", "reason": "why", "max_vla_steps": 50}}\n'
        f"- terminate: End the episode. "
        f'Parameters: {{"success": true/false, "reason": "why"}}\n\n'
        f"To call a tool, output:\n"
        f"<tool_call>\n"
        f'{{"name": "<tool_name>", "arguments": {{...}}}}\n'
        f"</tool_call>\n\n"
        f"Analyze the current observation and decide."
    )

    # Build qwen inputs for the LLM Head
    qwen_inputs = get_agent_qwen_inputs(obs_queue, task_description, model, agent_prompt)

    # Move to GPU
    device = model.vla.qwen_vl_interface.model.device
    for k, v in list(qwen_inputs.items()):
        if isinstance(v, torch.Tensor):
            qwen_inputs[k] = v.to(device)

    # Run LLM Head
    try:
        text = model.vla.predict_text(qwen_inputs, max_new_tokens=256)
    except Exception as e:
        logger.warning(f"LLM Head failed: {e}. Defaulting to continue.")
        log_message(f"[Agent] LLM error: {e}. Continue.", log_file)
        return AgentDecision(action="continue", reason=f"llm error: {e}")

    log_message(f"[Agent] Raw output: {text[:300]}", log_file)

    # Parse
    from unifolm_vla.agent.tool_call_parser import ToolCallParser

    if not ToolCallParser.has_tool_call(text):
        reasoning = ToolCallParser.extract_reasoning(text)
        log_message(f"[Agent] Decision: continue — {reasoning[:200]}", log_file)
        return AgentDecision(action="continue", reason=reasoning or "normal progress")

    tool_calls = ToolCallParser.parse(text)
    if not tool_calls:
        return AgentDecision(action="continue", reason="parse failed")

    tc = tool_calls[0]
    name = tc.get("name", "")
    args = tc.get("arguments", {})

    if name == "terminate":
        decision = AgentDecision(
            action="terminate",
            reason=args.get("reason", ""),
            success=args.get("success", False),
        )
        log_message(f"[Agent] Decision: terminate — {decision.reason}", log_file)
        return decision
    elif name == "call_vla":
        decision = AgentDecision(
            action="new_instruction",
            reason=args.get("reason", ""),
            instruction=args.get("instruction", ""),
            max_vla_steps=args.get("max_vla_steps", 50),
        )
        log_message(
            f"[Agent] Decision: new_instruction — {decision.instruction} "
            f"(reason: {decision.reason})",
            log_file,
        )
        return decision
    else:
        log_message(f"[Agent] Decision: continue — unknown tool '{name}'", log_file)
        return AgentDecision(action="continue", reason=f"unknown tool: {name}")


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------
def eval_libero_agent(args: Args) -> None:
    logging.info(f"Arguments: {json.dumps(dataclasses.asdict(args), indent=4)}")

    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    pathlib.Path(args.video_out_path + "/" + DATE).mkdir(parents=True, exist_ok=True)

    # Max steps per task suite
    max_steps_map = {
        "libero_spatial": 220,
        "libero_object": 280,
        "libero_goal": 300,
        "libero_10": 520,
        "libero_90": 400,
    }
    max_steps = max_steps_map.get(args.task_suite_name, 300)

    # Load model
    model = Unifolm_VLA_Inference(
        policy_ckpt_path=args.pretrained_path,
        image_size=args.resize_size,
        unnorm_key=args.unnorm_key,
        vlm_pretrained_path=args.vlm_pretrained_path,
    )

    # Create agent
    agent = ToolCallAgent(
        model=model,
        vla_interval=args.vla_interval,
        max_agent_rounds=args.max_agent_rounds,
    )

    log_file, local_log_filepath = setup_logging(args)
    log_message(f"Agent config: vla_interval={args.vla_interval}, "
                f"max_rounds={args.max_agent_rounds}", log_file)

    # Evaluation
    total_episodes, total_successes = 0, 0
    agent_interventions = 0  # Count how many times agent changed instruction

    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        task_episodes, task_successes = 0, 0

        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            log_message(f"\nTask: {task_description}", log_file)

            # Reset
            model.reset(task_description=task_description)
            agent.reset()
            env.reset()
            obs = env.set_init_state(initial_states[episode_idx])

            t = 0
            step = 0
            replay_images = []
            current_instruction = task_description
            agent_step_counter = 0  # Steps since last agent check

            action_queue = deque(maxlen=NUM_ACTIONS_CHUNK)
            obs_queue = deque(maxlen=args.window_size)
            success = False
            done = False

            log_message(f"Starting episode {task_episodes + 1}...", log_file)

            while t < max_steps + args.num_steps_wait and not done:
                # Wait phase
                if t < args.num_steps_wait:
                    obs, reward, done_flag, info = env.step(LIBERO_DUMMY_ACTION)
                    t += 1
                    continue

                # Build observation queue
                while len(obs_queue) < args.window_size:
                    observation, img = prepare_observation(obs, resize_size=224)
                    obs_queue.append(observation)
                replay_images.append(img)

                # VLA action inference
                if len(action_queue) == 0:
                    actions = get_action_state(obs_queue, current_instruction, model)
                    action_queue.extend(actions)

                obs_queue.popleft()
                action = action_queue.popleft()
                action_processed = process_action(action)

                obs, reward, done_flag, info = env.step(action_processed.tolist())

                # Record for agent
                agent.record_obs(observation)
                agent.record_action(action)

                if done_flag:
                    success = True
                    done = True
                    break

                t += 1
                step += 1
                agent_step_counter += 1

                # ── Agent check ───────────────────────────────────────
                if agent_step_counter >= args.vla_interval and not done:
                    decision = agent_check(
                        model=model,
                        agent=agent,
                        obs_queue=obs_queue,
                        task_description=task_description,
                        step_counter=step,
                        max_steps=max_steps,
                        log_file=log_file,
                    )

                    if decision.action == "terminate":
                        done = True
                        success = decision.success
                        log_message(
                            f"[Agent] Episode terminated: success={success}, "
                            f"reason={decision.reason}",
                            log_file,
                        )
                    elif decision.action == "new_instruction" and decision.instruction:
                        agent_interventions += 1
                        current_instruction = decision.instruction
                        model.reset(task_description=current_instruction)
                        action_queue.clear()
                        log_message(
                            f"[Agent] New instruction: {current_instruction}",
                            log_file,
                        )

                    agent_step_counter = 0  # Reset counter after check

            # Episode summary
            task_episodes += 1
            total_episodes += 1
            if success:
                task_successes += 1
                total_successes += 1

            # Save failure video
            if not success:
                task_segment = task_description.replace(" ", "_")
                imageio.mimwrite(
                    pathlib.Path(args.video_out_path)
                    / f"agent_rollout_{task_segment}_ep{episode_idx}_failure.mp4",
                    [np.asarray(x) for x in replay_images],
                    fps=10,
                )

            log_message(f"Success: {success}", log_file)
            log_message(
                f"# successes: {total_successes} "
                f"({total_successes / total_episodes * 100:.1f}%)",
                log_file,
            )

        log_message(
            f"Task success rate: {float(task_successes) / float(task_episodes)}",
            log_file,
        )

    # Final results
    final_rate = (
        float(total_successes) / float(total_episodes) if total_episodes > 0 else 0
    )
    log_message("=" * 60, log_file)
    log_message(f"FINAL RESULTS", log_file)
    log_message(f"Total episodes: {total_episodes}", log_file)
    log_message(f"Total successes: {total_successes}", log_file)
    log_message(f"Success rate: {final_rate:.4f} ({final_rate * 100:.1f}%)", log_file)
    log_message(f"Agent interventions: {agent_interventions}", log_file)
    log_message("=" * 60, log_file)

    if log_file:
        log_file.close()


if __name__ == "__main__":
    tyro.cli(eval_libero_agent)
