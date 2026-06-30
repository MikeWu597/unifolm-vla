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

import csv
import dataclasses
import json
import logging
import math
import os
import pathlib

# torch 2.6+: LIBERO uses torch.load with old numpy pickles.
# monkey-patch torch.load so 'weights_only=False' is the default again.
import torch
_orig_load = torch.load
torch.load = lambda *a, **kw: _orig_load(*a, **{'weights_only': False, **kw})
from pathlib import Path
import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import imageio
import numpy as np
import tqdm
import tyro
from PIL import Image, ImageDraw, ImageFont

# fps for video output
VIDEO_FPS = 30
AGENT_PAUSE_SEC = 0.5  # pause on agent frames


def _get_font(size: int = 14) -> ImageFont.FreeTypeFont:
    """Get a monospace font, falling back to default."""
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
                 "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
                 "C:\\Windows\\Fonts\\consola.ttf"):
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()


def draw_overlay(img: np.ndarray, lines: List[str], color: str = "white") -> np.ndarray:
    """
    Draw a semi-transparent text bar at the bottom of an image.

    Args:
        img: numpy RGB image (H, W, 3) uint8
        lines: list of text lines to draw
        color: "white" for normal, "yellow" for agent, "green" for success
    """
    pil = Image.fromarray(img).convert("RGBA")
    draw = ImageDraw.Draw(pil)
    font = _get_font(13)

    colors = {"white": (255, 255, 255), "yellow": (255, 255, 100),
              "green": (100, 255, 100), "red": (255, 100, 100)}

    # Bar dimensions
    line_h = 16
    pad = 6
    bar_h = len(lines) * line_h + pad * 2
    bar_w = pil.width

    # Semi-transparent black bar
    overlay = Image.new("RGBA", (bar_w, bar_h), (0, 0, 0, 180))
    pil.paste(overlay, (0, pil.height - bar_h), overlay)

    # Draw text
    rgb = colors.get(color, colors["white"])
    for i, line in enumerate(lines):
        y = pil.height - bar_h + pad + i * line_h
        draw.text((pad, y), line, fill=rgb, font=font)

    return np.array(pil.convert("RGB"))


def init_csv(csv_path: str):
    """Create CSV file with headers."""
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    f = open(csv_path, "w", newline="")
    w = csv.writer(f)
    w.writerow(["episode", "task", "step", "agent_round", "mode",
                 "agent_output", "tool_call_name", "tool_call_args", "decision"])
    return f, w

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from qwen_vl_utils import process_vision_info

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from unifolm_vla.rlds_dataloader.constants import NUM_ACTIONS_CHUNK
from unifolm_vla.agent.agent_loop import ToolCallAgent, AgentDecision

from experiments.LIBERO.libero_utils import DATE_TIME, DATE
from experiments.LIBERO.unifolm_vla_inference import Unifolm_VLA_Inference
from unifolm_vla.agent.tool_call_parser import ToolCallParser
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
LIBERO_ENV_RESOLUTION = 512  # MP4 render resolution (VLA still sees 224x224)


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
    vla_interval: int = 100  # VLA steps between agent checks
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
def agent_plan(
    model: Unifolm_VLA_Inference,
    obs_queue: deque,
    task_description: str,
    history: Optional[List[dict]] = None,
    log_file=None,
) -> Tuple[AgentDecision, List[dict]]:
    """
    Initial agent planning: look at the scene and decide the first VLA instruction.

    Returns (AgentDecision, updated_history).
    """
    raw_images = []
    for obs in obs_queue:
        raw_images.append(obs["full_image"])
    for obs in obs_queue:
        raw_images.extend([obs[k] for k in obs.keys() if "wrist" in k])

    plan_prompt = (
        f"You are a robot task planner. Look at the scene and break down this task "
        f"into a concrete first step for a VLA model to execute.\n"
        f"Make the instruction SPECIFIC — include actions like 'grasp', 'pick up', 'close gripper', not just 'move towards'.\n\n"
        f'Overall task: "{task_description}"\n\n'
        f"Output a <tool_call> with the first sub-task instruction:\n"
        f"<tool_call>\n"
        f'{{"name":"call_vla","arguments":{{"instruction":"<first step in English>","reason":"<why>","max_vla_steps":50}}}}\n'
        f"</tool_call>"
    )

    try:
        text, history = model.vla.predict_text(
            images=raw_images,
            prompt_text=plan_prompt,
            max_new_tokens=512,
            temperature=0.1,
            history=history,
        )
    except Exception as e:
        logger.warning(f"Agent plan failed: {e}. Using original task.")
        return AgentDecision(action="new_instruction", instruction=task_description), history

    log_message(f"[Agent Plan] {text}", log_file)

    if not ToolCallParser.has_tool_call(text):
        return AgentDecision(action="new_instruction", instruction=task_description), history

    tool_calls = ToolCallParser.parse(text)
    if not tool_calls:
        return AgentDecision(action="new_instruction", instruction=task_description), history

    tc = tool_calls[0]
    args = tc.get("arguments", {})
    return AgentDecision(
        action="new_instruction",
        instruction=args.get("instruction", task_description),
        reason=args.get("reason", ""),
        max_vla_steps=args.get("max_vla_steps", 50),
    ), history


def get_movement_status(agent: ToolCallAgent, window: int = 20) -> str:
    """
    Detect if the robot is moving, stuck, or idle based on recent action variance.
    Actions are 7D: [x, y, z, roll, pitch, yaw, grip].
    """
    if len(agent.action_history) < window:
        return "unknown (not enough history)"

    recent = list(agent.action_history)[-window:]
    actions = np.array(recent)  # (window, 7)

    # Variance of position deltas (first 3 dims)
    pos_var = np.var(actions[:, :3])
    # Variance of rotation + gripper
    rest_var = np.var(actions[:, 3:])

    if pos_var < 1e-6 and rest_var < 1e-6:
        return "STUCK — robot is nearly motionless"
    elif pos_var < 1e-4:
        return "SLOW — robot moving very little"
    else:
        return "MOVING — robot is actively moving"


def agent_check(
    model: Unifolm_VLA_Inference,
    agent: ToolCallAgent,
    obs_queue: deque,
    task_description: str,
    step_counter: int,
    max_steps: int,
    history: Optional[List[dict]] = None,
    log_file=None,
) -> Tuple[AgentDecision, str, List[dict]]:
    """
    Run the LLM Head to check task progress.

    Returns (AgentDecision, raw_text, updated_history) tuple.
    """
    move_status = get_movement_status(agent)

    agent_prompt = (
        f"You monitor a VLA robot arm. Look at the image and decide.\n\n"
        f'Task: "{task_description}"\n'
        f"Steps used: {step_counter}.\n"
        f"Movement status: {move_status}\n\n"
        f"RULES:\n"
        f"- If the robot is still moving/reaching/grasping normally → just describe. NO tool call.\n"
        f"- If the robot is stuck, doing wrong action, or needs redirection → call_vla with a corrected instruction.\n"
        f"- If the robot is shaking, holding something wrong, or stuck grasping → reset_arm.\n"
        f"- If you have said \"reaching\" or \"moving toward\" for 2+ rounds in a row with no progress → call_vla with a more specific instruction (e.g. include \"close gripper and grasp\").\n"
        f"- If the task is FULLY COMPLETE → terminate. NEVER call terminate unless done.\n\n"
        f"--- Examples ---\n\n"
        f"Example 1 — task COMPLETE, bowl IS on plate:\n"
        f"<tool_call>\n"
        f'{{"name":"terminate","arguments":{{"reason":"bowl is centered on the plate, task done"}}}}\n'
        f"</tool_call>\n\n"
        f"Example 2 — robot stuck, missed the target:\n"
        f"<tool_call>\n"
        f'{{"name":"call_vla","arguments":{{"instruction":"move the object 3cm to the right and place it on the plate","reason":"missed the target","max_vla_steps":30}}}}\n'
        f"</tool_call>\n\n"
        f"Example 3 — robot shaking, holding wrong object, or can't release:\n"
        f"<tool_call>\n"
        f'{{"name":"reset_arm","arguments":{{"reason":"gripper stuck holding object, shaking"}}}}\n'
        f"</tool_call>\n\n"
        f"Example 4 — normal progress, still moving:\n"
        f"The robot is reaching toward the bowl. Continuing.\n\n"
        f"--- End Examples ---\n\n"
        f"Current observation:"
    )

    # Collect raw images for the API call
    raw_images = []
    for obs in obs_queue:
        raw_images.append(obs["full_image"])
    for obs in obs_queue:
        raw_images.extend([obs[k] for k in obs.keys() if "wrist" in k])

    # Call agent API (Bailian Qwen-VL)
    try:
        text, history = model.vla.predict_text(
            images=raw_images,
            prompt_text=agent_prompt,
            max_new_tokens=2048,
            temperature=0.1,
            history=history,
        )
    except Exception as e:
        logger.warning(f"Agent API failed: {e}. Defaulting to continue.")
        log_message(f"[Agent] API error: {e}. Continue.", log_file)
        return AgentDecision(action="continue", reason=f"api error: {e}"), "", history

    log_message(f"[Agent] Raw output: {text}", log_file)

    # Parse
    if not ToolCallParser.has_tool_call(text):
        reasoning = ToolCallParser.extract_reasoning(text)
        log_message(f"[Agent] Decision: continue — {reasoning}", log_file)
        return AgentDecision(action="continue", reason=reasoning or "normal progress"), text, history

    tool_calls = ToolCallParser.parse(text)
    if not tool_calls:
        return AgentDecision(action="continue", reason="parse failed"), text, history

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
        return decision, text, history
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
        return decision, text, history
    elif name == "reset_arm":
        decision = AgentDecision(
            action="reset_arm",
            reason=args.get("reason", ""),
        )
        log_message(f"[Agent] Decision: reset_arm — {decision.reason}", log_file)
        return decision, text, history
    else:
        log_message(f"[Agent] Decision: continue — unknown tool '{name}'", log_file)
        return AgentDecision(action="continue", reason=f"unknown tool: {name}"), text, history


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

    # No step limit — Agent decides when to terminate
    max_steps = 9999

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

    # CSV for tool call records
    csv_path = pathlib.Path(args.video_out_path) / "tool_calls.csv"
    csv_f, csv_w = init_csv(str(csv_path))

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
            action_queue = deque(maxlen=NUM_ACTIONS_CHUNK)
            obs_queue = deque(maxlen=args.window_size)
            success = False
            done = False

            # ── Initial Agent Planning ────────────────────────────────
            # Build observation from first frame
            while len(obs_queue) < args.window_size:
                observation, img = prepare_observation(obs, resize_size=224)
                obs_queue.append(observation)

            plan_decision, agent_history = agent_plan(
                model=model,
                obs_queue=obs_queue,
                task_description=task_description,
                history=None,
                log_file=log_file,
            )
            current_instruction = plan_decision.instruction or task_description
            agent_step_counter = 0
            obs_queue.clear()

            # Annotate initial planning frame
            plan_lines = [
                f"AGENT PLAN | Task: {task_description[:70]}",
                f"  → {current_instruction[:100]}"
            ]
            plan_frame = draw_overlay(img, plan_lines, "yellow")
            for _ in range(int(AGENT_PAUSE_SEC * VIDEO_FPS)):
                replay_images.append(plan_frame)

            log_message(f"Starting episode {task_episodes + 1}...", log_file)
            log_message(f"[Agent] Initial instruction: {current_instruction}", log_file)

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

                # ── Annotate VLA frame ────────────────────────────────
                vla_info = f"VLA | Step {step+1} | {current_instruction[:80]}"
                img_annotated = draw_overlay(img, [vla_info], "white")
                replay_images.append(img_annotated)

                # VLA action inference
                if len(action_queue) == 0:
                    actions = get_action_state(obs_queue, current_instruction, model)
                    action_queue.extend(actions)

                obs_queue.popleft()
                action = action_queue.popleft()
                action_processed = process_action(action)

                obs, reward, lib_done, info = env.step(action_processed.tolist())

                # Record for agent
                agent.record_obs(observation)
                agent.record_action(action)

                # Environment terminated internally (time limit) → force agent check, then break
                if lib_done:
                    decision, agent_text, agent_history = agent_check(
                        model=model, agent=agent, obs_queue=obs_queue,
                        task_description=task_description, step_counter=step,
                        max_steps=max_steps, history=agent_history, log_file=log_file,
                    )
                    success = lib_done  # env internal timeout = failure
                    done = True
                    log_message(f"[Agent] Env terminated. LIBERO done={lib_done} → success={success}", log_file)
                    break

                t += 1
                step += 1
                agent_step_counter += 1

                # ── Agent check: every N steps ───────────────────────
                if agent_step_counter >= args.vla_interval and not done:
                    decision, agent_text, agent_history = agent_check(
                        model=model,
                        agent=agent,
                        obs_queue=obs_queue,
                        task_description=task_description,
                        step_counter=step,
                        max_steps=max_steps,
                        history=agent_history,
                        log_file=log_file,
                    )

                    # ── Annotate agent pause frames ───────────────────
                    agent_lines = [f"AGENT | Round {agent.agent_round} | {current_instruction[:70]}"]
                    agent_text_short = agent_text[:120].replace('\n', ' ')
                    agent_lines.append(f"  {agent_text_short}")
                    if decision.action == "new_instruction":
                        agent_lines.append(f"  TOOL: call_vla → {decision.instruction[:80]}")
                    elif decision.action == "reset_arm":
                        agent_lines.append(f"  TOOL: reset_arm — {decision.reason[:80]}")
                    elif decision.action == "terminate":
                        tag = "SUCCESS" if lib_done else "MISMATCH"
                        agent_lines.append(f"  TOOL: terminate ({tag}) — {decision.reason[:80]}")

                    agent_color = "green" if decision.action == "terminate" and lib_done else \
                                  "red" if decision.action == "terminate" else "yellow"
                    agent_frame = draw_overlay(img_annotated, agent_lines, agent_color)

                    pause_frames = int(AGENT_PAUSE_SEC * VIDEO_FPS)
                    for _ in range(pause_frames):
                        replay_images.append(agent_frame)

                    # ── CSV record ────────────────────────────────────
                    tool_name = ""
                    tool_args = ""
                    if decision.action == "new_instruction":
                        tool_name = "call_vla"
                        tool_args = decision.instruction or ""
                    elif decision.action == "reset_arm":
                        tool_name = "reset_arm"
                        tool_args = decision.reason
                    elif decision.action == "terminate":
                        tool_name = "terminate"
                        tool_args = decision.reason
                    csv_w.writerow([
                        total_episodes + 1, task_description, step, agent.agent_round,
                        "agent", agent_text_short, tool_name, tool_args, decision.action
                    ])
                    csv_f.flush()

                    if decision.action == "terminate":
                        # Agent ends the episode. Score by LIBERO's done_flag.
                        success = lib_done
                        done = True
                        log_message(
                            f"[Agent] terminate → episode end. "
                            f"LIBERO done={lib_done} → success={success}",
                            log_file
                        )
                    elif decision.action == "reset_arm":
                        agent_interventions += 1
                        log_message(f"[Agent] Reset arm: {decision.reason}", log_file)
                        # Open gripper + retreat
                        reset_inst = "open the gripper to release any object, then move the arm up and away from all objects"
                        current_instruction = reset_inst
                        model.reset(task_description=reset_inst)
                        action_queue.clear()
                        for _ in range(15):
                            if len(action_queue) == 0:
                                actions = get_action_state(obs_queue, current_instruction, model)
                                action_queue.extend(actions)
                            obs_queue.popleft()
                            a = action_queue.popleft()
                            obs, _, lib_done, _ = env.step(process_action(a).tolist())
                            while len(obs_queue) < args.window_size:
                                observation, img = prepare_observation(obs, resize_size=224)
                                obs_queue.append(observation)
                            replay_images.append(draw_overlay(img, [f"RESET | {reset_inst}"], "red"))
                            if lib_done:
                                break
                        # Re-plan after reset
                        plan_decision, agent_history = agent_plan(
                            model=model, obs_queue=obs_queue,
                            task_description=task_description,
                            history=agent_history, log_file=log_file,
                        )
                        current_instruction = plan_decision.instruction or task_description
                        model.reset(task_description=current_instruction)
                        action_queue.clear()
                        log_message(f"[Agent] Re-planned after reset: {current_instruction}", log_file)
                    elif decision.action == "new_instruction" and decision.instruction:
                        agent_interventions += 1
                        current_instruction = decision.instruction
                        model.reset(task_description=current_instruction)
                        action_queue.clear()
                        log_message(
                            f"[Agent] New instruction: {current_instruction}", log_file
                        )

                    agent_step_counter = 0

                if done:
                    break

            # Episode summary
            task_episodes += 1
            total_episodes += 1
            if success:
                task_successes += 1
                total_successes += 1

            # Save video for every episode
            task_segment = task_description.replace(" ", "_")
            suffix = "success" if success else "failure"
            imageio.mimwrite(
                pathlib.Path(args.video_out_path)
                / f"agent_{task_segment}_ep{episode_idx}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=VIDEO_FPS,
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
    csv_f.close()
    log_message(f"Tool call records saved to: {csv_path}", log_file=None)
    logger.info(f"Tool call records saved to: {csv_path}")


if __name__ == "__main__":
    tyro.cli(eval_libero_agent)
