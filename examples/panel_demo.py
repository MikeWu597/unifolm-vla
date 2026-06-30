"""
UnifoLM-VLA-OpenVLA: browser-controlled VLA demo with OpenVLA-OFT.

Start: python examples/panel_demo.py
Open:  http://localhost:8778
"""

import argparse
import logging
import os
import time
from collections import deque

import numpy as np
import torch
from PIL import Image

# torch 2.6+ compat
_orig_load = torch.load
torch.load = lambda *a, **kw: _orig_load(*a, **{'weights_only': False, **kw})

from unifolm_vla.panel.scene import create_scene, get_image, get_state

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10")
    parser.add_argument("--port", type=int, default=8778)
    parser.add_argument("--window", type=int, default=2)
    args = parser.parse_args()

    # ── Load OpenVLA ──────────────────────────────────────────────────
    logger.info("Loading OpenVLA-OFT...")
    from unifolm_vla.openvla.wrapper import OpenVLAWrapper
    vla = OpenVLAWrapper(model_id=args.model)
    logger.info("OpenVLA loaded.")

    # ── Create scene ───────────────────────────────────────────────────
    logger.info("Creating LIBERO scene...")
    env = create_scene(render_resolution=512)
    env.reset()

    # ── Start server ──────────────────────────────────────────────────
    from unifolm_vla.panel.server import app as server_app, push_frame, \
        next_command, set_status, get_status
    import threading, uvicorn

    def run_server():
        uvicorn.run(server_app, host="0.0.0.0", port=args.port, log_level="info")

    threading.Thread(target=run_server, daemon=True).start()
    time.sleep(2)
    logger.info(f"Server: http://localhost:{args.port}")

    # ── Main loop ─────────────────────────────────────────────────────
    obs_queue = deque(maxlen=args.window)
    current_instruction = None
    DUMMY = [0.0] * 6 + [-1.0]

    while True:
        cmd = next_command()
        if cmd is not None:
            if cmd == "__STOP__":
                current_instruction = None
                obs_queue.clear()
                set_status(running=False, instruction="")
                continue
            if cmd == "__RESET__":
                env.reset()
                obs_queue.clear()
                set_status(running=False, instruction="")
                logger.info("Scene reset")
                continue
            if cmd == current_instruction:
                continue
            current_instruction = cmd
            obs_queue.clear()
            set_status(running=True, instruction=cmd, step=0, active_since=time.time())
            logger.info(f"Command: {cmd}")

        if current_instruction:
            s = get_status()
            # Step env (use VLA action if enough obs collected, else dummy)
            if len(obs_queue) >= args.window:
                # Build prompt image from latest observation
                obs_data = obs_queue[-1]
                pil_img = Image.fromarray(obs_data["full_image"]).convert("RGB")
                action = vla.predict_action(pil_img, current_instruction)
                action[-1] *= -1.0   # invert gripper
                try:
                    obs, _, done, _ = env.step(action.tolist())
                except ValueError:
                    env.reset()
                    obs, _, done, _ = env.step(action.tolist())
                if done:
                    env.reset()
            else:
                try:
                    obs, _, _, _ = env.step(np.array(DUMMY))
                except ValueError:
                    env.reset()
                    obs, _, _, _ = env.step(np.array(DUMMY))

            img = get_image(env, obs)
            wrist_img = obs.get("robot0_eye_in_hand_image", img)
            if isinstance(wrist_img, np.ndarray):
                wrist_img = wrist_img[::-1, ::-1]
            state = get_state(env, obs)
            push_frame(img)
            obs_queue.append({"full_image": img, "wrist_image": wrist_img, "state": state})
            set_status(step=s["step"] + 1)
            time.sleep(0.02)
        else:
            # Idle
            try:
                obs, _, _, _ = env.step(np.array(DUMMY))
            except ValueError:
                env.reset()
                obs, _, _, _ = env.step(np.array(DUMMY))
            img = get_image(env, obs)
            push_frame(img)
            time.sleep(0.1)


if __name__ == "__main__":
    main()
