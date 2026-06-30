"""
LIBERO tabletop scene — randomly selects from libero_spatial tasks.

The VLA was trained on LIBERO spatial tasks — it CAN follow instructions
in these scenes. Each reset picks a random BDDL for variety.
"""

import random
import numpy as np
import math
import pathlib

from libero.libero.envs import OffScreenRenderEnv
from libero.libero import get_libero_path

BDDL_DIR = pathlib.Path(get_libero_path("bddl_files")) / "libero_spatial"
BDDL_FILES = sorted(BDDL_DIR.glob("*.bddl"))


def create_scene(render_resolution: int = 512, bddl_file=None):
    """Create a LIBERO scene. Random spatial BDDL if none specified."""
    if bddl_file is None:
        bddl_file = str(random.choice(BDDL_FILES))
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file,
        camera_heights=render_resolution,
        camera_widths=render_resolution,
    )
    env.seed(random.randint(0, 10000))
    env.reset()
    return env


def get_image(env) -> np.ndarray:
    obs = env._get_observations()
    return obs["agentview_image"][::-1, ::-1]


def get_state(env) -> np.ndarray:
    obs = env._get_observations()
    pos = obs["robot0_eef_pos"]
    quat = obs["robot0_eef_quat"]
    q = np.array(quat)
    if q[3] > 1.0:
        q = q / np.linalg.norm(q)
    den = np.sqrt(1.0 - q[3] * q[3])
    ax = np.zeros(3) if math.isclose(den, 0.0) else (q[:3] * 2.0 * math.acos(q[3])) / den
    return np.concatenate([pos, ax, obs["robot0_gripper_qpos"]])
