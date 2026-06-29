"""
Scene wrapper using LIBERO's OffScreenRenderEnv (known-compatible MuJoCo + robot).
No BDDL tasks — just one fixed tabletop scene, reusable for any instruction.
"""

import numpy as np
from libero.libero.envs import OffScreenRenderEnv
from libero.libero import get_libero_path
import pathlib

# Use LIBERO's built-in kitchen scene (libero_spatial task 0)
DEFAULT_BDDL = (
    pathlib.Path(get_libero_path("bddl_files"))
    / "libero_spatial"
    / "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate.bddl"
)


def create_scene(render_resolution: int = 512, bddl_file=None):
    """Create a LIBERO OffScreenRenderEnv with a fixed tabletop scene."""
    bddl = bddl_file or str(DEFAULT_BDDL)
    env = OffScreenRenderEnv(
        bddl_file_name=bddl,
        camera_heights=render_resolution,
        camera_widths=render_resolution,
    )
    env.seed(42)
    return env


def get_image(env) -> np.ndarray:
    """RGB render from agentview camera (rotated 180° to match training)."""
    obs = env.get_observation()
    return obs["agentview_image"][::-1, ::-1]


def get_state(env) -> np.ndarray:
    """7D state: eef_pos(3) + eef_quat→axisangle(3) + gripper(1)."""
    import math
    obs = env.get_observation()
    pos = obs["robot0_eef_pos"]
    quat = obs["robot0_eef_quat"]

    # quat2axisangle
    q = np.array(quat)
    if q[3] > 1.0:
        q = q / np.linalg.norm(q)
    den = np.sqrt(1.0 - q[3] * q[3])
    axisangle = np.zeros(3) if math.isclose(den, 0.0) else (q[:3] * 2.0 * math.acos(q[3])) / den

    gripper = obs["robot0_gripper_qpos"]
    return np.concatenate([pos, axisangle, gripper])
