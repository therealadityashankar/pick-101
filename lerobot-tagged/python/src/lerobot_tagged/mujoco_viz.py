"""mujoco_viz.py — Represent tag-detected positions in a MuJoCo scene.

Provides helpers to place a MuJoCo body (e.g. a cube) at a position derived
from tag detection, and to render the scene.

Requires the ``mujoco`` optional dependency:
    pip install "lerobot-tagged[mujoco]"

Example
-------
>>> import mujoco
>>> from lerobot_tagged.mujoco_viz import TaggedScene
>>>
>>> scene = TaggedScene.from_xml_path("models/so101/lift_cube.xml")
>>> scene.set_object_position("cube", x=0.15, y=0.0, z=0.015)
>>> rgb = scene.render(camera="topdown")
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


class TaggedScene:
    """Thin wrapper around a MuJoCo model for tag-driven visualisation.

    Parameters
    ----------
    model:
        A ``mujoco.MjModel`` instance.
    """

    def __init__(self, model, data=None):
        try:
            import mujoco
        except ImportError as e:
            raise ImportError(
                "MuJoCo is required for TaggedScene. "
                'Install it with: pip install "lerobot-tagged[mujoco]"'
            ) from e

        self._mujoco = mujoco
        self.model = model
        self.data = data if data is not None else mujoco.MjData(model)
        mujoco.mj_forward(model, self.data)

    @classmethod
    def from_xml_path(cls, xml_path: str) -> "TaggedScene":
        """Load a MuJoCo model from an XML file."""
        try:
            import mujoco
        except ImportError as e:
            raise ImportError(
                'Install mujoco with: pip install "lerobot-tagged[mujoco]"'
            ) from e
        model = mujoco.MjModel.from_xml_path(xml_path)
        return cls(model)

    @classmethod
    def from_xml_string(cls, xml: str) -> "TaggedScene":
        """Load a MuJoCo model from an XML string."""
        try:
            import mujoco
        except ImportError as e:
            raise ImportError(
                'Install mujoco with: pip install "lerobot-tagged[mujoco]"'
            ) from e
        model = mujoco.MjModel.from_xml_string(xml)
        return cls(model)

    def set_object_position(
        self,
        body_name: str,
        x: float,
        y: float,
        z: float,
        quat: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    ) -> None:
        """Move a free-joint body to (x, y, z) with optional quaternion.

        Parameters
        ----------
        body_name:
            Name of the MuJoCo body to move. The body must have a free joint.
        x, y, z:
            Target position in world metres.
        quat:
            Quaternion (w, x, y, z). Defaults to identity (no rotation).
        """
        mujoco = self._mujoco
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise ValueError(f"Body '{body_name}' not found in model.")

        # Find the free joint associated with this body
        jnt_adr = self.model.body_jntadr[body_id]
        if jnt_adr < 0:
            raise ValueError(f"Body '{body_name}' has no joint.")

        qpos_adr = self.model.jnt_qposadr[jnt_adr]
        self.data.qpos[qpos_adr:qpos_adr + 3] = [x, y, z]
        self.data.qpos[qpos_adr + 3:qpos_adr + 7] = quat
        self.data.qvel[qpos_adr:qpos_adr + 6] = 0
        mujoco.mj_forward(self.model, self.data)

    def render(
        self,
        camera: str = "topdown",
        width: int = 600,
        height: int = 600,
    ) -> np.ndarray:
        """Render the scene and return an RGB image.

        Parameters
        ----------
        camera:
            Name of the MuJoCo camera to render from.
        width, height:
            Output image dimensions in pixels.

        Returns
        -------
        np.ndarray
            (H, W, 3) uint8 RGB image.
        """
        mujoco = self._mujoco
        renderer = mujoco.Renderer(self.model, height=height, width=width)
        renderer.update_scene(self.data, camera=camera)
        rgb = renderer.render().copy()
        renderer.close()
        return rgb
