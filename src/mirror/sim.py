"""MuJoCo sink: the fast loop that chases the slow loop's target.

THE TWO-LOOP SPLIT, AND WHY IT IS NOT OPTIONAL
----------------------------------------------
Perception runs at ~31 Hz. The arm's position actuators run at the model's 500
Hz timestep with `kp = 998.22`, `kv = 2.731`, giving `zeta = 0.250` and a 17.9 Hz
resonance. Feeding a 31 Hz staircase into that rings it.

Measured earlier on this exact model: commanding a 30 Hz staircase produced
1.338 deg of peak-to-peak tracking ripple; commanding every 2 ms step produced
0.025 deg. A 200x difference, from the command RATE alone.

So the fast loop must not merely hold the slow loop's target between updates --
it must INTERPOLATE toward it. That is what the ECE 4560 students' architecture
did (their terminal printed `Detect Freq: 31 Hz` against `Arm Freq: 720-930 Hz`)
and it is the single cheapest improvement available.

`hold=True` is kept as a deliberately available comparison, because a claim
about interpolation is worth nothing without the staircase measured beside it.

WHAT IS COMPARED, AND WHY IT IS THE JOINTS
------------------------------------------
Tracking error is measured in JOINT space (commanded `ctrl` versus achieved
`qpos`) and the tool pose is derived from `qpos` through our own FK -- the one
already verified against `mj_forward` to 8.98e-09 m. Reading a MuJoCo body pose
instead would require matching our tool frame to a body frame, and the tool sits
103.4 mm along the wrist axis from the gripper body with a 90 deg twist. Deriving
it removes that ambiguity entirely.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, Self

import numpy as np

from .kinematics import forward_kinematics, gripper_rad
from .kinematics.limits import JOINTS
from .types import Pose

if TYPE_CHECKING:
    from .landmarks import F64

#: Every joint in the model, in the order MuJoCo holds them. The five arm
#: joints then the jaw -- matching `JOINTS` plus `gripper`.
MODEL_JOINTS: tuple[str, ...] = (*JOINTS, "gripper")


@dataclass(frozen=True, slots=True)
class SimStep:
    """One slow-loop tick: what was asked, what the arm did."""

    commanded: dict[str, float]
    achieved: dict[str, float]
    #: Largest per-joint difference at the END of the tick, degrees.
    joint_error_deg: float
    #: Largest per-joint difference at any 2 ms substep, degrees. This is the
    #: number the ripple lives in; the settled error hides it.
    peak_joint_error_deg: float
    #: Tool pose implied by the commanded joints, and by the achieved ones.
    commanded_pose: Pose
    achieved_pose: Pose
    position_error_m: float
    orientation_error_deg: float
    sim_time_s: float


class SimArm:
    """The SO-101 in MuJoCo, driven at the model's own timestep.

    Use as a context manager, or call `close()`. `set_target` is the slow loop;
    `advance` runs the fast one.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        interpolate: bool = True,
        start: dict[str, float] | None = None,
    ) -> None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"model not found at {model_path}. Run `make model`.")
        import mujoco  # noqa: PLC0415

        self._mujoco = mujoco
        self._model = mujoco.MjModel.from_xml_path(str(model_path))
        self._data = mujoco.MjData(self._model)
        self._interpolate = interpolate

        self._index = {
            name: mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in MODEL_JOINTS
        }
        missing = [n for n, i in self._index.items() if i < 0]
        if missing:
            raise ValueError(f"model has no joints named {missing}")

        self._target = {j: 0.0 for j in JOINTS}
        self._target_jaw = 0.0
        self._previous = dict(self._target)
        self._previous_jaw = 0.0
        self._renderer: Any | None = None
        if start is not None:
            self.teleport(start)

    # --- properties ---------------------------------------------------------
    @property
    def timestep_s(self) -> float:
        return float(self._model.opt.timestep)

    @property
    def time_s(self) -> float:
        return float(self._data.time)

    @property
    def joints(self) -> dict[str, float]:
        """Actual joint angles from the simulator, degrees."""
        return {
            name: float(np.degrees(self._data.qpos[self._index[name]]))
            for name in JOINTS
        }

    def pose(self) -> Pose:
        """Tool pose implied by the ACTUAL joint angles, via our verified FK."""
        return _pose_of(self.joints)

    # --- the loops ----------------------------------------------------------
    def teleport(self, joints: dict[str, float], jaw: float = 0.0) -> None:
        """Place the arm instantly. Startup and reset only -- never mid-motion."""
        for name in JOINTS:
            self._data.qpos[self._index[name]] = np.radians(joints[name])
        self._data.qpos[self._index["gripper"]] = gripper_rad(jaw)
        self._data.qvel[:] = 0.0
        self._set_ctrl(joints, jaw)
        self._target = dict(joints)
        self._previous = dict(joints)
        self._target_jaw = self._previous_jaw = jaw
        self._mujoco.mj_forward(self._model, self._data)

    def set_target(self, joints: dict[str, float], jaw: float) -> None:
        """The slow loop. Hands the fast loop somewhere to go.

        `_previous` is NOT updated here -- `advance` consumes it. Shifting it
        here as well double-counted, and a convergence test caught the result:
        calling `advance` repeatedly without a new target restarted the ramp
        from the OLD target every time, so the command oscillated and the arm
        settled 9.2014 deg away instead of arriving.
        """
        self._target = {j: float(joints[j]) for j in JOINTS}
        self._target_jaw = float(jaw)

    def advance(self, duration_s: float) -> SimStep:
        """The fast loop. Steps at the model's timestep until `duration_s` elapses.

        With `interpolate=True` the commanded `ctrl` ramps from the previous
        target to the current one across the interval, so the actuators see a
        continuous reference rather than a 31 Hz staircase.
        """
        steps = max(1, int(round(duration_s / self.timestep_s)))
        peak = 0.0
        for index in range(1, steps + 1):
            blend = index / steps if self._interpolate else 1.0
            joints = {
                j: self._previous[j] + blend * (self._target[j] - self._previous[j])
                for j in JOINTS
            }
            jaw = self._previous_jaw + blend * (self._target_jaw - self._previous_jaw)
            self._set_ctrl(joints, jaw)
            self._mujoco.mj_step(self._model, self._data)
            actual = self.joints
            peak = max(peak, max(abs(actual[j] - joints[j]) for j in JOINTS))

        # The ramp is now spent: a further `advance` with no new target must
        # hold at the target rather than sweep to it again.
        self._previous = dict(self._target)
        self._previous_jaw = self._target_jaw

        achieved = self.joints
        commanded_pose = _pose_of(self._target)
        achieved_pose = _pose_of(achieved)
        metres, degrees = achieved_pose.distance_to(commanded_pose)
        return SimStep(
            commanded=dict(self._target),
            achieved=achieved,
            joint_error_deg=max(abs(achieved[j] - self._target[j]) for j in JOINTS),
            peak_joint_error_deg=peak,
            commanded_pose=commanded_pose,
            achieved_pose=achieved_pose,
            position_error_m=metres,
            orientation_error_deg=degrees,
            sim_time_s=self.time_s,
        )

    # --- rendering ----------------------------------------------------------
    def renderer(self, width: int = 640, height: int = 480) -> Any:
        """An offscreen renderer.

        Capped at the model's own offscreen framebuffer, which `scene.xml` does
        not declare, so MuJoCo's 640x480 default applies. Asking for more is
        silently truncated rather than refused, so the default here matches what
        the model can actually give.
        """
        import mujoco  # noqa: PLC0415

        if self._renderer is None:
            self._renderer = mujoco.Renderer(self._model, height=height, width=width)
        return self._renderer

    def frame(self, renderer: Any, camera: int | str = -1) -> F64:
        renderer.update_scene(self._data, camera=camera)
        return np.asarray(renderer.render())

    # --- internals ----------------------------------------------------------
    def _set_ctrl(self, joints: dict[str, float], jaw: float) -> None:
        for name in JOINTS:
            self._data.ctrl[self._index[name]] = np.radians(joints[name])
        self._data.ctrl[self._index["gripper"]] = gripper_rad(jaw)

    def close(self) -> None:
        """Release the renderer, if one was made.

        `MjModel` and `MjData` hold no OS handle that needs freeing -- they are
        ordinary Python objects. The RENDERER does hold a GL context, so that is
        the only thing there is to close. An earlier version set `_data = None`,
        which freed nothing and turned any later call into an opaque
        AttributeError instead of a clear one.
        """
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def _pose_of(joints: dict[str, float]) -> Pose:
    position, rotation = forward_kinematics({**joints, "gripper": 0.0})
    return Pose(position=position, rotation=rotation)


__all__ = ["MODEL_JOINTS", "SimArm", "SimStep"]
