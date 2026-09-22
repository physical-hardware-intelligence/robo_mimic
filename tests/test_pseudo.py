"""Pseudo action labels and their error budget. No camera, no arm."""

from __future__ import annotations

import numpy as np
import pytest

from robo_mimic.arm import READY
from robo_mimic.kinematics import forward_kinematics
from robo_mimic.landmarks import HandLandmarks
from robo_mimic.project import plane_basis
from robo_mimic.pseudo import (
    JAW_LEVER_M,
    ErrorBudget,
    PseudoAction,
    decompose,
    extract,
    rigid_offset,
    structural_loss_deg,
)


def _rotation(yaw: float = 0.0, pitch: float = 0.0) -> np.ndarray:
    cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
    return np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]]) @ np.array(
        [[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]]
    )


class TestJawLever:
    def test_a_small_rotation_is_roughly_lever_times_radians(self) -> None:
        """Sanity on the deg->mm conversion: 10 deg at 58.4 mm is ~10.2 mm."""
        assert ErrorBudget.at_jaw_mm(10.0) == pytest.approx(10.19, abs=0.05)

    def test_zero_rotation_is_zero_millimetres(self) -> None:
        assert ErrorBudget.at_jaw_mm(0.0) == 0.0

    def test_it_is_a_chord_not_an_arc(self) -> None:
        """At 180 deg the jaw ends up one diameter away, not pi*r."""
        assert ErrorBudget.at_jaw_mm(180.0) == pytest.approx(2 * JAW_LEVER_M * 1000)


class TestStructuralLoss:
    def test_an_axis_lying_in_the_plane_costs_nothing(self) -> None:
        """The arm's own tool axis is in-plane by construction, so 0 deg."""
        _, tool = forward_kinematics(READY)
        pan = np.radians(READY["shoulder_pan"])
        assert structural_loss_deg(tool, float(pan)) == pytest.approx(0.0, abs=1e-6)

    def test_an_axis_along_the_normal_costs_everything(self) -> None:
        normal, _ = plane_basis(0.0)
        second = np.array([0.0, 0.0, 1.0])
        rotation = np.column_stack([normal, second, np.cross(normal, second)])
        assert structural_loss_deg(rotation, 0.0) == pytest.approx(90.0)

    def test_it_is_blind_to_the_parallel_antiparallel_convention(self) -> None:
        """Our hand->tool mapping is 180 deg ambiguous and undecided. This
        number must survive that, or it cannot be quoted until it is settled."""
        rotation = _rotation(yaw=0.7, pitch=0.3)
        assert structural_loss_deg(rotation, 0.0) == pytest.approx(
            structural_loss_deg(-rotation, 0.0)
        )

    def test_it_never_exceeds_a_right_angle(self) -> None:
        rng = np.random.default_rng(0)
        for _ in range(200):
            q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
            assert 0.0 <= structural_loss_deg(q, float(rng.uniform(-2, 2))) <= 90.0 + 1e-9


class TestExtract:
    @staticmethod
    def _hand(thumb: np.ndarray, index: np.ndarray) -> HandLandmarks:
        world = np.zeros((21, 3))
        world[9] = [0.0, 0.10, 0.0]     # middle MCP, 10 cm from the wrist
        world[5] = [0.03, 0.09, 0.0]    # index MCP
        world[17] = [-0.03, 0.09, 0.0]  # pinky MCP
        world[4], world[8] = thumb, index
        image = np.zeros((21, 3))
        image[:, :2] = 0.5
        image[9, 1] = 0.6
        return HandLandmarks(image, world, "Right", 0.99, 0)

    def test_a_pinch_reads_a_smaller_aperture_than_a_spread(self) -> None:
        pinched = extract(self._hand(np.array([0.0, 0.11, 0.0]), np.array([0.005, 0.11, 0.0])))
        spread = extract(self._hand(np.array([-0.05, 0.11, 0.0]), np.array([0.05, 0.11, 0.0])))
        assert pinched.aperture < spread.aperture

    def test_the_aperture_is_normalised_by_hand_size(self) -> None:
        """Dyna's grasp signal is a RATIO, so a big hand and a small hand
        making the same gesture must produce the same number."""
        small = self._hand(np.array([-0.02, 0.11, 0.0]), np.array([0.02, 0.11, 0.0]))
        big = HandLandmarks(small.image, small.world * 2.0, "Right", 0.99, 0)
        assert extract(small).aperture == pytest.approx(extract(big).aperture)

    def test_it_carries_the_timestamp_through(self) -> None:
        hand = self._hand(np.array([0.0, 0.11, 0.0]), np.array([0.02, 0.11, 0.0]))
        stamped = HandLandmarks(hand.image, hand.world, "Right", 0.99, 1234)
        assert extract(stamped).timestamp_ms == 1234


class TestRigidOffset:
    def test_it_recovers_a_known_offset_exactly(self) -> None:
        rng = np.random.default_rng(1)
        known = _rotation(yaw=1.1, pitch=-0.4)
        source = [np.linalg.qr(rng.normal(size=(3, 3)))[0] for _ in range(30)]
        source = [s * np.sign(np.linalg.det(s)) for s in source]
        target = [known @ s for s in source]
        offset, residual = rigid_offset(np.array(source), np.array(target))
        assert offset == pytest.approx(known, abs=1e-9)
        assert residual == pytest.approx(0.0, abs=1e-9)

    def test_the_residual_reports_per_frame_noise_the_offset_cannot_absorb(self) -> None:
        """The whole point: a CONSTANT convention error is absorbed, a
        per-frame perception error is not. Otherwise the two are confounded."""
        rng = np.random.default_rng(2)
        known = _rotation(yaw=0.5)
        source, target = [], []
        for _ in range(60):
            q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
            q = q * np.sign(np.linalg.det(q))
            source.append(q)
            target.append(known @ q @ _rotation(yaw=rng.normal(0, np.radians(5.0))))
        _, residual = rigid_offset(np.array(source), np.array(target))
        assert 1.0 < residual < 12.0, f"expected roughly 5 deg of noise, got {residual}"

    def test_mismatched_or_empty_input_is_refused(self) -> None:
        one = np.eye(3)[None]
        with pytest.raises(ValueError):
            rigid_offset(one, np.repeat(one, 2, axis=0))
        with pytest.raises(ValueError):
            rigid_offset(np.empty((0, 3, 3)), np.empty((0, 3, 3)))


class TestDecompose:
    def test_the_arms_own_orientations_cost_nothing(self) -> None:
        """Feed back poses the arm itself produced: a 5-DOF arm can obviously
        reach what a 5-DOF arm just did, so every component must be ~0."""
        position, _ = forward_kinematics(READY)
        actions = []
        for roll in (-20.0, 0.0, 20.0):
            joints = {**READY, "wrist_roll": roll}
            _, rotation = forward_kinematics(joints)
            actions.append(PseudoAction(rotation, 0.5, 0))
        budget = decompose(actions, position, READY)
        assert budget.reachable_fraction == 1.0
        assert budget.out_of_plane_deg.max() == pytest.approx(0.0, abs=1e-6)
        assert budget.orientation_deg.max() == pytest.approx(0.0, abs=1e-3)

    def test_an_out_of_plane_orientation_is_reported_as_structural(self) -> None:
        position, _ = forward_kinematics(READY)
        normal, _ = plane_basis(0.0)
        # Right-handed by construction: the third column is the cross product.
        # Stacking three axes by hand gave a reflection, which Pose refused --
        # correctly, and it would have made the measurement meaningless.
        second = np.array([0.0, 0.0, 1.0])
        rotation = np.column_stack([normal, second, np.cross(normal, second)])
        budget = decompose([PseudoAction(rotation, 0.5, 0)], position, READY)
        assert budget.out_of_plane_deg[0] == pytest.approx(90.0, abs=1.0)

    def test_the_summary_reports_millimetres_as_well_as_degrees(self) -> None:
        position, _ = forward_kinematics(READY)
        _, rotation = forward_kinematics(READY)
        summary = decompose([PseudoAction(rotation, 0.5, 0)], position, READY).summary()
        assert {"out_of_plane_p90_mm", "orientation_p90_mm"} <= summary.keys()
        assert all(np.isfinite(v) for v in summary.values())
