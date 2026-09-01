"""Saved arm configurations are named, complete, and model-validated."""

from __future__ import annotations

import pytest

from mimic.robot.presets import resolve_joint_preset

pytest_plugins = ("tests.test_robot.execution_fixtures",)


def test_explicit_preset_requires_every_named_arm_joint(small_robot) -> None:
    bindings, _, _ = small_robot
    preset = resolve_joint_preset(
        bindings,
        "home",
        joint_positions={"slide_x": 0.2, "slide_z": -0.1},
    )
    assert preset.joint_positions == {"slide_x": 0.2, "slide_z": -0.1}
    with pytest.raises(ValueError, match="preset/profile mismatch"):
        resolve_joint_preset(bindings, "home", joint_positions={"slide_x": 0.2})


def test_preset_rejects_multiple_sources(small_robot) -> None:
    bindings, _, _ = small_robot
    with pytest.raises(ValueError, match="exactly one"):
        resolve_joint_preset(
            bindings,
            "home",
            keyframe="home",
            joint_positions={"slide_x": 0.2, "slide_z": -0.1},
        )


def test_panda_home_keyframe_resolves_only_named_arm_joints(panda) -> None:
    bindings, _, _, _ = panda
    preset = resolve_joint_preset(bindings, "home", keyframe="home")
    assert tuple(preset.joint_positions) == bindings.profile.arm_joints
    assert not any("finger" in name for name in preset.joint_positions)
