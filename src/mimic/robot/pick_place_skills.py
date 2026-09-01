"""Default composite-skill preset for tabletop pick and place."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from mimic.common.types import PickPlaceWaypoints
from mimic.robot.action_primitives import CartesianMotion, JointPresetMotion, RobotAction
from mimic.robot.gripper import GripperAction
from mimic.robot.presets import JointPreset
from mimic.skills import SkillCatalog, StateDecision
from mimic.skills.registry import SkillRegistry


@dataclass(frozen=True)
class PickPlaceSkillContext:
    waypoints: PickPlaceWaypoints
    home: JointPreset
    placement_approach_clearance_m: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.placement_approach_clearance_m, bool)
            or not np.isfinite(self.placement_approach_clearance_m)
            or self.placement_approach_clearance_m <= 0
        ):
            raise ValueError("Placement approach clearance must be finite and positive")


def _idle(decision: StateDecision, context: PickPlaceSkillContext) -> tuple[RobotAction, ...]:
    # No arm or gripper command: IDLE preserves the preceding measured state.
    return ()


def _hover(decision: StateDecision, context: PickPlaceSkillContext) -> tuple[RobotAction, ...]:
    variant = decision.transition.variant if decision.transition is not None else None
    if variant == "TO_GRASP":
        return (
            CartesianMotion("MOVE_TO_GRASP_HOVER", context.waypoints.approach, GripperAction.OPEN),
        )
    if variant in ("TO_HOME", "TO_HOME_ABORT"):
        return (
            JointPresetMotion(
                "MOVE_TO_HOME",
                context.home.preset_id,
                context.home.joint_positions,
                GripperAction.HOLD,
            ),
        )
    raise ValueError(f"HOVER requires a contextual transition variant, received {variant!r}")


def _grasp(decision: StateDecision, context: PickPlaceSkillContext) -> tuple[RobotAction, ...]:
    # A missed continuation HOVER is an extraction exception, not a skipped approach.
    approach = ()
    variant = decision.transition.variant if decision.transition is not None else None
    if variant == "CONTINUATION_REGRASP":
        approach = (
            CartesianMotion(
                "MOVE_TO_GRASP_HOVER", context.waypoints.approach, GripperAction.OPEN
            ),
        )
    return approach + (
        CartesianMotion("DESCEND", context.waypoints.grasp, GripperAction.OPEN),
        CartesianMotion("CLOSE", context.waypoints.grasp, GripperAction.CLOSE),
    )


def _carry(decision: StateDecision, context: PickPlaceSkillContext) -> tuple[RobotAction, ...]:
    return (
        CartesianMotion("LIFT", context.waypoints.lift, GripperAction.HOLD),
        *(
            CartesianMotion("FOLLOW_PATH", pose, GripperAction.HOLD)
            for pose in context.waypoints.path
        ),
    )


def _release(decision: StateDecision, context: PickPlaceSkillContext) -> tuple[RobotAction, ...]:
    lower = context.waypoints.lower
    placement_approach = replace(
        lower,
        position=(
            lower.position[0],
            lower.position[1],
            lower.position[2] + context.placement_approach_clearance_m,
        ),
    )
    return (
        CartesianMotion("PLACE_APPROACH", placement_approach, GripperAction.HOLD),
        CartesianMotion("LOWER", context.waypoints.lower, GripperAction.HOLD),
        CartesianMotion("OPEN", context.waypoints.lower, GripperAction.OPEN),
    )


def build_pick_place_skill_registry(
    catalog: SkillCatalog,
) -> SkillRegistry[PickPlaceSkillContext, RobotAction]:
    """Bind the default catalog to replaceable composite-skill implementations."""
    return SkillRegistry(
        catalog,
        {
            "idle": _idle,
            "hover": _hover,
            "grasp": _grasp,
            "carry": _carry,
            "release": _release,
        },
    )
