"""Robot task geometry and execution, with backend-independent imports.

Exports are loaded on demand so extraction/retargeting does not import MuJoCo
or an IK backend. Existing execution exports keep their public names.
"""

from importlib import import_module

_EXPORT_MODULES = {
    "TaskExtractor": "task_extractor",
    "TaskExtractionError": "task_extractor",
    "extract_task": "task_extractor",
    "extract_tasks": "task_extractor",
    "CoordinateRetargeter": "coordinate_retargeter",
    "MappingConfig": "coordinate_retargeter",
    "retarget_task": "coordinate_retargeter",
    "PathInterpolation": "path_processing",
    "PathProcessingSettings": "path_processing",
    "ProcessedPath": "path_processing",
    "PathProcessor": "path_processing",
    "process_path": "path_processing",
    "WaypointConstructionSettings": "waypoint_builder",
    "WaypointBuilder": "waypoint_builder",
    "build_waypoints": "waypoint_builder",
    "TabletopCloneSettings": "tabletop_scene",
    "add_tabletop_clone": "tabletop_scene",
    "command_target": "commands",
    "CartesianMotion": "action_primitives",
    "JointPresetMotion": "action_primitives",
    "RobotAction": "action_primitives",
    "JointPreset": "presets",
    "resolve_joint_preset": "presets",
    "PickPlaceSkillContext": "pick_place_skills",
    "build_pick_place_skill_registry": "pick_place_skills",
    "RobotController": "controller",
    "GripperAction": "gripper",
    "GripperDriver": "gripper",
    "GripperLogic": "gripper",
    "GripperSettings": "gripper",
    "IKSettings": "inverse_kinematics",
    "IKSolver": "inverse_kinematics",
    "ModelBindings": "model",
    "RobotProfile": "model",
    "MuJoCoAdapter": "simulation",
    "RobotIO": "simulation",
    "ExecutionSettings": "state_machine",
    "SkillExecutor": "state_machine",
}

__all__ = list(_EXPORT_MODULES)


def __getattr__(name):
    if name not in _EXPORT_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{_EXPORT_MODULES[name]}", __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
