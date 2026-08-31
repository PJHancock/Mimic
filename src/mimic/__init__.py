"""
Mimic: Learning Manipulation Skills from Human Demonstration

A system that learns manipulation skills from human video demonstrations
and executes them with a simulated Franka Panda robot.
"""

__version__ = "0.1.0"

from . import common, data_pipeline, integration, robot, tracking, vision

__all__ = [
    "common",
    "data_pipeline",
    "vision",
    "tracking",
    "robot",
    "integration",
]
