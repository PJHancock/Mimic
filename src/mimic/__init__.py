"""
Mimic: Learning Manipulation Skills from Human Demonstration

A system that learns manipulation skills from human video demonstrations
and executes them with a simulated Franka Panda robot.
"""

__version__ = "0.1.0"

from importlib import import_module

__all__ = [
    "common",
    "data_pipeline",
    "vision",
    "tracking",
    "robot",
    "integration",
]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = import_module(f".{name}", __name__)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
