"""Render the existing fixed Panda verification fixture without changing it."""

import importlib.util
import sys
from pathlib import Path

import cv2
import imageio.v2 as imageio
import mujoco

from mimic.robot.simulation import MuJoCoAdapter as BaseMuJoCoAdapter
from mimic.robot.state_machine import SkillExecutor as BaseSkillExecutor


REPO = Path("/Users/joshuamcconkie/1 PROJECTS/competitions/BYU HireReadyHack 8-2026/Mimic")
OUTPUT = REPO / "outputs" / "simulations" / "panda_pick_place_example_20260831_v2"
VIDEO = OUTPUT / "panda_pick_place.mp4"
FPS = 30
ACTIVE_SKILL = {"name": "INITIALIZE"}
ADAPTERS = []


class RenderingAdapter(BaseMuJoCoAdapter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.renderer = mujoco.Renderer(self.model, height=480, width=640)
        self.camera = mujoco.MjvCamera()
        self.camera.lookat[:] = (0.49, 0.05, 0.17)
        self.camera.distance = 1.15
        self.camera.azimuth = 132
        self.camera.elevation = -25
        self.writer = imageio.get_writer(
            VIDEO,
            fps=FPS,
            codec="libx264",
            quality=8,
            pixelformat="yuv420p",
        )
        self.next_frame_s = 0.0
        self.frame_count = 0
        self.closed = False
        ADAPTERS.append(self)

    def _write_frame(self):
        self.renderer.update_scene(self.data, camera=self.camera)
        frame = self.renderer.render().copy()
        label = f"Skill: {ACTIVE_SKILL['name']}"
        timestamp = f"Simulation time: {self.data.time:05.2f} s"
        cv2.rectangle(frame, (18, 18), (445, 96), (248, 248, 248), -1)
        cv2.putText(
            frame,
            "Franka Panda pick-and-place",
            (34, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"{label}  |  {timestamp}",
            (34, 82),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (30, 30, 30),
            1,
            cv2.LINE_AA,
        )
        self.writer.append_data(frame)
        self.frame_count += 1

    def reset(self, keyframe):
        state = super().reset(keyframe)
        for _ in range(FPS):
            self._write_frame()
        return state

    def advance(self, duration_s):
        super().advance(duration_s)
        if self.data.time + 1e-9 >= self.next_frame_s:
            self._write_frame()
            self.next_frame_s += 1 / FPS

    def close(self):
        if self.closed:
            return
        ACTIVE_SKILL["name"] = "COMPLETE"
        for _ in range(FPS):
            self._write_frame()
        self.writer.close()
        self.renderer.close()
        self.closed = True


class RenderingSkillExecutor(BaseSkillExecutor):
    def __init__(self, controller, settings, record=None, *args, **kwargs):
        original_record = record or (lambda event: None)

        def recording_with_skill(event):
            if event.get("event") == "transition":
                ACTIVE_SKILL["name"] = event.get("skill", event.get("phase", "RUNNING"))
            original_record(event)

        super().__init__(controller, settings, recording_with_skill, *args, **kwargs)


def main():
    fixture_path = REPO / "scripts" / "verify_panda.py"
    spec = importlib.util.spec_from_file_location("rendered_verify_panda", fixture_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.MuJoCoAdapter = RenderingAdapter
    module.SkillExecutor = RenderingSkillExecutor
    sys.argv = [str(fixture_path), "--output", str(OUTPUT)]
    result = 1
    try:
        result = module.main()
    finally:
        for adapter in ADAPTERS:
            adapter.close()
    print(f"video={VIDEO}")
    print(f"frames={sum(adapter.frame_count for adapter in ADAPTERS)}")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
