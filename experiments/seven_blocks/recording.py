"""Three synchronized videos sampled every 10 physics steps (25 Hz)."""

import json
import time

import imageio_ffmpeg
import numpy as np

from env.camera_manager.capture.render_sync import wait_for_latest_cameras


class VideoRecorder:
    def __init__(self, env, directory):
        self.env = env
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=False)
        self.tick = 0
        self.frames = 0
        self.phase = "initial"
        self.writers = {}
        self.journal = (directory / "frames.jsonl").open("w")
        self.capture()

    def step(self):
        self.tick += 1
        if self.tick % 10 == 0:
            self.capture()

    def capture(self):
        wait_for_latest_cameras(self.env)
        for name, data in zip(self.env.camera_manager.camera_names[0], self.env.capture_manager.step(), strict=True):
            rgb = np.ascontiguousarray(data["rgb"][0]["data"][..., :3], dtype=np.uint8)
            if name not in self.writers:
                writer = imageio_ffmpeg.write_frames(
                    str(self.directory / f"{name}.mp4"), (rgb.shape[1], rgb.shape[0]),
                    fps=1 / (10 * self.env.dt), codec="libx264", pix_fmt_out="yuv420p",
                    output_params=["-crf", "18", "-preset", "fast", "-threads", "2", "-movflags", "+faststart"],
                )
                writer.send(None)
                self.writers[name] = writer
            self.writers[name].send(rgb)
        row = {"frame": self.frames, "physics_tick": self.tick,
               "simulation_seconds": self.tick * self.env.dt, "wall_unix": time.time(),
               "phase": self.phase, "state": self.env.snapshot()}
        self.journal.write(json.dumps(row) + "\n")
        self.journal.flush()
        self.frames += 1

    def close(self):
        for writer in self.writers.values():
            writer.close()
        self.journal.close()
        report = {"timebase": "simulation", "fps": 1 / (10 * self.env.dt),
                  "frames": self.frames, "physics_steps": self.tick,
                  "duration_seconds": self.frames * 10 * self.env.dt,
                  "planning_wall_time_included": False,
                  "videos": [f"{name}.mp4" for name in self.writers]}
        (self.directory / "recording.json").write_text(json.dumps(report, indent=2) + "\n")
        return report
