"""Verify continuous recordings and render wall-clock and accelerated head views."""

import argparse
import bisect
import hashlib
import json
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def render(directory):
    video_dir = directory / "videos"
    recording = json.loads((video_dir / "recording.json").read_text())
    timeline = [json.loads(line) for line in (video_dir / "frames.jsonl").read_text().splitlines()]
    events = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
    assert len(timeline) == recording["frames"]
    assert [row["frame"] for row in timeline] == list(range(len(timeline)))
    timestamps = [row["wall_unix"] for row in timeline]
    assert all(b >= a for a, b in zip(timestamps, timestamps[1:]))
    final = next(event for event in reversed(events) if event["event"] == "observation")
    started, ended = timestamps[0], final["wall_unix"]
    duration = ended - started
    assert duration > 0
    speed = min((1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64), key=lambda x: abs(duration / x - 150))
    reports = []
    for path in sorted(video_dir.glob("*.mp4")):
        reader = imageio_ffmpeg.read_frames(str(path))
        meta = next(reader)
        frames = sum(1 for _ in reader)
        assert frames == len(timeline)
        reports.append({"path": str(path.relative_to(directory)), "frames": frames,
                        "fps": meta["fps"], "full_decode_passed": True,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    assert len(reports) == 3
    # Keep every original physics-sampled video. The derived wall-clock views
    # hold the prior captured image while physics is paused for model/planner work.
    for factor, fps, name in [(1, 10, "全过程_现实时间_1倍速.mp4"), (speed, 30, f"全过程_观看版_{speed}倍速.mp4")]:
        reader = imageio_ffmpeg.read_frames(str(video_dir / "cam_head.mp4"))
        meta = next(reader)
        width, height = meta["size"]
        writer = imageio_ffmpeg.write_frames(str(directory / name), (width, height + 48), fps=fps,
                                             codec="libx264", pix_fmt_out="yuv420p",
                                             output_params=["-crf", "20", "-preset", "fast", "-threads", "2", "-movflags", "+faststart"])
        writer.send(None)
        frame_index = 0
        current = next(reader)
        final_rgb = Image.open(directory / Path(final["images"]["cam_head"]).name).convert("RGB")
        output_frames = int(np.ceil(duration / factor * fps)) + 1
        for frame in range(output_frames):
            wall = min(ended, started + frame / fps * factor)
            target = min(bisect.bisect_right(timestamps, wall) - 1, len(timeline) - 1)
            while frame_index < target:
                current = next(reader)
                frame_index += 1
            canvas = Image.new("RGB", (width, height + 48), "#101723")
            canvas.paste(final_rgb if wall >= ended else Image.frombytes("RGB", (width, height), current), (0, 0))
            elapsed = wall - started
            label = f"WALL TIME {int(elapsed)//60:02d}:{int(elapsed)%60:02d} | {factor}x | {timeline[target]['phase']}"
            ImageDraw.Draw(canvas).text((12, height + 15), label, fill="white", font=ImageFont.load_default(size=17))
            writer.send(np.asarray(canvas))
        writer.close()
        reader.close()
        check = imageio_ffmpeg.read_frames(str(directory / name))
        next(check)
        assert sum(1 for _ in check) == output_frames
        reports.append({"path": name, "frames": output_frames, "fps": fps, "speed":factor,
                        "full_decode_passed":True,
                        "sha256":hashlib.sha256((directory/name).read_bytes()).hexdigest()})
    manifest = {"passed":True,"wall_seconds":duration,"view_speed":speed,
                "timebase":"wall clock from first recorded scene to final observation",
                "idle_policy":"hold previous physical frame while physics is paused",
                "original_videos_timebase":"25fps simulation-time sampling, all original frames retained",
                "videos":reports}
    (directory/"video_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(manifest,ensure_ascii=False),flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path)
    render(parser.parse_args().directory)
