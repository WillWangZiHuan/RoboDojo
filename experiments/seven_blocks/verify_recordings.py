"""Decode every local acceptance video and verify frame counts and hashes."""

import argparse
import hashlib
import json
from pathlib import Path

import imageio_ffmpeg

from experiments.seven_blocks.protocol import COLORS


def verify(directory):
    acceptance = json.loads((directory / "manipulation.json").read_text())
    assert acceptance["passed"] and len(acceptance["checks"]) == 7
    assert {row["color"] for row in acceptance["checks"]} == set(COLORS)
    assert all(row["passed"] for row in acceptance["checks"])
    videos = []
    for color in COLORS:
        folder = directory / "videos" / color
        recording = json.loads((folder / "recording.json").read_text())
        expected_names = {"cam_head.mp4", "cam_left_wrist.mp4", "cam_right_wrist.mp4"}
        assert set(recording["videos"]) == expected_names
        timeline = [json.loads(line) for line in (folder / "frames.jsonl").read_text().splitlines()]
        assert len(timeline) == recording["frames"]
        assert [row["frame"] for row in timeline] == list(range(len(timeline)))
        assert [row["physics_tick"] for row in timeline] == list(range(0, len(timeline) * 10, 10))
        for name in sorted(expected_names):
            path = folder / name
            reader = imageio_ffmpeg.read_frames(str(path))
            metadata = next(reader)
            count = sum(1 for _ in reader)
            assert count == recording["frames"], (path, count, recording["frames"])
            assert abs(metadata["fps"] - recording["fps"]) < 0.001
            videos.append({"path": str(path.relative_to(directory)), "bytes": path.stat().st_size,
                           "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                           "frames": count, "fps": metadata["fps"], "size": metadata["size"],
                           "duration_seconds": count / metadata["fps"], "full_decode_passed": True})
    report = {"passed": True, "video_count": len(videos), "videos": videos}
    (directory / "video_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    report = verify(args.directory)
    print(f"Verified {report['video_count']} videos; wrote video_manifest.json")
