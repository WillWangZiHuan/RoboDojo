"""Collect a terminal simulator's complete videos, metadata and camera frames."""

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import tarfile


def collect(bridge, output):
    allocation = json.loads((bridge / "allocation.json").read_text())
    remote = allocation["remote_output"]
    ssh = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=30", allocation["host"]]
    source = """import json,sys
from pathlib import Path
p=Path(sys.argv[1]);e=json.loads((p/'episode.json').read_text())
assert e['status'] in ('submitted','interrupted'),e['status']
print(json.dumps({'episode':e,'images':sorted(x.name for x in p.glob('frame_*.png'))}))
"""
    info = json.loads(subprocess.check_output([*ssh, shlex.join(["python3", "-c", source, remote])], text=True))
    output.mkdir(parents=True, exist_ok=False)
    for path in bridge.parent.glob("bridge*/observations/*/frame_*.png"):
        target = output / path.name
        if target.exists():
            assert target.read_bytes() == path.read_bytes(), path.name
        else:
            shutil.copyfile(path, target)
    names = ["config.json", "layout.json", "episode.json", "events.jsonl", "recording.json", "videos"]
    names += [name for name in info["images"] if not (output / name).exists()]
    archive = output / "native_artifacts.tar.gz"
    with archive.open("wb") as stream:
        subprocess.run([*ssh, shlex.join(["tar", "-C", remote, "-czf", "-", "--", *names])],
                       stdout=stream, check=True)
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            if member.isdir():
                continue
            path = Path(member.name)
            assert member.isfile() and not path.is_absolute() and ".." not in path.parts
            target = output / path
            target.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
    assert all((output / name).exists() for name in info["images"])
    report = {"episode_status": info["episode"]["status"], "all_camera_pngs_local":len(info["images"]),
              "raw_videos":sorted(str(p.relative_to(output)) for p in (output/"videos").glob("*.mp4")),
              "archive_sha256":hashlib.sha256(archive.read_bytes()).hexdigest()}
    (output / "download_manifest.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    collect(args.bridge.resolve(), args.output.resolve())
