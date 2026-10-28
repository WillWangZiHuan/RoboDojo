"""Exercise the real GPU JSONL process without calling an agent/model."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    responses = []
    with (args.output / "simulator.log").open("w") as log:
        process = subprocess.Popen(
            [sys.executable, "-u", "-m", "experiments.seven_blocks.run", "--mode", "stdio",
             "--headless", "--output", str(args.output / "session")],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True, bufsize=1,
        )

        def read():
            value = json.loads(process.stdout.readline())
            responses.append(value)
            return value

        def request(value):
            process.stdin.write(json.dumps(value) + "\n")
            process.stdin.flush()
            return read()

        try:
            ready = read()
            assert ready["event"] == "ready" and ready["status"] == "running"
            assert set(ready["images"]) == {"cam_head", "cam_left_wrist", "cam_right_wrist"}
            assert request({"op": "observe"})["status"] == "running"
            assert request({"op": "act", "control": {}})["event"] == "request_error"
            assert request({"op": "act", "control": ready["control_template"],
                            "physics_steps": 10})["status"] == "running"
            assert request({"op": "reset", "reason": "synthetic protocol check"})["status"] == "running"
            assert request({"op": "submit", "description": "Synthetic interface check",
                            "reason": "Verify explicit submit transport"})["status"] == "submitted"
            process.stdin.close()
            assert process.wait(timeout=60) == 0
            episode = json.loads((args.output / "session/episode.json").read_text())
            assert episode["status"] == "submitted" and episode["actions"] == 1 and episode["resets"] == 1
            report = {"passed": True, "model_calls": 0, "synthetic_submission": True,
                      "responses": responses}
            (args.output / "stdio_acceptance.json").write_text(json.dumps(report, indent=2) + "\n")
            print("Real JSONL ready/observe/invalid-action/act/reset/submit checks passed")
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


if __name__ == "__main__":
    main()
