"""Native preview, smoke acceptance and JSONL agent interface (no model calls)."""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import signal
import sys
import time

from experiments.seven_blocks.protocol import COLORS, INSTRUCTION


def json_default(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=json_default) + "\n")


def check_scene(snapshot, table):
    """Physical smoke checks; these are NOT the agent's success criteria."""
    import numpy as np

    if set(snapshot["blocks"]) != set(COLORS):
        raise RuntimeError("Expected exactly seven labeled blocks")
    top = table["default_pos"][2] + table["scale"][2] / 2
    dimensions = []
    bounds = []
    for block in snapshot["blocks"].values():
        lower, upper = np.asarray(block["world_bounds"])
        if not np.isfinite([lower, upper]).all() or np.any(upper <= lower):
            raise RuntimeError("Invalid block geometry")
        if abs(lower[2] - top) > 0.015:
            raise RuntimeError("A block is not resting on the table")
        for axis in (0, 1):
            center = table["default_pos"][axis]
            half = table["scale"][axis] / 2
            if lower[axis] < center - half or upper[axis] > center + half:
                raise RuntimeError("A block extends beyond the tabletop")
        dimensions.append(upper - lower)
        bounds.append((lower, upper))
    if not np.allclose(dimensions, dimensions[0], atol=0.003):
        raise RuntimeError("Blocks do not have equal dimensions")
    for i, (lo, hi) in enumerate(bounds):
        for other_lo, other_hi in bounds[i + 1:]:
            if np.all(np.minimum(hi[:2], other_hi[:2]) - np.maximum(lo[:2], other_lo[:2]) > 0):
                raise RuntimeError("Initial block footprints overlap")
    return {"block_count": 7, "dimensions_m": [d.tolist() for d in dimensions], "tabletop_z": top}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preview", "smoke", "manipulation", "stdio"), default="preview")
    parser.add_argument("--output", type=Path, required=True, help="New directory; never overwrite a prior run")
    parser.add_argument("--template", type=Path, help="Native stack_blocks layout JSON override")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--preview-steps", type=int, default=250)
    parser.add_argument("--colors", nargs="+", choices=list(COLORS),
                        help="Run selected manipulation checks; default: all seven")
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.preview_steps < 0:
        parser.error("--preview-steps must be nonnegative")
    if args.colors and (args.mode != "manipulation" or len(set(args.colors)) != len(args.colors)):
        parser.error("--colors requires manipulation mode and distinct colors")
    args.enable_cameras = True
    args.output.mkdir(parents=True, exist_ok=False)
    # Isaac writes logs to stdout. Reserve the original descriptor for JSONL.
    reply_stream = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    sys.stdout.flush()
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    from env.camera_manager.capture.render_sync import add_zero_delay_kit_args

    add_zero_delay_kit_args(args)
    app = AppLauncher(args).app
    env = None
    journal = (args.output / "events.jsonl").open("w")
    frame_index = 0

    def record(event):
        event = {"wall_unix": time.time(), **event}
        journal.write(json.dumps(event, ensure_ascii=False, default=json_default) + "\n")
        journal.flush()
        return event

    def reply(event):
        print(json.dumps(event, ensure_ascii=False, default=json_default), file=reply_stream, flush=True)

    def observe():
        nonlocal frame_index
        import numpy as np
        from PIL import Image

        from env.camera_manager.capture.render_sync import wait_for_latest_cameras

        wait_for_latest_cameras(env, min_passes=2)
        images = {}
        for name, data in zip(env.camera_manager.camera_names[0], env.capture_manager.step(), strict=True):
            rgb = np.asarray(data["rgb"][0]["data"])
            if rgb.size == 0 or rgb.ndim != 3 or rgb.shape[2] < 3:
                raise RuntimeError(f"Invalid camera frame: {name}")
            path = args.output / f"frame_{frame_index:06d}_{name}.png"
            Image.fromarray(rgb[..., :3].astype(np.uint8)).save(path)
            images[name] = str(path.resolve())
        frame_index += 1
        # Ground truth is evidence on disk; the default agent reply is RGB only.
        state = env.snapshot()
        record({"event": "observation", "state": state, "images": images})
        return {"images": images, "status": env.episode.status}

    try:
        from experiments.seven_blocks.environment import SevenBlocksEnv, build_config
        from omegaconf import OmegaConf

        config = build_config(args.seed)
        env = SevenBlocksEnv(config, app, args.template)
        env.reset(seed=[args.seed])
        dump(args.output / "layout.json", env.layout)
        dump(args.output / "config.json", OmegaConf.to_container(config, resolve=True))
        initial = observe()
        record({"event": "ready", "instruction": INSTRUCTION, "seed": args.seed,
                "template": str(env.template_path), "mode": args.mode})
        if args.mode == "smoke":
            import numpy as np

            first = env.snapshot()
            report = check_scene(first, env.layout["Table"])
            assert env.run_reward() == [0.0], "Initial state must not be completed"
            # Hold robot joints while physics progresses; no model was invoked.
            control = json.loads(json.dumps(env.control_template(), default=json_default))
            env.act(control, 250)
            second = env.snapshot()
            check_scene(second, env.layout["Table"])
            for color in COLORS:
                drift = np.linalg.norm(np.asarray(first["blocks"][color]["position"])
                                       - np.asarray(second["blocks"][color]["position"]))
                if drift > 0.005:
                    raise RuntimeError(f"Block did not settle: {color}, drift={drift}")
            # Move a wrist joint so the reset check cannot pass by holding still.
            moved_control = json.loads(json.dumps(control))
            arm_key = next(key for key in moved_control if "arm" in key)
            moved_control[arm_key]["position"][-1] += 0.15
            env.act(moved_control, 250)
            moved_state = env.snapshot()
            assert any(not np.allclose(moved_state["robot_joints"][arm], joints, atol=0.03)
                       for arm, joints in first["robot_joints"].items()), "Reset probe did not move the robot"
            env.soft_reset("synthetic smoke reset after wrist movement")
            reset_state = env.snapshot()
            check_scene(reset_state, env.layout["Table"])
            for arm, joints in first["robot_joints"].items():
                assert np.allclose(reset_state["robot_joints"][arm], joints, atol=0.003), arm
            assert env.episode.actions == 2 and env.episode.resets == 1
            env.submit("Synthetic acceptance test", "Test the explicit submit interface")
            assert env.run_reward() == [1.0]
            try:
                env.act(control)
            except RuntimeError:
                pass
            else:
                raise AssertionError("Actions after submission must be rejected")
            observe()
            dump(args.output / "acceptance.json", {**report, "passed": True, "model_calls": 0,
                                                   "synthetic_submission": True})
        elif args.mode == "manipulation":
            from experiments.seven_blocks.manipulation import check_manipulation

            report = check_manipulation(env, observe, record, args.output, args.colors or COLORS)
            dump(args.output / "manipulation.json", report)
            env.episode.interrupt("scripted manipulation check ended; no agent submission")
            if not report["passed"]:
                raise RuntimeError("Manipulation acceptance failed; see manipulation.json")
        elif args.mode == "preview":
            for _ in range(args.preview_steps):
                env.sim_step()
            observe()
            env.episode.interrupt("preview ended; no agent submission")
        else:
            reply({"event": "ready", "instruction": INSTRUCTION, **initial,
                   "control_template": env.control_template(), "dt": env.dt,
                   "operations": ["observe", "act", "reset", "submit", "stop"]})
            for line in sys.stdin:
                if not line.strip():
                    continue
                try:
                    request = json.loads(line)
                    if not isinstance(request, dict):
                        raise ValueError("Request must be a JSON object")
                    record({"event": "request", "request": request})
                    op = request["op"]
                    if op == "act":
                        env.act(request["control"], request.get("physics_steps", 10))
                    elif op == "reset":
                        env.soft_reset(request.get("reason", "agent requested scene reset"))
                    elif op == "submit":
                        env.submit(request["description"], request["reason"])
                    elif op == "stop":
                        env.episode.interrupt(request.get("reason", "operator stopped session"))
                    elif op != "observe":
                        raise ValueError(f"Unknown operation: {op}")
                    reply({"event": "result", **observe()})
                    if env.episode.status != "running":
                        break
                except (KeyError, ValueError, TypeError) as exc:
                    reply(record({"event": "request_error", "message": str(exc)}))
            if env.episode.status == "running":
                env.episode.interrupt("input closed without submission")
    except BaseException as exc:
        record({"event": "error", "type": type(exc).__name__, "message": str(exc)})
        if env is not None and env.episode.status == "running":
            env.episode.interrupt(f"runtime error: {type(exc).__name__}")
        raise
    finally:
        try:
            if env is not None:
                dump(args.output / "episode.json", asdict(env.episode))
                env.close()
        finally:
            journal.close()
            app.close()
            reply_stream.close()


if __name__ == "__main__":
    main()
