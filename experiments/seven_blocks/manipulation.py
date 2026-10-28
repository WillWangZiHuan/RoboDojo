"""Scripted physical acceptance only; never exposed as an agent objective.

Uses evaluator-only geometry and native cuRobo trajectories. No object pose is
written during a grasp: only robot joint targets advance the physics.
"""

import json

import numpy as np
from transforms3d.quaternions import mat2quat

from experiments.seven_blocks.protocol import COLORS
from experiments.seven_blocks.recording import VideoRecorder


def check_manipulation(env, observe, record, output, colors=COLORS):
    manager = env.robot_manager
    results = []
    table = env.layout["Table"]
    table_top = table["default_pos"][2] + table["scale"][2] / 2
    home_joints = env.snapshot()["robot_joints"]

    def control():
        return json.loads(json.dumps(env.control_template(), default=lambda v: v.tolist()))

    def grasp_geometry(color, state):
        center = np.mean(state["blocks"][color]["world_bounds"], axis=0)
        robot = min(manager.robot_list, key=lambda r: abs(r.default_root_pos[0] - center[0]))
        # Both X5 bases share the same orientation. Use the same wrist roll;
        # tilt rear-row grasps towards +Y to keep the wrist within reach.
        tilt = np.deg2rad(30. if center[1] > 0 else 0.)
        rotation = np.array([[0., -1., 0.],
                             [np.sin(tilt), 0., np.cos(tilt)],
                             [-np.cos(tilt), 0., np.sin(tilt)]])
        return center, robot, rotation, mat2quat(rotation).tolist(), tilt

    # Check every approach from the same measured home before recording any
    # long physical sequence. Reuse those plans after verified full resets.
    approaches = {}
    initial_state = env.snapshot()
    for color in colors:
        center, robot, rotation, quaternion, _ = grasp_geometry(color, initial_state)
        wrist = center + [0., 0., 0.08] - float(robot.gripper_bias) * rotation[:, 0]
        plan = manager.planner[robot.robot_name].plan_path(
            manager.get_joint(robot)[0], [*wrist, *quaternion], list(robot.entity_origin_pose))
        record({"event": "approach_preflight", "color": color, "status": plan["status"]})
        approaches[color] = plan
    failed = [color for color, plan in approaches.items() if plan["status"] != "Success"]
    if failed:
        raise RuntimeError(f"Approach preflight failed: {failed}")

    for color in colors:
        env.soft_reset(f"independent scripted grasp check: {color}")
        start = env.snapshot()
        for arm, joints in home_joints.items():
            if not np.allclose(start["robot_joints"][arm], joints, atol=0.003):
                raise RuntimeError(f"Reset did not restore {arm} before checking {color}")
        center, robot, rotation, quaternion, tilt = grasp_geometry(color, start)
        arm_key = manager.process_name(robot.arm_name)
        gripper_key = manager.process_name(robot.gripper_name)
        planner = manager.planner[robot.robot_name]
        target = center + [0., -0.08, 0.]
        row = {"color": color, "arm": robot.arm_name, "initial_center": center.tolist(),
               "target_center": target.tolist(), "grasp_tilt_degrees": float(np.rad2deg(tilt)),
               "initial_robot_joints": start["robot_joints"],
               "passed": False}
        record({"event": "manipulation_start", **row})
        observe()
        video = VideoRecorder(env, output / "videos" / color)
        env.physics_observer = video.step

        def move(tcp, phase):
            video.phase = phase
            wrist = np.asarray(tcp) - float(robot.gripper_bias) * rotation[:, 0]
            plan = approaches[color] if phase == "approach" else planner.plan_path(
                manager.get_joint(robot)[0], [*wrist, *quaternion], list(robot.entity_origin_pose))
            record({"event": "motion_plan", "color": color, "phase": phase,
                    "status": plan["status"], "target_tcp": np.asarray(tcp).tolist()})
            if plan["status"] != "Success":
                raise RuntimeError(f"Cannot plan {phase}")
            command = control()
            for positions, velocities in zip(plan["position"], plan["velocity"], strict=True):
                command[arm_key] = {"position": positions.tolist(), "velocity": velocities.tolist()}
                env.act(command, 1)
            command[arm_key]["velocity"] = [0.] * len(command[arm_key]["position"])
            env.act(command, 125)
            actual = manager.get_real_endpose(robot)[0]
            error = float(np.linalg.norm(actual[:3] - wrist))
            record({"event": "motion_end", "color": color, "phase": phase, "position_error_m": error})
            if error > 0.015:
                raise RuntimeError(f"Tracking error during {phase}: {error:.4f} m")

        def grip(opened):
            video.phase = "open" if opened else "close"
            command = control()
            value = robot.gripper_scale[1 if opened else 0]
            mimic = robot.gripper_move["mimic"]
            command[gripper_key] = {"position": [value, value * mimic[1] + mimic[2]]}
            env.act(command, 250)

        try:
            grip(True)
            move(center + [0., 0., 0.08], "approach")
            move(center, "descend")
            grip(False)
            observe()
            move(center + [0., 0., 0.08], "lift")
            lifted = env.snapshot()["blocks"][color]
            row["lift_height_m"] = float(lifted["world_bounds"][0][2] - table_top)
            observe()
            if row["lift_height_m"] < 0.07:
                raise RuntimeError("Block was not physically lifted")
            move(target + [0., 0., 0.08], "transport")
            move(target + [0., 0., 0.003], "lower")
            grip(True)
            move(target + [0., 0., 0.08], "retreat")
            env.act(control(), 250)
            final = env.snapshot()["blocks"][color]
            final_center = np.mean(final["world_bounds"], axis=0)
            row["final_center"] = final_center.tolist()
            row["placement_xy_error_m"] = float(np.linalg.norm(final_center[:2] - target[:2]))
            row["table_gap_m"] = float(final["world_bounds"][0][2] - table_top)
            row["passed"] = row["placement_xy_error_m"] < 0.02 and abs(row["table_gap_m"]) < 0.01
        except RuntimeError as exc:
            row["error"] = str(exc)
        finally:
            env.physics_observer = None
            row["recording"] = video.close()
        observe()
        record({"event": "manipulation_result", **row})
        results.append(row)

    return {"passed": all(row["passed"] for row in results), "model_calls": 0,
            "evaluator_geometry": True, "synthetic_submission": False,
            "checks": results}
