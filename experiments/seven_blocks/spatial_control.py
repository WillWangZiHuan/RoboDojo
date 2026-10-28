"""Agent-directed Cartesian planning; no target artwork or grasp sequence."""

import json
import math

import numpy as np
from transforms3d.quaternions import mat2quat, quat2mat


class SpatialControl:
    def __init__(self, env, record):
        self.env, self.record = env, record
        self.manager = env.robot_manager
        self.robots = {robot.arm_name: robot for robot in self.manager.robot_list}

    def state(self):
        state = self.env.snapshot()
        state["robot_tcp"] = {}
        for name, robot in self.robots.items():
            pose = self.manager.get_real_endpose(robot)[0]
            tcp = pose[:3] + float(robot.gripper_bias) * quat2mat(pose[3:])[:, 0]
            state["robot_tcp"][name] = {
                "position": tcp.tolist(), "orientation_wxyz": pose[3:].tolist(),
                "base_position": list(robot.default_root_pos),
                "gripper_joints": self.manager.get_end_effector_real_val(robot)[0].tolist(),
            }
        table = self.env.layout["Table"]
        state["tabletop_z"] = table["default_pos"][2] + table["scale"][2] / 2
        return state

    def control(self):
        return json.loads(json.dumps(self.env.control_template(), default=lambda v: v.tolist()))

    def execute(self, request):
        op = request["op"]
        if op == "wait":
            steps = request.get("physics_steps", 250)
            self.env.act(self.control(), steps)
            return {"executed": True}
        robot = self.robots[request["arm"]]
        if op == "gripper":
            opening = request["opening"]
            if type(opening) not in (float, int) or not math.isfinite(opening) or not 0 <= opening <= 1:
                raise ValueError("opening must be a finite number in [0,1]")
            low, high = robot.gripper_scale
            value = low + opening * (high - low)
            mimic = robot.gripper_move["mimic"]
            control = self.control()
            control[self.manager.process_name(robot.gripper_name)] = {
                "position": [value, value * mimic[1] + mimic[2]]}
            self.env.act(control, 250)
            return {"executed": True}
        if op != "move":
            raise ValueError(f"Unknown spatial operation: {op}")
        tcp = np.asarray(request["tcp"], dtype=float)
        if tcp.shape != (3,) or not np.isfinite(tcp).all():
            raise ValueError("tcp requires three finite coordinates in metres")
        if "orientation_wxyz" in request:
            quat = np.asarray(request["orientation_wxyz"], dtype=float)
            if quat.shape != (4,) or not np.isfinite(quat).all() or np.linalg.norm(quat) < 1e-6:
                raise ValueError("orientation_wxyz requires a nonzero finite quaternion")
            quat /= np.linalg.norm(quat)
            rotation = quat2mat(quat)
        else:
            tilt, yaw = np.deg2rad([request.get("tilt_degrees", 0), request.get("yaw_degrees", 0)])
            if not np.isfinite([tilt, yaw]).all():
                raise ValueError("tilt and yaw must be finite")
            rotation = np.array([[0., -1., 0.],
                                 [np.sin(tilt), 0., np.cos(tilt)],
                                 [-np.cos(tilt), 0., np.sin(tilt)]])
            rotation = np.array([[np.cos(yaw), -np.sin(yaw), 0.],
                                 [np.sin(yaw), np.cos(yaw), 0.], [0., 0., 1.]]) @ rotation
            quat = mat2quat(rotation)
        wrist = tcp - float(robot.gripper_bias) * rotation[:, 0]
        plan = self.manager.planner[robot.robot_name].plan_path(
            self.manager.get_joint(robot)[0], [*wrist, *quat], list(robot.entity_origin_pose))
        self.record({"event": "agent_motion_plan", "request": request, "status": plan["status"]})
        if plan["status"] != "Success":
            return {"executed": False, "planning_status": plan["status"]}
        control = self.control()
        key = self.manager.process_name(robot.arm_name)
        for positions, velocities in zip(plan["position"], plan["velocity"], strict=True):
            control[key] = {"position": positions.tolist(), "velocity": velocities.tolist()}
            self.env.act(control, 1)
        control[key]["velocity"] = [0.] * len(control[key]["position"])
        self.env.act(control, 125)
        actual = self.manager.get_real_endpose(robot)[0]
        return {"executed": True, "planning_status": plan["status"],
                "wrist_tracking_error_m": float(np.linalg.norm(actual[:3] - wrist))}
