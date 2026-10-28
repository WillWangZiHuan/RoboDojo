"""Seven colored native blocks on a native RoboDojo table.

Import only after IsaacLab AppLauncher. This research environment intentionally
does not enter the fixed-answer benchmark inventory or its success-rate loop.
"""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from pxr import Gf, Sdf, UsdShade

from env.environment.task_env import TaskEnv
from env.global_configs import ASSETS_PATH, ENV_CONFIG_PATH
from experiments.seven_blocks.protocol import COLORS, INSTRUCTION, Episode, make_layout, validate_control


def build_config(seed=0):
    root = Path(ENV_CONFIG_PATH)
    names = {"sim": "sim_config", "scene": "default", "robot": "dual_x5", "camera": "camera_config"}
    config = OmegaConf.create({key: OmegaConf.load(root / key / f"{name}.yml") for key, name in names.items()})
    config.sim.scene.num_envs = 1
    config.sim.seed = [seed]
    config.camera.default_frequency = 25
    # Remove the decorative camera stand; camera/robot mounts still work.
    config.scene.pop("Geometry", None)
    config.task_env = {}
    return config


class SevenBlocksEnv(TaskEnv):
    def __init__(self, config, app, template_path=None):
        if config.sim.scene.num_envs != 1:
            raise ValueError("The research protocol currently supports one environment")
        template_path = Path(template_path) if template_path else (
            Path(ASSETS_PATH) / "Eval_Layout/RoboDojo/arx_x5/1/stack_blocks_0.json")
        self.template_path = template_path
        self.layout = make_layout(json.loads(template_path.read_text()))
        self.episode = Episode()
        super().__init__(config, app)
        self.scene_manager.layout_manager.set_saved_layout(0, deepcopy(self.layout))

    def reset(self, seed=None, options=None):
        """Start a NEW episode; soft_reset is the retry operation within one."""
        if self.sim is not None:
            raise RuntimeError("Use soft_reset for retries; create a new instance for a new episode")
        super().reset(seed=seed, options=options)
        self._restore_scene()

    def _restore_scene(self):
        self.robot_manager.reset()
        self.scene_manager.apply_saved_poses([0])
        self.robot_manager.set_origin_endpose()
        self.robot_manager.set_robot_init_state()
        self._apply_colors()
        for _ in range(100):
            self.sim_step(render=False)

    def soft_reset(self, reason="agent requested scene reset"):
        """Explicit full-scene recovery, preserving history and action counts."""
        self.episode.require_running()
        self._restore_scene()
        self.episode.reset_scene(reason)

    def _objects(self):
        manager = self.scene_manager.layout_manager
        for color in COLORS:
            name = manager.get_instance_name(0, f"block_{color}")
            obj = manager.get_scene_object(0, name)
            if obj is None:
                raise RuntimeError(f"Missing block: {color}")
            yield color, obj

    def _apply_colors(self):
        for color, obj in self._objects():
            root = obj.stage.GetPrimAtPath(obj.prim_path)
            path = f"{obj.prim_path}/SevenBlocksMaterial"
            material = UsdShade.Material.Define(obj.stage, path)
            shader = UsdShade.Shader.Define(obj.stage, path + "/Shader")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*COLORS[color]))
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.65)
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            # Override the native textured material on descendants as well.
            UsdShade.MaterialBindingAPI.Apply(root).Bind(material, UsdShade.Tokens.strongerThanDescendants)
            metadata = self.scene_manager.layout_manager.get_instance_metadata(0, label=f"block_{color}")
            metadata["description"] = [f"a {color} block"]
            metadata["caption"] = f"{color} block"

    def snapshot(self):
        blocks = {}
        for color, obj in self._objects():
            position, orientation = obj.get_local_pose()
            vertices, _, _, _ = obj.sample_mesh_vertices()
            blocks[color] = {"position": np.asarray(position).tolist(),
                             "orientation_wxyz": np.asarray(orientation).tolist(),
                             "world_bounds": [vertices.min(axis=0).tolist(), vertices.max(axis=0).tolist()],
                             "rgb": list(COLORS[color])}
        return {"blocks": blocks, "status": self.episode.status, "actions": self.episode.actions,
                "physics_steps": self.episode.physics_steps, "resets": self.episode.resets}

    def control_template(self):
        return deepcopy(self.robot_manager.control_manager.prev_control[0])

    def act(self, control, physics_steps=10):
        from env.robot_manager.control_manager import MetaControl

        self.episode.require_running()
        if type(physics_steps) is not int or not 1 <= physics_steps <= 250:
            raise ValueError("Each command must contain 1..250 physics steps; commands may be repeated")
        validate_control(control, self.control_template())
        for _ in range(physics_steps):
            self.step([MetaControl(control)])
            self.sim_step(render=False)
        self.robot_manager.control_manager.update_prev_control(0, MetaControl(control))
        self.episode.record_action(physics_steps)

    def submit(self, description, reason):
        return self.episode.submit(description, reason, self.snapshot())

    def run_reward(self):
        # Completion flag only, never an objective assessment of the work.
        return [float(self.episode.status == "submitted")]

    def gen_instruction(self, env_idx=0):
        return [INSTRUCTION]
