"""Simulator-independent scene recipe and explicit submission protocol."""

from copy import deepcopy
from dataclasses import dataclass, field
import math

INSTRUCTION = "用这些小方块展示你模型的最强能力。不限动作，不限失败，可以反复尝试，直到你满意、愿意提交答案为止。"
COLORS = {
    "red": (0.85, 0.04, 0.04),
    "orange": (1.0, 0.30, 0.015),
    "yellow": (0.95, 0.80, 0.02),
    "green": (0.04, 0.55, 0.10),
    "blue": (0.035, 0.18, 0.85),
    "purple": (0.45, 0.05, 0.65),
    "white": (0.90, 0.90, 0.90),
}
# Two staggered rows, clear of both robot bases. No target arrangement is supplied.
POSITIONS_XY = ((-0.30, -0.14), (-0.10, -0.14), (0.10, -0.14), (0.30, -0.14),
                (-0.20, 0.04), (0.0, 0.04), (0.20, 0.04))


def make_layout(template):
    """Reuse an installed RoboDojo stack_blocks asset, scale and table exactly.

    Seven copies of ONE source block guarantee equal geometry. The initial Z
    is inherited from its accepted native layout, rather than guessed from an
    undocumented mesh origin. Dimensions are measured by the runtime smoke test.
    """
    source = template["Rigid"]["block"][0]
    if not source.get("scale") or len(source["default_pos"]) != 3:
        raise ValueError("Native block template must contain scale and default_pos")
    layout = {key: deepcopy(template[key]) for key in ("Table", "Room", "Ground", "Light") if key in template}
    if "Table" not in layout:
        raise ValueError("Native layout has no Table")
    blocks = []
    for color, xy in zip(COLORS, POSITIONS_XY, strict=True):
        block = deepcopy(source)
        for key in ("prim_path", "inst_name", "usd_path"):
            block.pop(key, None)
        block.update(label=f"block_{color}", default_pos=[*xy, source["default_pos"][2]])
        # Physical properties are explicit and shared across colors.
        block["physics"] = {**block.get("physics", {}), "type": "rigid", "mass": 0.05,
                            "static_friction": 0.8, "dynamic_friction": 0.6, "restitution": 0.0}
        blocks.append(block)
    layout["Rigid"] = {"block": blocks}
    return layout


@dataclass
class Episode:
    """Submission is a voluntary terminal event, never a quality score."""

    status: str = "running"
    actions: int = 0
    physics_steps: int = 0
    resets: int = 0
    events: list = field(default_factory=list)
    submission: dict | None = None

    def require_running(self):
        if self.status != "running":
            raise RuntimeError(f"Episode already ended: {self.status}")

    def record_action(self, steps):
        self.require_running()
        if type(steps) is not int or steps <= 0:
            raise ValueError("steps must be a positive integer")
        self.actions += 1
        self.physics_steps += steps

    def reset_scene(self, reason):
        self.require_running()
        self.resets += 1
        self.events.append({"event": "scene_reset", "reason": reason, "after_action": self.actions})

    def submit(self, description, reason, snapshot):
        self.require_running()
        if not isinstance(description, str) or not description.strip():
            raise ValueError("A submission description is required")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("A satisfaction reason is required")
        self.submission = deepcopy({"description": description, "reason": reason, "snapshot": snapshot})
        self.status = "submitted"
        return deepcopy(self.submission)

    def interrupt(self, reason):
        self.require_running()
        self.status = "interrupted"
        self.events.append({"event": "interrupted", "reason": reason})


def validate_control(control, expected):
    """Validate native joint-position commands before touching simulation."""
    if not isinstance(control, dict) or set(control) != set(expected):
        raise ValueError("control must contain every arm/gripper key from control_template")
    for key, reference in expected.items():
        values = control[key]
        if not isinstance(values, dict) or set(values) - {"position", "velocity"}:
            raise ValueError(f"Invalid fields for {key}")
        position = values.get("position")
        if not isinstance(position, list) or len(position) != len(reference["position"]):
            raise ValueError(f"Incorrect position dimensions for {key}")
        velocity = values.get("velocity", [0.0] * len(position))
        if not isinstance(velocity, list) or len(velocity) != len(position):
            raise ValueError(f"Incorrect velocity dimensions for {key}")
        if any(type(x) not in (int, float) or not math.isfinite(x) for x in position + velocity):
            raise ValueError("Joint commands must contain finite numbers")
