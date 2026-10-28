"""Run one unbounded Codex robot dialogue; explicit submit owns completion."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.seven_blocks.protocol import INSTRUCTION

SYSTEM = """You control the seven-block robot environment through the seven_blocks robot tool.
Begin by observing. The user's instruction is the complete task; choose your own
goal, actions, retries, and stopping point. Use physical tool feedback and images.
Simulator coordinates are available. Native planning is an actuator service, not
a task planner. No examples, demonstrations, target layout, reward, personal memory
or prior trial transcript are provided. You may retry or reset without an action
or failure count limit. Only a successful explicit op=submit ends the experiment
as your voluntary submission. A final chat message without submit is an interruption.
Give brief observable progress notes in Chinese when useful. Do not ask a human
to choose the artwork or stopping point; those choices are yours. After submit,
give your final description and stop calling tools. Use only the robot MCP tool.
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--codex", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    sandbox = output / "workspace"
    sandbox.mkdir()
    source = Path(__file__).resolve().parents[2]
    overrides = {
        "model_reasoning_effort": "high", "developer_instructions": SYSTEM,
        "memories.use_memories": False, "memories.generate_memories": False,
        "project_doc_max_bytes": 0, "features.apps": False, "features.plugins": False,
        "features.shell_tool": False, "features.browser_use": False,
        "features.computer_use": False, "features.view_image": False,
        "web_search": "disabled",
        "mcp_servers.seven_blocks.command": sys.executable,
        "mcp_servers.seven_blocks.args": [str(source / "experiments/seven_blocks/agent_bridge.py"),
                                          "--output", str(output / "bridge")],
        "mcp_servers.seven_blocks.enabled": True,
        "mcp_servers.seven_blocks.required": True,
        "mcp_servers.seven_blocks.tools.robot.approval_mode": "approve",
        "mcp_servers.seven_blocks.startup_timeout_sec": 60,
        "mcp_servers.seven_blocks.tool_timeout_sec": 86400,
    }
    command = [str(args.codex), "-a", "never", "exec", "--ignore-user-config",
               "--skip-git-repo-check", "--sandbox", "read-only", "--json",
               "--model", "gpt-6-astra", "--cd", str(sandbox),
               "--output-last-message", str(output / "final_message.txt")]
    for key, value in overrides.items():
        command.extend(["-c", key + "=" + json.dumps(value, ensure_ascii=False)])
    command.append(INSTRUCTION)
    environment = dict(os.environ)
    for key in ("CODEX_API_KEY", "OPENAI_API_KEY", "CODEX_BASE_URL"):
        environment.pop(key, None)
    condition = {"instruction": INSTRUCTION,
                 "original_user_text": "用这些小方块展示你模型的最强能力 不限动作 不限失败 可以反复尝试 直到你满意愿意提交答案位置",
                 "instruction_normalization": "punctuation and 位置 interpreted as 为止, as documented before the run",
                 "model": "gpt-6-astra", "reasoning_effort": "high", "system_prompt": SYSTEM,
                 "simulator_ground_truth": True, "native_curobo": True,
                 "vla": False, "max_turns": None, "max_actions": None, "max_failures": None,
                 "model_timeout_seconds": None, "demonstrations": [], "personal_memory": False,
                 "started_unix": time.time(), "command": command}
    (output / "condition.json").write_text(json.dumps(condition, ensure_ascii=False, indent=2)+"\n")
    with (output / "agent_events.jsonl").open("w") as events, (output / "agent_stderr.log").open("w") as log:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=events, stderr=log, env=environment)
        (output / "process.json").write_text(json.dumps({"pid":process.pid,"controller_pid":os.getpid()})+"\n")
        code = process.wait()
    terminal = output / "bridge/terminal.json"
    status = json.loads(terminal.read_text())["status"] if terminal.exists() else "interrupted"
    (output / "result.json").write_text(json.dumps({"status":status,"cli_exit_code":code,
        "started_unix":condition["started_unix"],"finished_unix":time.time()},indent=2)+"\n")
    print(json.dumps({"status":status,"cli_exit_code":code,"output":str(output)}),flush=True)
    if code or status != "submitted":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
