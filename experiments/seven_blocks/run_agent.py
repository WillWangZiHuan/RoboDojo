"""Run one unbounded Codex robot dialogue; explicit submit owns completion."""

import argparse
import json
import os
from pathlib import Path
import signal
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


def bridge_in_flight(directory):
    journal = directory / "rpc.jsonl"
    if not journal.exists():
        return False
    requested = answered = -1
    for line in journal.read_text().splitlines():
        row = json.loads(line)
        if row.get("event") == "request":
            requested = row["index"]
        elif row.get("event") == "response":
            answered = row["index"]
    return requested > answered


def end_client(process):
    """End only this client's process group; the HTTP simulator stays alive."""
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--codex", type=Path, required=True)
    parser.add_argument("--resume-thread", help="Resume the original model dialogue after transport recovery")
    parser.add_argument("--bridge-url", help="Persistent loopback MCP server for a resumed dialogue")
    parser.add_argument("--bridge-output", type=Path, help="Evidence directory owned by that MCP server")
    parser.add_argument("--model", choices=("gpt-6-astra", "gpt-6-sol", "gpt-6.1-terra"),
                        default="gpt-6-astra")
    parser.add_argument("--transport-idle-seconds", type=int, default=600,
                        help="Recover a silent HTTP-backed model client after this interval")
    parser.add_argument("--auto-compact-tokens", type=int, default=60000,
                        help="Use native context compaction before image history becomes too large")
    args = parser.parse_args()
    if bool(args.bridge_url) != bool(args.bridge_output) or (args.resume_thread and not args.bridge_url):
        parser.error("HTTP requires --bridge-url and --bridge-output; resume requires HTTP")
    if args.transport_idle_seconds < 60:
        parser.error("Transport idle recovery must allow at least 60 seconds")
    if args.bridge_url and not args.bridge_url.startswith("http://127.0.0.1:"):
        parser.error("The persistent bridge must be a loopback server")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    sandbox = output / "workspace"
    sandbox.mkdir()
    source = Path(__file__).resolve().parents[2]
    overrides = {
        "model_reasoning_effort": "high", "developer_instructions": SYSTEM,
        "model_auto_compact_token_limit": args.auto_compact_tokens,
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
    if args.bridge_url:
        del overrides["mcp_servers.seven_blocks.command"]
        del overrides["mcp_servers.seven_blocks.args"]
        overrides["mcp_servers.seven_blocks.url"] = args.bridge_url
    command = [str(args.codex), "-a", "never", "exec", "--ignore-user-config",
               "--skip-git-repo-check", "--sandbox", "read-only", "--json",
               "--model", args.model, "--cd", str(sandbox),
               "--output-last-message", str(output / "final_message.txt")]
    for key, value in overrides.items():
        command.extend(["-c", key + "=" + json.dumps(value, ensure_ascii=False)])
    command_prefix = list(command)
    resume_prompt = ("模型连接已恢复。继续原任务；仿真现场未重置，请先观察当前环境。"
                     "继续自行决定后续动作、尝试与主动提交时机。")
    if args.resume_thread:
        command.extend(["resume", args.resume_thread, resume_prompt])
    else:
        command.append(INSTRUCTION)
    environment = dict(os.environ)
    for key in ("CODEX_API_KEY", "OPENAI_API_KEY", "CODEX_BASE_URL"):
        environment.pop(key, None)
    condition = {"instruction": INSTRUCTION,
                 "original_user_text": "用这些小方块展示你模型的最强能力 不限动作 不限失败 可以反复尝试 直到你满意愿意提交答案位置",
                 "instruction_normalization": "punctuation and 位置 interpreted as 为止, as documented before the run",
                 "model": args.model, "reasoning_effort": "high", "system_prompt": SYSTEM,
                 "simulator_ground_truth": True, "native_curobo": True,
                 "vla": False, "max_turns": None, "max_actions": None, "max_failures": None,
                 "model_timeout_seconds": None, "demonstrations": [], "personal_memory": False,
                 "resume_thread": args.resume_thread,
                 "resume_prompt": resume_prompt if args.resume_thread else None,
                 "bridge_url": args.bridge_url,
                 "transport_idle_recovery_seconds": args.transport_idle_seconds if args.bridge_url else None,
                 "auto_compact_token_limit": args.auto_compact_tokens,
                 "started_unix": time.time(), "command": command}
    (output / "condition.json").write_text(json.dumps(condition, ensure_ascii=False, indent=2)+"\n")
    bridge_output = args.bridge_output if args.bridge_output else output / "bridge"
    terminal = bridge_output / "terminal.json"
    events_path = output / "agent_events.jsonl"
    recoveries = 0
    with events_path.open("w") as events, (output / "agent_stderr.log").open("w") as log:
        while True:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=events, stderr=log,
                                       env=environment, start_new_session=True)
            launched = time.time()
            (output / "process.json").write_text(json.dumps({"pid":process.pid,"controller_pid":os.getpid(),
                                                           "recovery_count":recoveries})+"\n")
            retry = False
            while process.poll() is None:
                time.sleep(5)
                if not args.bridge_url or process.poll() is not None:
                    continue
                rpc = bridge_output / "rpc.jsonl"
                last_progress = max(launched, events_path.stat().st_mtime,
                                    rpc.stat().st_mtime if rpc.exists() else launched)
                if time.time() - last_progress < args.transport_idle_seconds or bridge_in_flight(bridge_output):
                    continue
                # Never reissue an uncertain physical command. Resume the saved
                # dialogue, which first observes the still-live environment.
                thread = args.resume_thread
                for line in events_path.read_text().splitlines():
                    row = json.loads(line)
                    if row.get("type") == "thread.started":
                        thread = row["thread_id"]
                if not thread:
                    raise RuntimeError("Silent client has no resumable thread ID")
                end_client(process)
                entry = {"wall_unix":time.time(), "reason":"model_transport_idle", "thread_id":thread,
                         "idle_seconds":time.time()-last_progress, "client_pid":process.pid,
                         "physical_command_replayed":False, "simulator_reset":False}
                with (output / "transport_recoveries.jsonl").open("a") as recovery_log:
                    recovery_log.write(json.dumps(entry)+"\n")
                if not terminal.exists():
                    command = [*command_prefix, "resume", thread, resume_prompt]
                    recoveries += 1
                    retry = True
                break
            code = process.wait()
            if not retry:
                break
    status = json.loads(terminal.read_text())["status"] if terminal.exists() else "interrupted"
    (output / "result.json").write_text(json.dumps({"status":status,"cli_exit_code":code,
        "started_unix":condition["started_unix"],"finished_unix":time.time(),
        "transport_recoveries":recoveries},indent=2)+"\n")
    print(json.dumps({"status":status,"cli_exit_code":code,"output":str(output)}),flush=True)
    if code or status != "submitted":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
