"""Exercise the watchdog with a synthetic client; no model or simulator runs."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


def check(output):
    with tempfile.TemporaryDirectory(prefix="seven-blocks-watchdog-") as temporary:
        root = Path(temporary)
        bridge = root / "bridge"
        bridge.mkdir()
        journal = bridge / "rpc.jsonl"
        journal.write_text("")
        pending_item = {"type":"mcp_tool_call","server":"seven_blocks","tool":"robot","id":"initial-observe"}
        client = root / "synthetic_client"
        client.write_text("#!/usr/bin/env python3\nimport json,sys,time\nfrom pathlib import Path\n"
                          "print(json.dumps({'type':'thread.started','thread_id':'watchdog-test'}),flush=True)\n"
                          "if 'resume' in sys.argv:\n"
                          f" Path({str(bridge / 'terminal.json')!r}).write_text(json.dumps({{'status':'submitted'}}))\n"
                          "else:\n"
                          f" print(json.dumps({{'type':'item.started','item':{pending_item!r}}}),flush=True)\n"
                          " time.sleep(3600)\n")
        client.chmod(0o700)
        result_dir = root / "controller"
        command = [sys.executable,"-m","experiments.seven_blocks.run_agent","--codex",str(client),
                   "--output",str(result_dir),"--bridge-url","http://127.0.0.1:1/mcp",
                   "--bridge-output",str(bridge),"--transport-idle-seconds","60"]
        process = subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        try:
            time.sleep(67)
            assert process.poll() is None, "Controller exited during initial MCP/queue wait"
            assert not (result_dir/"transport_recoveries.jsonl").exists(), "Restarted before the initial bridge reply"
            events = result_dir / "agent_events.jsonl"
            with events.open("a") as stream:
                stream.write(json.dumps({"type":"item.completed","item":pending_item})+"\n")
            os.utime(events,(time.time()-120,time.time()-120))
            journal.write_text(json.dumps({"event":"request","index":0,"request":{"op":"wait"}})+"\n")
            os.utime(journal,(time.time()-120,time.time()-120))
            time.sleep(7)
            assert process.poll() is None, "Controller exited during pending physical request"
            assert not (result_dir/"transport_recoveries.jsonl").exists(), "Replayed an uncertain physical command"
            with journal.open("a") as stream:
                stream.write(json.dumps({"event":"response","index":0,"response":{"status":"running"}})+"\n")
            # Simulate an already-completed response followed by a silent model.
            os.utime(journal,(time.time()-120,time.time()-120))
            stdout,stderr = process.communicate(timeout=30)
            assert process.returncode == 0, (stdout,stderr)
            result = json.loads((result_dir/"result.json").read_text())
            recovery = [json.loads(line) for line in (result_dir/"transport_recoveries.jsonl").read_text().splitlines()]
            assert result["transport_recoveries"] == 1 and len(recovery) == 1
            assert recovery[0]["thread_id"] == "watchdog-test"
            report = {"passed":True,"synthetic_only":True,"model_calls":0,"physical_simulators":0,
                      "pending_initial_mcp_without_rpc_prevents_restart":True,
                      "pending_physical_request_prevents_restart":True,
                      "silent_client_restarted_after_response":True,
                      "original_thread_id_preserved":True,"physical_command_replayed":False}
            output.write_text(json.dumps(report,indent=2)+"\n")
            print(json.dumps(report),flush=True)
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=10)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    check(parser.parse_args().output)
