"""Local MCP bridge to one owned Slurm simulator, with immutable evidence."""

import argparse
import atexit
import base64
import io
import json
from pathlib import Path
import shlex
import subprocess
import tarfile
import threading
import time

from mcp.server.fastmcp import FastMCP
from mcp.types import ImageContent, TextContent

HOST = "wangzihuan@166.111.72.174"
ROOT = "/cluster/nvme6/wangzihuan/robodojo_20261025"
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=30", "-o",
       "ServerAliveInterval=30", "-o", "ServerAliveCountMax=10", HOST]


class Session:
    def __init__(self, output):
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=False)
        self.remote_output = ROOT + "/seven_blocks_code/eval_result/seven_blocks/" + self.output.parent.name + "_" + self.output.name
        self.log = (self.output / "transport.log").open("w")
        self.journal = (self.output / "rpc.jsonl").open("w")
        self.index = 0
        self.ended = False
        command = ["srun", "--partition=RTX3090", "--nodelist=node01", "--nodes=1", "--ntasks=1",
                   "--cpus-per-task=8", "--mem=48G", "--gres=gpu:3090:1", "--time=3-00:00:00",
                   "--job-name=seven-" + self.output.parent.name[-30:], "--kill-on-bad-exit=1",
                   "python", "-u", "-m", "experiments.seven_blocks.run", "--mode", "stdio",
                   "--headless", "--agent-tools", "--record-video", "--output", self.remote_output]
        shell = "source " + shlex.quote(ROOT + "/control/activate.sh") + " && cd " + shlex.quote(ROOT + "/seven_blocks_code") + " && exec " + shlex.join(command)
        self.process = subprocess.Popen([*SSH, shell], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=self.log, text=True, bufsize=1)
        self.current = self.read()
        assert self.current["event"] == "ready", self.current
        (self.output / "allocation.json").write_text(json.dumps({
            "job_id": self.current["slurm_job_id"], "remote_output": self.remote_output,
            "host": HOST, "infrastructure_walltime_seconds": 259200,
            "agent_action_limit": None, "agent_failure_limit": None,
        }, indent=2) + "\n")
        self.save({"event": "ready", "response": self.current})

    def save(self, value):
        self.journal.write(json.dumps({"wall_unix": time.time(), **value}, ensure_ascii=False) + "\n")
        self.journal.flush()

    def read(self):
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("Simulator stream ended; inspect transport.log")
        return json.loads(line)

    def call(self, request):
        if self.ended:
            return self.current
        self.save({"event": "request", "request": request, "index": self.index})
        self.process.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
        self.process.stdin.flush()
        self.current = self.read()
        self.save({"event": "response", "response": self.current, "index": self.index})
        if self.current.get("status") in ("submitted", "interrupted"):
            self.ended = True
            (self.output / "terminal.json").write_text(json.dumps(self.current, ensure_ascii=False, indent=2) + "\n")
        self.index += 1
        return self.current

    def content(self, response):
        result = [TextContent(type="text", text=json.dumps(response, ensure_ascii=False))]
        sources = response.get("images", {})
        if sources:
            names = [Path(source).name for source in sources.values()]
            archive = subprocess.run([*SSH, shlex.join(["tar", "-C", self.remote_output, "-cf", "-", "--", *names])],
                                     check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout
            directory = self.output / "observations" / f"{self.index:06d}"
            directory.mkdir(parents=True, exist_ok=True)
            with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
                for camera, name in zip(sources, names, strict=True):
                    member = tar.getmember(name)
                    assert member.isfile()
                    data = tar.extractfile(member).read()
                    (directory / name).write_bytes(data)
                    result.extend([TextContent(type="text", text=camera),
                                   ImageContent(type="image", data=base64.b64encode(data).decode(), mimeType="image/png")])
        return result

    def close(self):
        if self.process.poll() is None:
            if not self.ended:
                self.call({"op": "stop", "reason": "agent transport closed without explicit submission"})
            self.process.stdin.close()
            self.process.wait(timeout=180)
        self.journal.close()
        self.log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        session = Session(args.output)
        try:
            state = session.current["state"]
            bounds = state["blocks"]["red"]["world_bounds"]
            center = [(a+b)/2 for a,b in zip(*bounds)]
            for request in [
                {"op": "gripper", "arm": "left_arm", "opening": 1},
                {"op": "move", "arm": "left_arm", "tcp": [*center[:2],center[2]+.08]},
                {"op": "move", "arm": "left_arm", "tcp": center},
                {"op": "gripper", "arm": "left_arm", "opening": 0},
                {"op": "move", "arm": "left_arm", "tcp": [*center[:2],center[2]+.08]},
            ]:
                response = session.call(request)
                assert response.get("executed"), response
            assert response["state"]["blocks"]["red"]["world_bounds"][0][2] - state["tabletop_z"] > .07
            assert session.call({"op":"move","arm":"left_arm","tcp":[999,0,1]})["executed"] is False
            assert session.call({"op":"reset","reason":"pre-agent interface check"})["state"]["resets"] == 1
            session.call({"op":"stop","reason":"pre-agent interface check complete; no model involved"})
            (args.output/"interface_check.json").write_text(json.dumps({"passed":True,"model_calls":0})+"\n")
        finally:
            session.close()
        return
    mcp = FastMCP("seven_blocks")
    holder = []
    lock = threading.Lock()

    @mcp.tool()
    def robot(request: dict) -> list[TextContent | ImageContent]:
        """Operate the physical seven-block environment. Start with {op:observe}.

        Each reply includes three current RGB images, block world bounds and poses,
        robot TCP and joint state. Coordinates are metres in one world frame:
        X left/right, +Y away from robot bases, +Z up; table surface z=0.765.
        move: {op:move,arm:left_arm|right_arm,tcp:[x,y,z],tilt_degrees:0,yaw_degrees:0}.
        TCP is finger grasp centre. Default tilt 0 points down; positive tilt leans
        towards +Y. Rear-row reach was mechanically checked with tilt 30 degrees.
        yaw rotates around world Z; alternatively provide orientation_wxyz.
        move plans and physically executes a native cuRobo trajectory, holding the
        gripper opening unchanged. Planning failures do not move or reset objects.
        gripper: {op:gripper,arm:left_arm|right_arm,opening:0..1}; 0 closes, 1 opens.
        wait: {op:wait,physics_steps:1..250}; dt=.004 seconds, repeat freely.
        act: {op:act,control:complete control_template,physics_steps:1..250} permits
        arbitrary joint targets. move/gripper/wait/act may be repeated without cap.
        reset: {op:reset,reason:text} restores the full scene but preserves dialogue
        and trial history. observe: {op:observe} changes no physics.
        submit: {op:submit,description:text,reason:text} submits the current physical
        result and ends this experiment. Only call when you choose to submit.
        There is no target layout, artwork example, success score or failure limit.
        """
        with lock:
            if not holder:
                holder.append(Session(args.output))
            session = holder[0]
            if request.get("op") == "stop":
                return [TextContent(type="text",text="stop is an operator-only interruption; use submit when satisfied")]
            return session.content(session.call(request))

    def close():
        if holder:
            holder[0].close()

    atexit.register(close)
    try:
        mcp.run(transport="stdio")
    finally:
        close()
        atexit.unregister(close)


if __name__ == "__main__":
    main()
