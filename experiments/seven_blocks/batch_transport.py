"""A Slurm-owned simulator with a persistent stdin and durable JSONL replies."""

import json
from pathlib import Path
import shlex
import subprocess
import time


class BatchTransport:
    def __init__(self, output, ssh, root, remote_output, log, previous=None):
        self.output, self.ssh, self.log = output, ssh, log
        self.state_path = output / "transport_state.json"
        if previous:
            allocation = json.loads((previous / "allocation.json").read_text())
            if allocation.get("transport") != "slurm_batch_fifo":
                raise ValueError("Only independently owned batch simulators can be reattached")
            self.control = allocation["control_directory"]
            self.state = json.loads((previous / "transport_state.json").read_text())
            self.allocation = {**allocation, "reattached_from": str(previous)}
        else:
            self.control = remote_output + "_control"
            self.state = {"offset": 0, "pending": None, "response": None}
            native = ["python", "-u", "-m", "experiments.seven_blocks.run", "--mode", "stdio",
                      "--headless", "--agent-tools", "--record-video", "--output", remote_output]
            runner = (
                "import os,json,subprocess,time\nfrom pathlib import Path\n"
                f"p=Path({self.control!r})\n"
                # The batch job itself owns BOTH ends. Client EOF cannot close
                # the simulator's stdin, even if every local client disappears.
                "fd=os.open(p/'requests.fifo',os.O_RDWR)\n"
                "with os.fdopen(fd,'r') as inp,(p/'replies.jsonl').open('w') as out:\n"
                f" child=subprocess.Popen({native!r},stdin=inp,stdout=out)\n"
                " (p/'native_pid.json').write_text(json.dumps({'pid':child.pid}))\n"
                " code=child.wait()\n"
                "(p/'exit.json').write_text(json.dumps({'exit_code':code,'wall_unix':time.time()}))\n"
                "raise SystemExit(code)\n"
            )
            setup = (
                "import json,os,sys\nfrom pathlib import Path\n"
                "d=json.load(sys.stdin);p=Path(d['directory']);p.mkdir(parents=True,exist_ok=False)\n"
                "os.mkfifo(p/'requests.fifo',0o600)\n"
                "(p/'runner.py').write_text(d['runner'])\n(p/'replies.jsonl').touch()\n"
            )
            self.remote(setup, {"directory": self.control, "runner": runner})
            launch = ("source " + shlex.quote(root + "/control/activate.sh") + " && cd "
                      + shlex.quote(root + "/seven_blocks_code") + " && exec python "
                      + shlex.quote(self.control + "/runner.py"))
            command = ["sbatch", "--parsable", "--partition=RTX3090", "--nodelist=node01",
                       "--nodes=1", "--ntasks=1", "--cpus-per-task=8", "--mem=48G",
                       "--gres=gpu:3090:1", "--time=3-00:00:00", "--job-name=seven-"+output.parent.name[-25:],
                       "--output="+self.control+"/batch.stdout.log", "--error="+self.control+"/batch.stderr.log",
                       "--wrap="+shlex.join(["bash", "-lc", launch])]
            job = subprocess.check_output([*ssh, shlex.join(command)], stderr=log, text=True).strip().split(";")[0]
            if not job.isdigit():
                raise RuntimeError("Invalid sbatch job ID")
            self.allocation = {"job_id":job,"host":ssh[-1],"remote_output":remote_output,
                               "control_directory":self.control,"transport":"slurm_batch_fifo",
                               "infrastructure_walltime_seconds":259200,
                               "agent_action_limit":None,"agent_failure_limit":None}
        (output / "allocation.json").write_text(json.dumps(self.allocation,indent=2)+"\n")
        self.save()

    def save(self):
        temporary = self.state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.state,ensure_ascii=False)+"\n")
        temporary.replace(self.state_path)

    def remote(self, source, data):
        return subprocess.check_output([*self.ssh, shlex.join(["python3", "-c", source])],
                                       input=json.dumps(data), text=True, stderr=self.log)

    def exchange(self, request=None):
        recovering = self.state["pending"] is not None
        outgoing = None if recovering else request
        if outgoing is not None:
            self.state["pending"] = outgoing
            self.save()
        source = """import json,os,sys,time
from pathlib import Path
d=json.load(sys.stdin);p=Path(d['directory'])
if d['request'] is not None:
    if (p/'exit.json').exists():raise RuntimeError('Simulator already exited')
    fd=os.open(p/'requests.fifo',os.O_WRONLY|os.O_NONBLOCK)
    with os.fdopen(fd,'w') as stream:stream.write(json.dumps(d['request'])+'\\n')
with (p/'replies.jsonl').open('rb') as stream:
    stream.seek(d['offset'])
    while True:
        start=stream.tell();line=stream.readline()
        if line.endswith(b'\\n'):
            print(json.dumps({'offset':stream.tell(),'response':json.loads(line)}),flush=True);break
        stream.seek(start)
        if (p/'exit.json').exists():raise RuntimeError('Simulator exited without another reply')
        time.sleep(.1)
"""
        reply = json.loads(self.remote(source, {"directory":self.control,"offset":self.state["offset"],
                                                "request":outgoing}))
        pending = self.state["pending"]
        self.state.update(reply)
        self.state["pending"] = None
        self.save()
        response = reply["response"]
        if recovering:
            response = {**response,"recovered_pending_request":pending,"new_request_executed":False}
        return response
