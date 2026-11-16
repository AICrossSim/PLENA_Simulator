"""SWE-bench Verified with Qwen3.5 served by serve.py (one endpoint per weight format), agents and grading in rootless
podman on the machine running this script.

Instances: the 500 Verified instance ids sorted, then random.Random(20261113).sample(ids, 100). Agent: mini-swe-agent
2.4.6 with its tool-calling SWE-bench configuration (swebench.yaml) and, identical for every format except the
endpoint, a step limit of 100, 32,768 new tokens per step, 120 s per command and no cost limit; 3,600 s wall clock per
instance. The server runs Qwen3.5 in thinking mode and returns the reasoning separately (serve.py -- --reasoning-parser
qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder --max-model-len 131072), so the history keeps only the
answers. Grading: the SWE-bench harness 5.0.2 (SWE-bench/SWE-bench_Verified) through podman's Docker API socket. One
image per instance is pulled once and shared by the formats; at most --workers instances and --image-budget bytes of
images are on disk at a time.

usage: python -m complementary_arrays.gpu.accuracy.swebench WORK_DIR --endpoint bf16=http://127.0.0.1:28001/v1
       [--endpoint FORMAT=URL ...] [--podman podman] [--workers 3] [--limit N]
  The Docker API socket of podman must be served at WORK_DIR/podman.sock (podman system service).
"""

import argparse
import json
import os
import random
import re
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SEED = 20261113
WALL = 3600
AGENT = """agent:
  step_limit: 100
  cost_limit: 0
environment:
  timeout: 120
  pull_timeout: 900
model:
  model_name: "hosted_vllm/m"
  cost_tracking: "ignore_errors"
  model_kwargs:
    api_base: "{url}"
    api_key: "x"
    max_tokens: 32768
    timeout: 3600
    drop_params: true
"""
LOG_LOCK, IMAGE_LOCK = threading.Lock(), threading.Lock()  # pulls and removals one at a time (shared layers)


def log(*a):
    with LOG_LOCK:
        print(time.strftime("%F %T"), *a, flush=True)


def instances(n=100):
    from datasets import load_dataset

    ids = sorted(load_dataset("SWE-bench/SWE-bench_Verified", split="test")["instance_id"])
    return random.Random(SEED).sample(ids, n)


def image(iid):
    return f"docker.io/swebench/sweb.eval.x86_64.{iid.replace('__', '_1776_')}:latest".lower()


class Runner:
    def __init__(self, a):
        self.a, self.work, self.pm = a, a.work.resolve(), a.podman
        self.urls = dict(e.split("=", 1) for e in a.endpoint)
        self.env = dict(
            os.environ,
            MSWEA_DOCKER_EXECUTABLE=self.pm,
            MSWEA_COST_TRACKING="ignore_errors",
            DOCKER_HOST=f"unix://{self.work}/podman.sock",
            MSWEA_SILENT_STARTUP="1",
            PYTHONUNBUFFERED="1",
            HF_HUB_DISABLE_XET="1",
        )
        self.sizes = {}
        for fmt, url in self.urls.items():
            (self.work / f"agent_{fmt}.yaml").write_text(AGENT.format(url=url))

    def sh(self, cmd, timeout=None, **kw):
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=self.env, **kw)

    def server_ok(self, fmt):
        r = subprocess.run(["curl", "-sf", self.urls[fmt].rstrip("/") + "/models"], capture_output=True, text=True)
        return r.returncode == 0

    def estimate(self, iid):
        """Unpacked image size: Docker Hub compressed size x 2.6."""
        if iid not in self.sizes:
            repo = image(iid).split("/", 1)[1].rsplit(":", 1)[0]
            r = subprocess.run(
                ["curl", "-s", f"https://hub.docker.com/v2/repositories/{repo}/tags/latest"],
                capture_output=True,
                text=True,
            )
            try:
                self.sizes[iid] = 2.6 * json.loads(r.stdout)["full_size"]
            except Exception:
                self.sizes[iid] = 7.5e9
        return self.sizes[iid]

    def images_bytes(self):
        unit = {"B": 1, "kB": 1e3, "KB": 1e3, "MB": 1e6, "GB": 1e9}
        total = 0.0
        for x in self.sh([self.pm, "images", "--format", "{{.Size}}"]).stdout.split("\n"):
            if x.strip():
                num, u = x.split()
                total += float(num) * unit.get(u, 1e9)
        return total

    def agent_and_eval(self, fmt, iid):
        out = self.work / "runs" / fmt / iid
        if (out / "done.json").exists():
            return json.loads((out / "done.json").read_text())
        out.mkdir(parents=True, exist_ok=True)
        rec, t1, timed_out = dict(format=fmt, instance=iid), time.time(), False
        cmd = [
            "mini-extra",
            "swebench",
            "--subset",
            "verified",
            "--split",
            "test",
            "--filter",
            f"^{iid}$",
            "-o",
            str(out),
            "-c",
            "swebench.yaml",
            "-c",
            str(self.work / f"agent_{fmt}.yaml"),
            "-w",
            "1",
        ]
        with open(out / "agent.log", "w") as fo:
            p = subprocess.Popen(
                ["nice", "-n", "10", *cmd],
                stdout=fo,
                stderr=subprocess.STDOUT,
                env=self.env,
                start_new_session=True,
                cwd=str(self.work),
            )
            try:
                p.wait(timeout=WALL)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(p.pid, signal.SIGTERM)
                time.sleep(10)
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                p.wait()
        rec.update(agent_s=time.time() - t1, wall_timeout=timed_out)
        names = sorted(set(re.findall(r"minisweagent-[0-9a-f]{8}", (out / "agent.log").read_text(errors="ignore"))))
        if names:  # this agent's container only (other formats share the image)
            self.sh([self.pm, "rm", "-f", *names], timeout=300)
        patch, status, calls = "", None, None
        if (out / "preds.json").exists():
            patch = json.loads((out / "preds.json").read_text()).get(iid, {}).get("model_patch") or ""
        traj = out / iid / f"{iid}.traj.json"
        if traj.exists():
            info = json.loads(traj.read_text()).get("info", {})
            status, calls = info.get("exit_status"), (info.get("model_stats") or {}).get("api_calls")
        rec.update(exit_status=status, api_calls=calls, patch_chars=len(patch))
        t2, resolved = time.time(), False
        if patch.strip():
            ev, run_id = self.work / "eval" / fmt, f"{fmt}_{iid}"
            ev.mkdir(parents=True, exist_ok=True)
            pred = dict(instance_id=iid, model_name_or_path=f"qwen35_{fmt}", model_patch=patch)
            (out / "pred.jsonl").write_text(json.dumps(pred) + "\n")
            r = self.sh(
                [
                    "nice",
                    "-n",
                    "10",
                    sys.executable,
                    "-m",
                    "swebench.harness.run_evaluation",
                    "--dataset_name",
                    "SWE-bench/SWE-bench_Verified",
                    "--split",
                    "test",
                    "--predictions_path",
                    str(out / "pred.jsonl"),
                    "--instance_ids",
                    iid,
                    "--max_workers",
                    "1",
                    "--run_id",
                    run_id,
                    "--timeout",
                    "1800",
                ],
                timeout=3600,
                cwd=str(ev),
            )
            (out / "eval.log").write_text(r.stdout[-20000:] + r.stderr[-20000:])
            report = ev / "logs" / "run_evaluation" / run_id / f"qwen35_{fmt}" / iid / "report.json"
            if report.exists():
                resolved = bool(json.loads(report.read_text()).get(iid, {}).get("resolved"))
            else:
                rec["eval_error"] = (r.stdout + r.stderr)[-500:]
            self.sh([self.pm, "rm", "-f", f"sweb.eval.{iid.lower()}.{run_id}"], timeout=300)
        outcome = (
            "resolved"
            if resolved
            else "wall_timeout"
            if timed_out
            else "step_limit"
            if status == "LimitsExceeded"
            else "patch_invalid"
            if patch.strip() and "eval_error" in rec
            else "tests_failed"
            if patch.strip()
            else "no_patch"
        )
        rec.update(eval_s=time.time() - t2, resolved=resolved, outcome=outcome)
        (out / "done.json").write_text(json.dumps(rec, indent=1))
        log("done", fmt, iid, outcome, f"agent {rec['agent_s']:.0f}s calls {calls}")
        return rec

    def instance(self, iid):
        """Pull the instance image once, run the agents of all pending formats on it in parallel, remove it."""
        todo = [f for f in self.urls if not (self.work / "runs" / f / iid / "done.json").exists()]
        if not todo:
            return
        for f in todo:
            while not self.server_ok(f):
                log("waiting for server", f)
                time.sleep(60)
        img, need = image(iid), self.estimate(iid)
        for _ in range(3):
            while True:  # wait until the image fits the disk budget
                with IMAGE_LOCK:
                    if self.images_bytes() + need <= self.a.image_budget:
                        r = self.sh([self.pm, "pull", "-q", img], timeout=1800)
                        break
                time.sleep(30)
            if r.returncode == 0:
                break
            log("pull failed", iid, r.stderr[-300:])
            time.sleep(60)
        with ThreadPoolExecutor(len(todo)) as ex:
            list(ex.map(lambda f: self.safe(self.agent_and_eval, f, iid), todo))
        ids = self.sh([self.pm, "ps", "-aq", "--filter", f"ancestor={img}"]).stdout.split()
        if ids:
            self.sh([self.pm, "rm", "-f", *ids], timeout=300)
        with IMAGE_LOCK:
            self.sh([self.pm, "rmi", "-f", img], timeout=600)

    def safe(self, fn, *args):
        try:
            return fn(*args)
        except Exception as e:
            log("INFRA_ERROR", *args, repr(e)[:300])


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("work", type=Path)
    p.add_argument("--endpoint", action="append", required=True, help="FORMAT=URL of the OpenAI API (…/v1)")
    p.add_argument("--podman", default="podman")
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--image-budget", type=float, default=14.5e9, help="bytes of instance images on disk")
    p.add_argument("--limit", type=int)
    a = p.parse_args()
    a.work.mkdir(parents=True, exist_ok=True)
    ids = instances()[: a.limit]
    (a.work / "instances.txt").write_text("\n".join(ids) + "\n")
    runner = Runner(a)
    log("instances", len(ids), "formats", list(runner.urls), "workers", a.workers)
    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(lambda i: runner.safe(runner.instance, i), ids))
    resolved = {
        f: sum(json.loads(p.read_text())["resolved"] for p in (a.work / "runs" / f).glob("*/done.json"))
        for f in runner.urls
    }
    log("SWEBENCH_DONE", json.dumps(resolved))


if __name__ == "__main__":
    main()
