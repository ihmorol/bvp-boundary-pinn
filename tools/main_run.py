"""The main run from start to finish: checks, reference, PINN campaign, evaluation. Resumable.

    python tools/main_run.py --settings run_settings.json

Made for the Colab notebook tools/colab_main_run.ipynb; it runs the same way on a laptop. The
settings file names the output folder (on Colab, a Google Drive folder) and the few run choices;
see SETTINGS_DEFAULTS. Steps, in order (AGENTS.md section 4: tests, then reference, then
training, then evaluation):

  1 tests       pytest and the record check
  2 reference   the classical reference from scratch, the box-scheme engine, the branch search,
                the L error bars; then the target is checked (G3 at fixed L, decaying far field)
  3 smoke       a dry campaign (tiny settings, seed 0, every arm) on the training device, then its
                evaluation: the whole chain once, in a minute, before hours of training
  4 timing      measured Adam and Levenberg-Marquardt step times on this device, and the run-time
                estimate they give for the campaign
  5 campaign    every arm and seed of configs/pinn.json; run folders go to <out>/runs
  6 evaluate    metrics, solve_bvp timing, figures and tables; results go to <out>/results
  7 package     configs, environment and a README into <out>

Each finished step writes <out>/steps/<step>.json, and a restart skips it. The campaign step always
starts again and skips the runs that are finished, so a restart resumes run by run, and arms or seeds
added later still run. Run one session at a time per output folder. Every step's output is shown and
kept in <out>/logs/<step>.log. A failing step stops the whole run with its name and the end of its log.

The first start records the git commit and a hash of configs/pinn.json in <out>/steps/start.json.
A later start stops if the training settings differ, or if its commit differs from that one in any
file that decides how a run trains or is judged (METHOD_PATHS): one campaign, one training code. Only
scheduling (pinn/campaign.py), evaluation, tools, tests and docs may differ between sessions.

session_hours (for Colab, whose free sessions end after a few hours): no run starts unless it is
expected to end before session_start + session_hours, so the last hours go to runs that finish.
During the campaign a line every ten minutes shows how busy the GPU and the CPUs are (logs/usage.log).
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SETTINGS_DEFAULTS = {
    "out_dir": str(ROOT / "main-run"),  # everything this run keeps
    "device": "auto",                   # auto (GPU if there is one), cuda or cpu
    "jobs": "auto",                     # runs at once: auto is 2 on a GPU (1 with one CPU), else the CPU count (at most 4)
    "arms": None,                       # None: the campaign arms of configs/pinn.json
    "seeds": None,                      # None: the campaign seeds of configs/pinn.json
    "campaign_name": None,              # None: keep configs/pinn.json; else set campaign.name
    "pinn_common": {},                  # training settings to write into configs/pinn.json "common"
    "run_tests": True,
    "smoke_test": True,
    "measure_timing": True,
    "classical_repeats": 5,
    "session_hours": None,              # None: no limit; else start no run expected to end after this many hours
    "session_start": None,              # Unix time the session started (the notebook sets it); None: now
}

# Files that decide how a run trains and how the acceptance rule judges it. A later session may use a newer
# commit only if none of these differ from the first session's commit. Scheduling (pinn/campaign.py) and
# evaluation (rebuilt for every run at the end) may change; so may tools, tests and docs.
METHOD_PATHS = ["src/bvp_pinn", ":(exclude)src/bvp_pinn/pinn/campaign.py", ":(exclude)src/bvp_pinn/evaluation",
                ":(exclude,glob)**/*.md", "configs", "engine/bvp_engine.cpp", "uv.lock", "pyproject.toml"]
GPU_GB_PER_RUN = 3.0   # one run of any arm: about 1.6 to 1.9 GB of tensors (measured on CPU) plus the CUDA context
RESERVE_SECONDS = 600  # kept free before the session limit: saving, the acceptance rule, a slow first epoch


class StepFailed(SystemExit):
    pass


def now():
    return datetime.datetime.now().isoformat(timespec="seconds")


def sha256_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


class Pipeline:
    def __init__(self, settings):
        self.s = {**SETTINGS_DEFAULTS, **settings}
        self.t0 = time.time()
        self.out = Path(self.s["out_dir"]).resolve()
        for sub in ("steps", "logs", "runs", "results"):
            (self.out / sub).mkdir(parents=True, exist_ok=True)
        self.py = sys.executable
        # MPLBACKEND: a notebook's inline backend (Colab sets one) is not installed here and stops matplotlib
        self.env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONUNBUFFERED": "1", "MPLBACKEND": "Agg"}
        for var in ("BVP_RUNS_DIR", "BVP_RESULTS_DIR"):
            self.env.pop(var, None)  # tests and the reference never see the campaign folders

    # ---- bookkeeping -------------------------------------------------------------------------
    def marker(self, step):
        return self.out / "steps" / f"{step}.json"

    def done(self, step):
        return self.marker(step).exists()

    def mark(self, step, **info):
        from bvp_pinn.pinn.runs import write_json
        write_json(self.marker(step), {"step": step, "finished": now(), **info})

    def say(self, text):
        print(text, flush=True)

    def sh(self, step, args, env=None, cwd=ROOT):
        """Run one command, show its output as it comes and append it to logs/<step>.log."""
        log = self.out / "logs" / f"{step}.log"
        self.say(f"   $ {' '.join(str(a) for a in args)}")
        with open(log, "a") as fh:
            fh.write(f"\n===== {now()} $ {' '.join(str(a) for a in args)}\n")
            proc = subprocess.Popen([str(a) for a in args], cwd=cwd, env=env or self.env, text=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)
            tail = []
            for line in proc.stdout:
                print(line, end="", flush=True)
                fh.write(line)
                tail = (tail + [line.rstrip()])[-25:]
            code = proc.wait()
        if code:
            raise StepFailed(f"\nSTEP FAILED: {step} (exit code {code}). Full log: {log}\nLast lines:\n"
                             + "\n".join(tail))

    # ---- settings, code version, device --------------------------------------------------------
    def apply_pinn_settings(self):
        """Write the notebook's training settings into configs/pinn.json, then check every arm."""
        path = ROOT / "configs" / "pinn.json"
        cfg = json.loads(path.read_text())
        unknown = sorted(set(self.s["pinn_common"]) - set(cfg["common"]))
        if unknown:
            raise StepFailed(f"pinn_common has settings configs/pinn.json does not define: {unknown}")
        changed = {k: (cfg["common"][k], v) for k, v in self.s["pinn_common"].items() if cfg["common"][k] != v}
        cfg["common"].update(self.s["pinn_common"])
        if self.s["campaign_name"] and self.s["campaign_name"] != cfg["campaign"]["name"]:
            changed["campaign.name"] = (cfg["campaign"]["name"], self.s["campaign_name"])
            cfg["campaign"]["name"] = self.s["campaign_name"]
        if changed:
            path.write_text(json.dumps(cfg, indent=2) + "\n")
            self.say(f"Settings changed from configs/pinn.json (old, new): {changed}")
        from bvp_pinn.pinn.runs import resolve
        for arm in cfg["arms"]:
            resolve(arm)  # refuses physical values, unknown keys, another L
        return cfg

    def git(self, *args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()

    def method_changes(self, old):
        """Files of METHOD_PATHS that differ between commit `old` and the checked-out commit."""
        r = subprocess.run(["git", "diff", "--name-only", old, "HEAD", "--", *METHOD_PATHS], cwd=ROOT,
                           capture_output=True, text=True)
        if r.returncode:
            raise StepFailed(f"Cannot compare this code with the first session's commit {old}: "
                             f"{(r.stderr or r.stdout).strip()}. Fetch the full history of the branch.")
        return r.stdout.split()

    def check_same_code(self, cfg):
        """The first start pins the commit and the training settings. A later start must have the same settings,
        no uncommitted code changes, and the same training code (METHOD_PATHS) as that commit."""
        here = {"commit": self.git("rev-parse", "HEAD"),
                "pinn_json_sha256": sha256_text(json.dumps(cfg, sort_keys=True)),
                # code only: the reference step rewrites data files such as engine/out/*.csv and reference/
                "code_changes": self.git("status", "--porcelain", "--", "src", "tools", "tests", "engine/bvp_engine.cpp")}
        start = self.out / "steps" / "start.json"
        if not start.exists():
            from bvp_pinn.pinn.runs import write_json
            write_json(start, {**here, "started": now()})
            return
        first = json.loads(start.read_text())
        problems = []
        if here["pinn_json_sha256"] != first["pinn_json_sha256"]:
            problems.append("other training settings (configs/pinn.json or PINN_SETTINGS)")
        if here["code_changes"] != first["code_changes"]:
            problems.append(f"uncommitted code changes: {here['code_changes'] or 'none'} (first session: "
                            f"{first['code_changes'] or 'none'})")
        changed = self.method_changes(first["commit"]) if here["commit"] != first["commit"] else []
        if changed:
            problems.append(f"other training code than commit {first['commit'][:8]}: {', '.join(changed)}")
        if problems:
            raise StepFailed("This output folder was started with other code or settings:\n  - " + "\n  - ".join(problems)
                             + "\nUse a commit whose training code matches (set REPO_COMMIT in the notebook to "
                             f"{first['commit']}), or start a new output folder.")
        if here["commit"] != first["commit"]:
            self.say(f"Code: commit {here['commit'][:8]}; the first session used {first['commit'][:8]}. Training code, "
                     "acceptance rule and settings are the same in both; only scheduling, evaluation, tools, tests or "
                     "docs differ.")

    def pick_device(self):
        import torch
        want = self.s["device"]
        has_gpu = torch.cuda.is_available()
        if want == "cuda" and not has_gpu:
            raise StepFailed("device is cuda but PyTorch sees no GPU")
        device = "cuda" if want == "auto" and has_gpu else ("cpu" if want == "auto" else want)
        jobs, cpus = self.s["jobs"], os.cpu_count() or 1
        if jobs == "auto":
            # On a GPU each run still needs a CPU to send it its many small operations: free Colab has 2 CPUs.
            jobs = min(2, cpus) if device == "cuda" else max(1, min(4, cpus))
        jobs = int(jobs)
        gpu = torch.cuda.get_device_name(0) if has_gpu else None
        if device == "cuda":  # a real float64 product on the GPU, so a broken driver fails here, not mid-run
            x = torch.ones(64, 64, dtype=torch.float64, device="cuda")
            assert float((x @ x).sum()) == 64.0**3
        self.say(f"Device: {device}{f' ({gpu})' if gpu else ''}; GPU available: {has_gpu}; runs at once: {jobs}; "
                 f"CPU count: {cpus}")
        if device == "cuda":
            total = torch.cuda.get_device_properties(0).total_memory / 2**30
            self.say(f"GPU memory {total:.1f} GB; one run needs about {GPU_GB_PER_RUN:.0f} GB.")
            if jobs * GPU_GB_PER_RUN > total:
                self.say(f"WARNING: {jobs} runs at once may not fit. A run that runs out of memory beside others "
                         "has no result and trains again alone, so nothing is lost but time.")
        if jobs > cpus:
            self.say(f"WARNING: {jobs} runs at once on {cpus} CPUs: they will slow each other down.")
        return device, jobs, gpu

    def check_same_device(self, device, gpu):
        """Accuracy does not depend on the device beyond rounding, but run times do: warn if it changes."""
        path = self.out / "steps" / "devices.json"
        seen = json.loads(path.read_text()) if path.exists() else []
        here = f"{device} ({gpu})" if device == "cuda" else device
        if seen and here not in seen:
            self.say(f"\nWARNING: earlier sessions trained on {seen}, this one on {here}. Accuracy is not affected, "
                     "but run times of different devices do not compare; the cost table lists the device per arm.\n")
        if here not in seen:
            path.write_text(json.dumps(seen + [here]) + "\n")

    def backfill_schedule(self):
        """Runs finished before schedule.json existed have none. If every earlier session trained one run at a time
        (jobs in its environment-*.json), each of those runs had the machine to itself: record that."""
        from bvp_pinn.pinn.campaign import SCHEDULE
        from bvp_pinn.pinn.runs import write_json
        envs = sorted(self.out.glob("environment-*.json"))
        if not envs or {json.loads(p.read_text()).get("jobs") for p in envs} != {1}:
            return
        for folder in sorted((self.out / "runs").glob("bvp-*")):
            meta = folder / "meta.json"
            if meta.exists() and not (folder / SCHEDULE).exists():
                m = json.loads(meta.read_text())
                write_json(folder / SCHEDULE, {"jobs": 1, "max_runs_at_once": 1, "device": m.get("device"),
                                               "seconds": m.get("wall_seconds"),
                                               "note": "written later: every session before this file existed trained "
                                                       f"one run at a time ({', '.join(p.name for p in envs)})"})

    def environment(self, device, jobs, gpu):
        import matplotlib
        import numpy
        import scipy
        import torch
        info = {"time": now(), "python": platform.python_version(), "platform": platform.platform(),
                "processor": platform.processor(), "cpu_count": os.cpu_count(), "device": device, "jobs": jobs,
                "gpu": gpu, "torch": torch.__version__, "torch_cuda": torch.version.cuda, "numpy": numpy.__version__,
                "scipy": scipy.__version__, "matplotlib": matplotlib.__version__,
                "commit": self.git("rev-parse", "HEAD"), "branch": self.git("rev-parse", "--abbrev-ref", "HEAD"),
                "remote": self.git("remote", "get-url", "origin"), "dirty": self.git("status", "--porcelain"),
                "settings": self.s,
                "packages": sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions())}
        name = "environment-" + info["time"].replace(":", "") + ".json"
        (self.out / name).write_text(json.dumps(info, indent=2) + "\n")
        self.say(f"Environment recorded in {self.out / name}")

    # ---- steps ---------------------------------------------------------------------------------
    def step_tests(self):
        if not self.s["run_tests"]:
            return {"skipped": "run_tests is off"}
        self.sh("tests", [self.py, "-m", "pytest", "-q", "-p", "no:cacheprovider"])
        self.sh("tests", [self.py, "-I", "tools/check_records.py"])
        return {}

    def step_reference(self):
        py = self.py
        self.sh("reference", [py, "-m", "bvp_pinn.reference.run"])
        self.sh("reference", [py, "-c", "from bvp_pinn.pinn.estimate import ensure_engine; print(ensure_engine())"])
        self.sh("reference", [ROOT / "engine" / "bvp_engine"])               # engine_convergence.csv
        self.sh("reference", [ROOT / "engine" / "bvp_engine", "--deflate"])  # engine_branches.csv
        self.sh("reference", [py, "tools/reference_error_bars.py"])
        self.sh("reference", [py, "tools/l_choice.py"])
        summary = self.check_reference()
        for name in ("reference", "engine_out"):
            src = ROOT / ("reference" if name == "reference" else "engine/out")
            shutil.copytree(src, self.out / name, dirs_exist_ok=True)
        return summary

    def check_reference(self):
        """Stop unless branch A is a usable target; print what the paper needs from the study."""
        from bvp_pinn.evaluation.evaluate import REFERENCES, load_references
        study = json.loads((ROOT / "reference" / "study.json").read_text())
        names = [b["name"] for b in study["branch_scan"]["branches"]]
        load_references()  # raises unless A is at the reference L, passes G3 and decays
        meta = json.loads((REFERENCES["A"] / "meta.json").read_text())
        c = meta["checks"]
        branches = (ROOT / "engine" / "out" / "engine_branches.csv").read_text().strip().splitlines()[1:]
        self.say(f"Reference: branches {names} from the scan, {len(branches)} from the deflation search; "
                 f"target {REFERENCES['A'].name}, max |B| {c['V7_max_abs_bc']:.1e}, m {c['far_decay_exponent_m']:.4f}, "
                 f"RK check pass {c['V10_pass']}, tol pass {c['V4_tol_two_tightest']}")
        self.say(f"   change of the wall values from L = 20 to the next L: {c['V5_core_change_to_next_L']:.2e} "
                 f"(bar 1e-4; L error bars in reference/l_choice.md and error_bars.md)")
        return {"branches": names, "deflation_roots": len(branches), "target": REFERENCES["A"].name,
                "wall_values": meta["wall_values"], "checks": {k: v for k, v in c.items() if not isinstance(v, dict)}}

    def restore_reference(self):
        """After a restart on a fresh machine: put the saved reference back where evaluation reads it."""
        shutil.copytree(self.out / "reference", ROOT / "reference", dirs_exist_ok=True)
        shutil.copytree(self.out / "engine_out", ROOT / "engine" / "out", dirs_exist_ok=True)
        from bvp_pinn.evaluation.evaluate import load_references
        load_references()  # checks the hashes and the target again

    def campaign_env(self, runs, results):
        return {**self.env, "BVP_RUNS_DIR": str(runs), "BVP_RESULTS_DIR": str(results)}

    def step_smoke(self, device, jobs):
        if not self.s["smoke_test"]:
            return {"skipped": "smoke_test is off"}
        scratch = self.out / "smoke"
        shutil.rmtree(scratch, ignore_errors=True)
        env = self.campaign_env(scratch / "runs", scratch / "results")
        self.sh("smoke", [self.py, "-m", "bvp_pinn.pinn.campaign", "--dry-run", "--device", device, "--seeds", "0",
                          "--jobs", str(jobs)], env)
        from bvp_pinn.pinn.campaign import campaign_name
        name = campaign_name(dry_run=True)
        for mod in ("evaluate", "plots"):
            self.sh("smoke", [self.py, "-m", f"bvp_pinn.evaluation.{mod}", "--campaign", name], env)
        made = sorted(p.name for p in (scratch / "results" / name).rglob("*") if p.is_file())
        self.say(f"Smoke test made {len(made)} result files. Its numbers mean nothing (tiny settings).")
        return {"device": device, "jobs": jobs, "files": made}

    def step_timing(self, device, jobs, cfg, arms, seeds):
        if not self.s["measure_timing"]:
            return {"skipped": "measure_timing is off"}
        from bvp_pinn.pinn.runs import resolve
        times = {}
        for arm in ("B0", "B1"):
            out = self.out / "logs" / f"timing-{device}-{arm}.json"
            self.sh("timing", [self.py, "tools/time_pinn_step.py", "--arm", arm, "--device", device, "--jets-only",
                               "--lm", "--reps", "20", "--json", out])
            times[arm] = json.loads(out.read_text())
        est = {}
        for arm in arms:
            c = resolve(arm)
            t = times["B1" if c["structure"] == "windows" else "B0"]
            step = t["jets_full_step_s"]
            # Adam steps, then L-BFGS (PyTorch allows 1.25 loss evaluations per iteration), then LM.
            # Staged arms train on part of the points in early stages, so this is an upper bound there.
            secs = c["adam_steps"] * step * 1.05 + 1.25 * c["lbfgs_iters"] * step + c["lm_iters"] * t["lm_step_s"]
            est[arm] = secs
        total = sum(est[a] for a in arms) * len(seeds)
        hours = {a: round(v / 3600, 2) for a, v in est.items()}
        self.say(f"Measured: Adam step B0 {times['B0']['jets_full_step_s'] * 1e3:.0f} ms, B1 "
                 f"{times['B1']['jets_full_step_s'] * 1e3:.0f} ms; LM step B0 {times['B0']['lm_step_s']:.1f} s, "
                 f"B1 {times['B1']['lm_step_s']:.1f} s on {device}")
        self.say(f"Estimated hours per run: {hours}. Campaign of {len(arms)} arms x {len(seeds)} seeds: about "
                 f"{total / 3600:.1f} h of runs, about {total / 3600 / jobs:.1f} h of wall time with {jobs} at once "
                 "(more if runs at once slow each other).")
        return {"device": device, "measured": times, "hours_per_run": hours, "campaign_hours": total / 3600,
                "wall_hours": total / 3600 / jobs}

    def deadline(self):
        if not self.s["session_hours"]:
            return None
        return (self.s["session_start"] or self.t0) + 3600 * float(self.s["session_hours"]) - RESERVE_SECONDS

    def run_time_priors(self, device, arms):
        """Seconds one run of each arm takes alone on this device: the median of this folder's finished campaign
        runs that had the machine to themselves, else the timing step's estimate, scaled by how far that estimate
        was from the runs measured here."""
        from bvp_pinn.pinn.campaign import SCHEDULE
        from bvp_pinn.pinn.runs import read_config
        tag, alone = read_config()["campaign"]["name"], {}
        for folder in (self.out / "runs").glob("bvp-*"):
            read = lambda name: json.loads((folder / name).read_text()) if (folder / name).exists() else {}
            cfg, meta, sched = read("config.json"), read("meta.json"), read(SCHEDULE)
            if (cfg.get("campaign") == tag and meta.get("device") == device and (folder / "model.pt").exists()
                    and sched.get("max_runs_at_once") == 1 and meta.get("wall_seconds")):
                alone.setdefault(cfg["arm"], []).append(float(meta["wall_seconds"]))
        measured = {a: statistics.median(v) for a, v in alone.items()}
        timing = self.marker(f"timing-{device}")
        hours = json.loads(timing.read_text()).get("hours_per_run", {}) if timing.exists() else {}
        estimate = {a: 3600 * h for a, h in hours.items() if h}
        ratios = [measured[a] / estimate[a] for a in measured if a in estimate]
        scale = statistics.median(ratios) if ratios else 1.0
        prior = {a: measured.get(a, estimate.get(a, 0) * scale) for a in arms if a in measured or a in estimate}
        source = {a: "measured here" if a in measured else f"timing estimate x {scale:.2f}" for a in prior}
        return prior, source

    def usage(self):
        """One line: how busy the GPU is (share of time a GPU operation ran), its memory, and the CPU load."""
        parts = []
        if shutil.which("nvidia-smi"):
            q = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                                "--format=csv,noheader,nounits"], capture_output=True, text=True)
            if q.returncode == 0 and q.stdout.strip():
                busy, used, total = (x.strip() for x in q.stdout.splitlines()[0].split(","))
                parts.append(f"GPU busy {busy} % of the time, GPU memory {float(used) / 1024:.1f} of "
                             f"{float(total) / 1024:.1f} GB")
        parts.append(f"CPU load {os.getloadavg()[0]:.1f} on {os.cpu_count()} CPUs")
        return "; ".join(parts)

    def monitor(self, stop, every=600):
        log = self.out / "logs" / "usage.log"
        while not stop.wait(every):
            line = f"[usage {now()}] {self.usage()}"
            self.say("   " + line)
            with open(log, "a") as fh:
                fh.write(line + "\n")

    def step_campaign(self, device, jobs, arms, seeds):
        env = self.campaign_env(self.out / "runs", self.out / "results")
        cmd = [self.py, "-m", "bvp_pinn.pinn.campaign", "--arms", *arms, "--seeds", *map(str, seeds),
               "--device", device, "--jobs", str(jobs)]
        deadline = self.deadline()
        if deadline:
            prior, source = self.run_time_priors(device, arms)
            cmd += ["--deadline", f"{deadline:.0f}"] + (["--expect", *(f"{a}={v:.0f}" for a, v in prior.items())] if prior else [])
            self.say(f"Session limit {self.s['session_hours']} h: no run starts unless it is expected to end before "
                     f"{datetime.datetime.fromtimestamp(deadline):%H:%M} (machine time, {RESERVE_SECONDS // 60} min kept free). "
                     "Hours of one run alone: " + ", ".join(f"{a} {v / 3600:.2f} ({source[a]})" for a, v in prior.items()))
        stop = threading.Event()
        threading.Thread(target=self.monitor, args=(stop,), daemon=True).start()
        try:
            self.sh("campaign", cmd, env)
        finally:
            stop.set()
        from bvp_pinn.pinn.runs import resolve, run_id, read_config
        tag = read_config()["campaign"]["name"]
        missing = [f"{a} seed {s}" for a in arms for s in seeds
                   if not (self.out / "runs" / run_id(resolve(a, campaign=tag), s) / "meta.json").exists()]
        if missing and deadline:
            return {"device": device, "jobs": jobs, "left": missing}
        if missing:
            raise StepFailed(f"STEP FAILED: campaign. These runs have no meta.json: {missing}")
        return {"device": device, "jobs": jobs, "arms": arms, "seeds": seeds}

    def step_evaluate(self):
        env = self.campaign_env(self.out / "runs", self.out / "results")
        self.sh("evaluate", [self.py, "-m", "bvp_pinn.evaluation.evaluate"], env)
        self.sh("evaluate", [self.py, "-m", "bvp_pinn.evaluation.classical", "--repeats", str(self.s["classical_repeats"])], env)
        self.sh("evaluate", [self.py, "-m", "bvp_pinn.evaluation.plots"], env)
        return {}

    def step_package(self, cfg):
        shutil.copytree(ROOT / "configs", self.out / "configs", dirs_exist_ok=True)
        name = cfg["campaign"]["name"]
        readme = f"""Main run of the five-field boundary problem, written by tools/main_run.py.

steps/            one file per finished step; start.json holds the commit and settings hash
logs/             the full output of every step (tests, reference, smoke, timing, campaign, evaluate);
                  usage.log: GPU and CPU use during the campaign, every ten minutes
environment-*.json  machine, packages, GPU and settings of every session
configs/          the configs the run used (physical case, reference settings, PINN settings)
reference/        the classical reference built in this run: branch folders, study, L error bars
engine_out/       box-scheme convergence and the deflation branch search
runs/             one folder per PINN run: config, history, models, verdict, metrics, predictions;
                  schedule.json: how many runs trained beside it
results/{name}/   summary.csv, failures.md, settings.json, accuracy.md (TB3), wall_values.md (TB2),
                  cost.md (TX1), classical_timing.json, figures/ (FB1 to FB5, FX1)
smoke/            the dry pipeline check; its numbers mean nothing

Commit: {self.git('rev-parse', 'HEAD')}
"""
        (self.out / "README.txt").write_text(readme)
        return {"campaign": name}

    # ---- the whole run -------------------------------------------------------------------------
    def run(self):
        t0 = time.time()
        self.say(f"Main run, output folder {self.out}")
        cfg = self.apply_pinn_settings()
        self.check_same_code(cfg)
        device, jobs, gpu = self.pick_device()
        self.backfill_schedule()  # before this session's environment file exists
        self.environment(device, jobs, gpu)
        arms = self.s["arms"] or cfg["campaign"]["arms"]
        seeds = self.s["seeds"] if self.s["seeds"] is not None else cfg["campaign"]["seeds"]
        plan = [("tests", lambda: self.step_tests()),
                ("reference", lambda: self.step_reference()),
                # again for every new commit and job count: the scheduling code is checked on this device first
                (f"smoke-{device}-x{jobs}-{self.git('rev-parse', '--short=8', 'HEAD')}", lambda: self.step_smoke(device, jobs)),
                (f"timing-{device}", lambda: self.step_timing(device, jobs, cfg, arms, seeds)),
                ("campaign", lambda: self.step_campaign(device, jobs, arms, seeds)),
                ("evaluate", lambda: self.step_evaluate()),
                ("package", lambda: self.step_package(cfg))]
        self.check_same_device(device, gpu)
        for k, (step, fn) in enumerate(plan, 1):
            # The campaign skips its own finished runs (so new arms or seeds still run); evaluate and package
            # are cheap and rebuild everything. The other steps run once per output folder (smoke, timing per device).
            if step not in ("campaign", "evaluate", "package") and self.done(step):
                self.say(f"\n== [{k}/{len(plan)}] {step}: already finished "
                         f"({json.loads(self.marker(step).read_text())['finished']}), skipped")
                if step == "reference":
                    self.restore_reference()
                continue
            self.say(f"\n== [{k}/{len(plan)}] {step}: started {now()}")
            s0 = time.time()
            info = fn()
            if step == "campaign" and info.get("left"):
                self.say(f"\n== Session limit: {len(info['left'])} runs left ({', '.join(info['left'])}). Nothing is lost. "
                         "Open the notebook again in a new session (this or another Google account, the same Drive "
                         "folder) and Run all to continue. Evaluation runs after the last run.")
                return
            self.mark(step, seconds=round(time.time() - s0, 1), **(info or {}))
            self.say(f"== [{k}/{len(plan)}] {step}: finished in {(time.time() - s0) / 60:.1f} min")
        self.say(f"\nAll steps finished ({(time.time() - t0) / 3600:.2f} h this session). "
                 f"Results: {self.out / 'results' / cfg['campaign']['name']}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--settings", type=Path, help="JSON file with any of the keys of SETTINGS_DEFAULTS")
    args = ap.parse_args(argv)
    settings = json.loads(args.settings.read_text()) if args.settings else {}
    unknown = sorted(set(settings) - set(SETTINGS_DEFAULTS))
    if unknown:
        raise SystemExit(f"unknown settings {unknown}; known: {sorted(SETTINGS_DEFAULTS)}")
    Pipeline(settings).run()


if __name__ == "__main__":
    main()
