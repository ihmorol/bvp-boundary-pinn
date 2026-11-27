"""Campaign scheduling: runs at once, out of memory beside others, the session limit, the code check.

None of this may change how a run trains: the first test trains the same run alone and beside
another and checks the networks are the same, bit for bit."""
import json
import os
import subprocess
import sys

import pytest
import torch

from bvp_pinn.config import ROOT
from bvp_pinn.pinn import campaign

torch.set_default_dtype(torch.float64)


def test_a_run_beside_another_trains_exactly_as_alone(tmp_path):
    """Two dry runs through the campaign, two at once in their own processes, against the same runs
    trained alone in this process: every weight is equal, and schedule.json says they shared."""
    from bvp_pinn.pinn.run import run
    shared, alone = tmp_path / "shared", tmp_path / "alone"
    env = {**os.environ, "BVP_RUNS_DIR": str(shared), "PYTHONPATH": str(ROOT / "src")}
    subprocess.run([sys.executable, "-m", "bvp_pinn.pinn.campaign", "--dry-run", "--arms", "B0", "B2",
                    "--seeds", "1", "--jobs", "2"], cwd=ROOT, env=env, check=True, capture_output=True)
    for arm in ("B0", "B2"):
        folder = run(arm, 1, dry_run=True, runs_root=alone, assess=False, campaign=campaign.campaign_name(True))
        torch.set_default_device("cpu")
        other = shared / folder.name
        a = torch.load(folder / "model.pt")
        b = torch.load(other / "model.pt")
        assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a), arm
        sched = json.loads((other / campaign.SCHEDULE).read_text())
        assert sched["jobs"] == 2 and sched["max_runs_at_once"] == 2
        assert json.loads((other / "selection.json").read_text())["accepted"] is False  # judged, tiny settings


class FakeJob:
    def __init__(self, clock, seconds, code=0):
        self.clock, self.end, self.code = clock, clock.t + seconds, code

    def poll(self):
        return self.code if self.clock.t >= self.end else None


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


def run_schedule(todo, jobs, durations, deadline=None, prior=None, codes=None):
    clock, started = Clock(), []
    codes = codes or {}

    def submit(item, shared):
        started.append((item[0], item[1], shared, clock.t))
        return FakeJob(clock, durations[item[0]], codes.pop((item[0], item[1]), 0))

    left = campaign.schedule(todo, jobs, submit, deadline=deadline, prior=prior, clock=clock, sleep=clock.sleep,
                             poll_seconds=10, say=lambda text: None)
    return started, left


def items(tmp_path, *runs):
    return [(arm, seed, tmp_path / f"{arm}-{seed}", False) for arm, seed in runs]


def test_schedule_keeps_the_slots_full_in_order(tmp_path):
    todo = items(tmp_path, ("B1", 0), ("B1", 1), ("B2", 0), ("B2", 1))
    started, left = run_schedule(todo, 2, {"B1": 100, "B2": 30})
    assert [(a, s) for a, s, *_ in started] == [("B1", 0), ("B1", 1), ("B2", 0), ("B2", 1)]
    assert started[0][3] == started[1][3] == 0 and all(shared for *_, shared, _ in started)
    assert left == []


def test_schedule_starts_only_runs_that_end_before_the_deadline(tmp_path):
    """Deadline 180 s. The first two long runs are expected to take 100 s x 1.6 (shared) and fit. When they end
    at 100 s, a third long run (now expected at the 100 s measured) would end at 200 s and does not start,
    but the two short ones fit: the session ends with finished runs, not cut ones."""
    todo = items(tmp_path, ("B1", 0), ("B1", 1), ("B1", 2), ("B2", 0), ("B2", 1))
    started, left = run_schedule(todo, 2, {"B1": 100, "B2": 30}, deadline=180, prior={"B1": 100, "B2": 30})
    names = [(a, s) for a, s, *_ in started]
    assert ("B1", 2) not in names and {("B2", 0), ("B2", 1)} <= set(names)
    assert [(a, s) for a, s, *_ in left] == [("B1", 2)]


def test_out_of_memory_beside_others_trains_again_alone(tmp_path):
    """Exit code OUT_OF_MEMORY_EXIT means no result: the run goes back in the queue and trains again with no
    other run beside it. A failure alone (exit 1) is a result and is not repeated."""
    todo = items(tmp_path, ("B1", 0), ("B1", 1), ("B2", 0))
    codes = {("B1", 1): campaign.OUT_OF_MEMORY_EXIT}
    started, left = run_schedule(todo, 2, {"B1": 100, "B2": 30}, codes=codes)
    again = [x for x in started if x[:2] == ("B1", 1)]
    assert len(again) == 2 and again[1][2] is False  # the second try is not shared
    others = [x for x in started if x[:2] != ("B1", 1)]
    t_alone = again[1][3]
    assert all(t + {"B1": 100, "B2": 30}[a] <= t_alone for a, _, _, t in others)  # it waits for the machine
    assert left == []


def test_out_of_memory_record_is_removed_and_marked(tmp_path):
    (tmp_path / "meta.json").write_text('{"result": "error: OutOfMemoryError: CUDA out of memory"}')
    campaign.forget_out_of_memory(tmp_path, "OutOfMemoryError: CUDA out of memory")
    assert not (tmp_path / "meta.json").exists() and (tmp_path / campaign.ALONE).exists()
    assert campaign.out_of_memory("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB")
    assert campaign.out_of_memory("RuntimeError: CUBLAS_STATUS_ALLOC_FAILED when calling cublasCreate")
    assert not campaign.out_of_memory("ValueError: non-finite loss")


def test_later_commits_changed_no_training_code():
    """The main run lets a later session use a newer commit only if METHOD_PATHS are the same. Commit 2895060
    (the main-run fixes) changed training code; everything after it, up to now, changed none."""
    sys.path.insert(0, str(ROOT / "tools"))
    import main_run
    known = subprocess.run(["git", "cat-file", "-e", "2895060^{commit}"], cwd=ROOT).returncode == 0
    if not known:
        pytest.skip("needs the git history of the branch")
    pipe = main_run.Pipeline.__new__(main_run.Pipeline)
    assert pipe.method_changes("2895060") == []
    assert "src/bvp_pinn/pinn/run.py" in pipe.method_changes("2895060~1")
