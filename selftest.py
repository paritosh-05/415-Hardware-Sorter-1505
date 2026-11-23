"""Self-test on a simulated belt (no camera needed). Run:  python sorter.py selftest

Teaches the six parts from simulated passes, then checks:
  1. a mixed, well-spaced stream is sorted with no part in a wrong bin
  2. the gap rule: parts too close for the collector both go to re-sort, and the simulated
     collector is never caught moving
  3. abuse cases (stacked, touching, foreign, dropped in, rolling) all go to re-sort
  4. a lighting change sends everything to re-sort until it recovers
Works in a temporary folder, so it never touches your own teach/ model/ config files.
See sim.py for what the simulation does and does not reproduce.
"""
import os
import sys
import tempfile
import time

import cv2
import numpy as np

from collector import SimCollector
from config import RESORT, Config
from pipeline import BG_FILE, Pipeline, build_models
from segment import Segmenter
from sim import Sim

NAMES = ["bolt-0.25", "nut-0.25", "washer-0.25", "bolt-0.375", "nut-0.375", "washer-0.375"]


def make_cfg(sim):
    cfg = Config()
    cfg.px_per_mm, cfg.fps = sim.ppm, sim.fps
    bg = sim.background()
    cv2.imwrite(BG_FILE, bg)
    cfg.lane = Segmenter(cfg).auto_lane(bg, 2.0)
    return cfg


def lane_len(cfg, sim):
    return cfg.lane_len_mm((sim.size[1], sim.size[0]))


class Logged:
    """A pipeline that keeps its log rows in memory."""

    def __init__(self, cfg, **kw):
        self.pipe, self.rows = Pipeline(cfg, quiet=True, **kw), []
        orig = self.pipe._dropped

        def hook(job):
            rec = self.pipe.records.get(job.id)
            orig(job)
            if rec is not None:
                self.rows.append(rec)
        self.pipe._dropped = hook

    def run(self, sim, light=None, timing=None):
        for frame, t in sim.frames(light):
            t0 = time.perf_counter()
            self.pipe.process(frame, t)
            if timing is not None:
                timing.append((time.perf_counter() - t0) * 1000)
        self.pipe.finish()
        return self


def match(sim, rows, cfg):
    """Pair each simulated part with the log row decided nearest its decision-line crossing."""
    x_hand = cfg.handoff_frac * lane_len(cfg, sim)
    out = []
    for p in sim.parts:
        tc = sim.t_cross(p, x_hand)
        best = min(rows, key=lambda r: abs(r["t_decide"] - tc), default=None)
        out.append((p, best if best and abs(best["t_decide"] - tc) < 0.25 else None))
    return out


def score(pairs):
    c, why = dict(n=0, correct=0, resort=0, wrong=0, untracked=0), {}
    for p, r in pairs:
        c["n"] += 1
        if r is None:
            c["untracked"] += 1
        elif r["final"] == RESORT:
            c["resort"] += 1
            k = r["final_reason"].split(" (")[0]
            why[k] = why.get(k, 0) + 1
        elif r["final"] == p["label"]:
            c["correct"] += 1
        else:
            c["wrong"] += 1
    return c, why


def main(speed=100.0, fps=60, n_teach=20, n_mixed=120, n_gaps=90, reps=4):
    here = os.getcwd()
    work = tempfile.mkdtemp(prefix="sorter_selftest_")
    os.chdir(work)
    rng = np.random.default_rng(7)
    ok_all, t_start = True, time.time()

    def check(name, ok, detail):
        nonlocal ok_all
        ok_all &= bool(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    try:
        print(f"Simulated belt: {speed:.0f} mm/s, {fps} fps. Teaching {n_teach} passes of each part...")
        for name in NAMES:
            sim = Sim(speed=speed, fps=fps, seed=sum(map(ord, name)))
            cfg = make_cfg(sim)
            for i in range(n_teach):
                sim.add(name, 0.2 + i * 0.9)
            pipe = Pipeline(cfg, mode="teach", label=name, quiet=True)
            for frame, t in sim.frames():
                pipe.process(frame, t)
            pipe.finish()
        report = build_models(cfg, quiet=True)
        check("teach + build", "WRONG" not in report, "all six parts taught; models built")

        # 1 ---- mixed stream
        sim = Sim(speed=speed, fps=fps, seed=11); cfg = make_cfg(sim)
        t = 0.2
        for _ in range(n_mixed):
            sim.add(NAMES[rng.integers(6)], t); t += rng.uniform(0.75, 1.3) + 40.0 / speed   # gap + a part length
        tm = []
        L = Logged(cfg, mode="run", link=SimCollector(cfg)).run(sim, timing=tm)
        c, why = score(match(sim, L.rows, cfg))
        check("mixed stream", c["wrong"] == 0 and c["untracked"] == 0 and c["correct"] >= 0.9 * c["n"],
              f"{c['correct']}/{c['n']} in the correct bin, {c['resort']} re-sorted {why or ''}, {c['wrong']} WRONG BIN")
        check("speed", np.median(tm) < 1000.0 / 120,
              f"{np.median(tm):.1f} ms per frame (median), {np.percentile(tm, 99):.1f} ms worst 1% "
              f"(120 fps allows 8.3 ms)")
        x_drop = cfg.handoff_frac * lane_len(cfg, sim) + cfg.drop_distance_mm
        err = [1000 * (r["t_drop"] - sim.t_cross(p, x_drop) - cfg.fall_ms / 1000)
               for p, r in match(sim, L.rows, cfg) if r]
        slack = cfg.min_gap_ms - L.pipe.link.move_ms      # time to spare in each gap beyond the collector's move
        check("landing-time prediction", np.max(np.abs(err)) < 0.25 * slack,
              f"worst error {np.max(np.abs(err)):.1f} ms over {cfg.drop_distance_mm:.0f} mm of belt "
              f"(must be under a quarter of the {slack:.0f} ms spare in each gap)")

        # 2 ---- gap rule
        sim = Sim(speed=speed, fps=fps, seed=12); cfg = make_cfg(sim)
        t = 0.2
        for _ in range(n_gaps):
            sim.add(NAMES[rng.integers(6)], t); t += rng.choice([0.25, 0.4, 0.55, 0.65, 0.8, 1.2])
        L = Logged(cfg, mode="run", link=SimCollector(cfg, move_ms=500)).run(sim)
        pairs = match(sim, L.rows, cfg)
        c, why = score(pairs)
        # independent check: where is a simulated 500 ms collector when each part REALLY lands?
        # "missed" = the scheduler itself reported it could not protect that part (it turned up after the
        # part in front was already committed). Everything else must be safe.
        moving = bad_slot = 0
        for p, r in pairs:
            if r and r["missed"]:
                continue
            at = L.pipe.link.slot_at(sim.t_cross(p, x_drop) + cfg.fall_ms / 1000)
            moving += at is None
            bad_slot += at not in (None, cfg.resort_slot, cfg.slots[p["label"]])
        rows = sorted(L.rows, key=lambda r: r["t_drop"])
        viol = sum(1 for a, b in zip(rows, rows[1:]) if a["slot"] != b["slot"] and not b["missed"] and
                   (b["t_drop"] - a["t_drop"]) * 1000 - (a["window_ms"] + b["window_ms"]) / 2
                   < cfg.gap_needed_ms(a["slot"], b["slot"]) - 1)
        missed = L.pipe.sched.counts["missed"]
        check("gap rule", c["wrong"] == 0 and viol == 0 and moving == 0 and bad_slot == 0 and missed <= 0.03 * c["n"],
              f"parts 0.25-1.2 s apart: {c['correct']} sorted, {c['resort']} re-sorted, {c['wrong']} WRONG BIN; "
              f"different-slot windows closer than the collector needs: {viol}; real landings while the "
              f"collector was turning: {moving}; real landings in another part's slot: {bad_slot}; "
              f"reported as missed: {missed}")

        # 3 ---- abuse
        sim = Sim(speed=speed, fps=fps, seed=13); cfg = make_cfg(sim)
        t = 0.3

        def case(kind, items):
            nonlocal t
            for name, dt, kw in items:
                p = sim.add(name, t + dt, **kw)
                p["case"] = kind
                if kind != "normal":
                    p["label"] = "must-resort"
            t += max(1.5, 150.0 / speed)                 # keep the cases out of each other's way
        for _ in range(reps):
            y = rng.uniform(22, 32)
            case("nut sitting on a washer", [("washer-0.375", 0, dict(y_mm=y)), ("nut-0.25", 0, dict(y_mm=y))])
            case("two parts touching", [("nut-0.375", 0, dict(y_mm=y)), ("washer-0.25", 16.0 / speed, dict(y_mm=y + 2))])
            case("bolt lying on a washer", [("washer-0.375", 0, dict(y_mm=y)), ("bolt-0.25", 0.05, dict(y_mm=y))])
            case("foreign: washer 25% bigger", [("washer-0.375", 0, dict(scale=1.25))])
            case("foreign: screw 30% smaller", [("bolt-0.25", 0, dict(scale=0.7))])
            case("foreign: nut between the two sizes", [("nut-0.375", 0, dict(scale=0.89))])
            case("dropped onto the belt mid-view", [("nut-0.25", 0, dict(x_start_mm=45.0))])
            case("normal", [(NAMES[rng.integers(6)], 0, {})])
            case("rolling 15% slower than the belt", [("washer-0.25", 0, dict(slip=0.85))])
            t += 0.2 * (cfg.drop_distance_mm + 100) / speed   # it lands late: leave room before the next case
        L = Logged(cfg, mode="run", link=SimCollector(cfg)).run(sim)
        res = {}
        for p, r in match(sim, L.rows, cfg):
            d = res.setdefault(p["case"], dict(n=0, resort=0, wrong=0, ok=0))
            d["n"] += 1
            if r is None or r["final"] == RESORT:
                d["resort"] += 1
            elif r["final"] == p["label"]:
                d["ok"] += 1
            else:
                d["wrong"] += 1
        for k, d in res.items():
            if k == "normal":
                check("abuse: normal parts in between", d["ok"] == d["n"], f"{d['ok']}/{d['n']} sorted correctly")
            else:
                check(f"abuse: {k}", d["resort"] == d["n"], f"{d['resort']}/{d['n']} sent to re-sort, {d['wrong']} WRONG BIN")

        # 4 ---- lighting change
        sim = Sim(speed=speed, fps=fps, seed=14); cfg = make_cfg(sim)
        for i in range(24):
            sim.add(NAMES[i % 6], 0.2 + i * 1.0)
        L = Logged(cfg, mode="run", link=SimCollector(cfg)).run(sim, light=lambda t: 0.75 if 8 < t < 16 else 1.0)
        pairs = match(sim, L.rows, cfg)
        dim, _ = score([(p, r) for p, r in pairs if 8.5 < p["t0"] < 15])
        rest, _ = score([(p, r) for p, r in pairs if not 7 < p["t0"] < 17])
        check("lights dim 25% for 8 s", dim["resort"] == dim["n"] and rest["wrong"] == 0 and rest["correct"] > 0,
              f"during: {dim['resort']}/{dim['n']} re-sorted; before/after: {rest['correct']}/{rest['n']} sorted, "
              f"{rest['wrong']} WRONG BIN")
    finally:
        os.chdir(here)
    print(f"\n{'ALL CHECKS PASSED' if ok_all else 'SOME CHECKS FAILED'}  ({time.time() - t_start:.0f} s)")
    return ok_all


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
