"""Unsteady belt on the simulator: does each part REALLY land in its slot (or Rework)?"""
import os, sys, tempfile, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collector import SimCollector
from config import RESORT
from pipeline import Pipeline, build_models
from selftest import NAMES, make_cfg, Logged, match, lane_len
from sim import Sim

os.chdir(tempfile.mkdtemp())
rng = np.random.default_rng(3)
for name in NAMES:
    sim = Sim(speed=85.0, fps=120, seed=sum(map(ord, name))); cfg = make_cfg(sim)
    for i in range(14):
        sim.add(name, 0.2 + i * 1.1)
    p = Pipeline(cfg, mode="teach", label=name, quiet=True)
    for fr, t in sim.frames():
        p.process(fr, t)
    p.finish()
build_models(cfg, quiet=True)

def profile(kind):
    ts = np.arange(0, 400, 0.01)
    if kind == "steady":
        v = np.full_like(ts, 85.0)
    elif kind == "slow drift":      # 60..90 mm/s, 25 s period
        v = 75 + 15 * np.sin(2 * np.pi * ts / 25.0)
    elif kind == "fast surge":      # 60..90 mm/s, 6 s period
        v = 75 + 15 * np.sin(2 * np.pi * ts / 6.0)
    elif kind == "steps":           # jumps between 85 and 62 every 7 s
        v = np.where((ts // 7) % 2 == 0, 85.0, 62.0)
    S = np.concatenate([[0], np.cumsum(v[:-1] * 0.01)])
    return lambda t: float(np.interp(t, ts, S))

def run(kind, follow, gap, n=45, tol=0.10):
    sim = Sim(speed=85.0, fps=120, seed=21); cfg = make_cfg(sim)
    sim.travel = profile(kind)
    cfg.n_slots, cfg.resort_slot, cfg.min_gap_ms, cfg.drop_distance_mm = 8, 7, 400.0, 366.0
    cfg.follow_belt, cfg.speed_tol = follow, tol
    r2 = np.random.default_rng(5); t = 0.3
    for _ in range(n):
        sim.add(NAMES[r2.integers(6)], t); t += r2.uniform(*gap)
    L = Logged(cfg, mode="run", link=SimCollector(cfg, move_ms=300, idle_slot=7)).run(sim)
    x_drop = cfg.handoff_frac * lane_len(cfg, sim) + cfg.drop_distance_mm
    res = dict(ok=0, rework=0, turning=0, WRONG=0, lost=0); errs = []; why = {}
    for p, r in match(sim, L.rows, cfg):
        if r is None:
            res["lost"] += 1; continue
        t_real = sim.t_cross(p, x_drop) + cfg.fall_ms / 1000
        errs.append(1000 * (r["t_drop"] - t_real))
        at = L.pipe.link.slot_at(t_real)
        if at is None: res["turning"] += 1
        elif at == cfg.slots[p["label"]]: res["ok"] += 1
        elif at == cfg.resort_slot:
            res["rework"] += 1
            k = (r["final_reason"] or "landed outside its window").split(" (")[0]; why[k] = why.get(k, 0) + 1
        else: res["WRONG"] += 1
    e = np.abs(errs)
    print(f"{kind:11s} follow={str(follow):5s} gap {gap}: {res}  timing error median {np.median(e):.0f} ms, worst {e.max():.0f} ms  {why}")

for kind in ["steady", "slow drift", "fast surge", "steps"]:
    for follow in (False, True):
        run(kind, follow, (2.0, 4.0))
for kind in ["slow drift", "steps"]:
    run(kind, True, (0.8, 1.6))
