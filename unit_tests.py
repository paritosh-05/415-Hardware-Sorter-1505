"""Quick logic checks. Run: python unit_tests.py"""
import numpy as np
from config import Config, RESORT
from scheduler import Scheduler
from collector import NullLink, SimCollector
from models import GeoModel, AppModel, Verdict, fuse
from segment import GEO

def slots(s): return {j.id: j.slot for j in s.queue + s.done}
cfg = Config(); A, B, C = "bolt-0.25", "nut-0.25", "washer-0.25"

# 1 well spaced -> all keep their slots
s = Scheduler(cfg, NullLink())
for i, (lab, t) in enumerate([(A, 5.0), (B, 5.7), (C, 6.4)]): s.add(i, lab, t, now=1.0)
assert slots(s) == {0: 1, 1: 2, 2: 3}, slots(s)
# 2 too close, different slots -> both re-sort; third far enough stays
s = Scheduler(cfg, NullLink())
for i, (lab, t) in enumerate([(A, 5.0), (B, 5.3), (C, 6.4)]): s.add(i, lab, t, now=1.0)
assert slots(s) == {0: 0, 1: 0, 2: 3}, slots(s)
# 3 same slot close together -> both sorted
s = Scheduler(cfg, NullLink())
for i, (lab, t) in enumerate([(A, 5.0), (A, 5.1)]): s.add(i, lab, t, now=1.0)
assert slots(s) == {0: 1, 1: 1}
# 4 ripple backwards: P(5.0) A(5.5) B(5.7): A,B conflict -> re-sort; then P->re-sort needs 600 but only 500 -> P re-sort too
s = Scheduler(cfg, NullLink())
for i, (lab, t) in enumerate([(C, 5.0), (A, 5.5), (B, 5.7)]): s.add(i, lab, t, now=1.0)
assert slots(s) == {0: 0, 1: 0, 2: 0}, slots(s)
# 5 ripple stops when the gap is big enough: P(4.8) A(5.5) B(5.7)
s = Scheduler(cfg, NullLink())
for i, (lab, t) in enumerate([(C, 4.8), (A, 5.5), (B, 5.7)]): s.add(i, lab, t, now=1.0)
assert slots(s) == {0: 3, 1: 0, 2: 0}, slots(s)
# 6 locked predecessor: P already committed (drop in 0.2 s) when A appears 0.3 s behind -> A re-sort + MISS recorded
s = Scheduler(cfg, NullLink()); s.add(0, C, 5.0, now=1.0); s.tick(4.8)
s.add(1, A, 5.3, now=4.8)
assert slots(s) == {0: 3, 1: 0} and s.counts["missed"] == 1 and s.queue[-1].missed, (slots(s), s.counts)
# 7 after P has dropped, a part right behind it for another slot -> re-sort + miss
s = Scheduler(cfg, NullLink()); s.add(0, C, 5.0, now=1.0); s.tick(5.05); s.add(1, A, 5.2, now=5.05)
assert slots(s)[1] == 0 and s.counts["missed"] == 1
# 8 a re-sort part (vision unsure) needs its gap too: A(5.0) then unknown at 5.3 -> A becomes re-sort as well
s = Scheduler(cfg, NullLink()); s.add(0, A, 5.0, 1.0); s.add(1, RESORT, 5.3, 1.0, "unknown")
assert slots(s) == {0: 0, 1: 0}
# 9 late insert between two parts (arrives out of order)
s = Scheduler(cfg, NullLink()); s.add(0, A, 5.0, 1.0); s.add(1, B, 6.0, 1.0); s.add(2, C, 5.5, 1.0)
assert slots(s) == {0: 0, 1: 0, 2: 0}, slots(s)
# 10 per-step timing: neighbours need less time than opposite slots
c2 = Config(); c2.step_ms, c2.settle_ms = 100.0, 100.0
assert c2.gap_needed_ms(1, 2) == 200 and c2.gap_needed_ms(1, 4) == 400 and c2.gap_needed_ms(1, 6) == 300 and c2.gap_needed_ms(0, 6) == 200
s = Scheduler(c2, NullLink()); s.add(0, A, 5.0, 1.0); s.add(1, B, 5.25, 1.0)   # slots 1,2: needs 200 ms -> fine
assert slots(s) == {0: 1, 1: 2}
# 11 same_slot_ok = False -> two close same-type parts are re-sorted
c3 = Config(); c3.same_slot_ok = False
s = Scheduler(c3, NullLink()); s.add(0, A, 5.0, 1.0); s.add(1, A, 5.1, 1.0)
assert slots(s) == {0: 0, 1: 0}
# 12 messages: every job is sent once, and re-sent when its slot changes
link = NullLink(); s = Scheduler(cfg, link); s.add(0, A, 5.0, 1.0); s.add(1, B, 5.3, 1.0)
assert [(i, sl) for i, sl, _ in link.sent] == [(0, 1), (0, 0), (1, 0)], link.sent
assert abs(link.sent[0][2] - 4000) < 1
# 13 landing windows: centres 0.7 s apart are fine as points, but not once each has a +-0.1 s window
s = Scheduler(cfg, NullLink()); s.add(0, A, 5.0, 1.0); s.add(1, B, 5.7, 1.0)
assert slots(s) == {0: 1, 1: 2}
s = Scheduler(cfg, NullLink()); s.add(0, A, 5.0, 1.0, half=0.1); s.add(1, B, 5.7, 1.0, half=0.1)
assert slots(s) == {0: 0, 1: 0}
link = NullLink(); s = Scheduler(cfg, link); s.add(0, A, 5.0, 1.0, half=0.1)
assert abs(link.sent[0][2] - 4100) < 1                     # collector is told to stay until the window ends
s.tick(5.05); assert not s.done; s.tick(5.11); assert len(s.done) == 1
# 14 simulated collector: where is it at a given time?
sc = SimCollector(cfg, move_ms=500); sc.set_time(0.0); sc.send(0, 1, 5000); sc.send(1, 2, 5700)
assert sc.slot_at(4.0) == 1 and sc.slot_at(5.0) == 1 and sc.slot_at(5.2) is None and sc.slot_at(5.5) == 2 and sc.slot_at(5.7) == 2
# 15 the real collector (collector.ino): 8 slices, every move 300 ms (1 to 4 slices), back to Rework (7) when idle
c5 = Config(); c5.n_slots, c5.resort_slot, c5.min_gap_ms = 8, 7, 300.0
assert c5.gap_needed_ms(1, 2) == 300 and c5.gap_needed_ms(7, 3) == 300 and c5.gap_needed_ms(4, 4) == 0
sc = SimCollector(c5, move_ms=300, idle_slot=7); sc.set_time(1.0); sc.send(0, 1, 3000)
assert sc.slot_at(1.2) is None and abs(sc.schedule()[0][2] - 1.3) < 1e-9 and sc.slot_at(1.4) == 1
assert sc.slot_at(4.0) == 1 and sc.slot_at(4.2) is None and sc.slot_at(4.35) == 7      # lands at 4.0, back by 4.3
sc.set_time(4.1); sc.send(1, 3, 2000)              # sent while returning: reaches 7 at 4.3, then 7 -> 3 by 4.6
assert abs(sc.schedule()[1][2] - 4.6) < 1e-9 and sc.landed() == {0: 1, 1: 3}
sc.set_time(7.0); sc.send(2, 5, 1000); sc.send(3, 2, 1300)   # lands 8.0 then 8.3 (+0.3 s), back to back: both caught
sc.set_time(7.1); sc.send(4, 6, 1400)              # lands 8.5, only 0.2 s after part 3: the pie is still turning
assert sc.landed()[2] == 5 and sc.landed()[3] == 2 and sc.landed()[4] is None
# ...which is why the scheduler sends such a pair to Rework: 300 ms apart is fine, 200 ms is not
s = Scheduler(c5, NullLink()); s.add(0, A, 5.0, 1.0); s.add(1, B, 5.3, 1.0)
assert slots(s) == {0: 1, 1: 2}, slots(s)
s = Scheduler(c5, NullLink()); s.add(0, A, 5.0, 1.0); s.add(1, B, 5.2, 1.0)
assert slots(s) == {0: 7, 1: 7}, slots(s)
print("scheduler: 15 checks passed")

# ---- fusion table
V = lambda lab, lev: Verdict(lab, lev)
t = [(("A", "confident"), ("A", "confident"), "A"), (("A", "certain"), ("A", "certain"), "A"),
     (("A", "certain"), ("B", "certain"), RESORT), (("A", "confident"), ("B", "confident"), RESORT),
     (("A", "certain"), ("A", "unsure"), RESORT), (("A", "certain"), ("B", "unsure"), RESORT),
     (("A", "certain"), (None, "unknown"), RESORT), ((None, "unknown"), ("A", "certain"), RESORT),
     (("A", "unsure"), ("A", "unsure"), RESORT), (("A", "confident"), ("A", "unsure"), RESORT)]
for g, a, want in t:
    got, why, cand = fuse(V(*g), V(*a), cfg); assert got == want, (g, a, got, why)
assert fuse(V("A", "certain"), V("A", "unsure"), cfg)[2] is True            # logged as candidate
assert fuse(V("A", "certain"), V("B", "unsure"), cfg)[2] is False
c4 = Config(); c4.allow_single_certain = True
assert fuse(V("A", "certain"), V("A", "unsure"), c4)[0] == "A"
assert fuse(V("A", "unsure"), V("A", "certain"), c4)[0] == "A"
assert fuse(V("A", "certain"), V("B", "unsure"), c4)[0] == RESORT           # pointing elsewhere: never
assert fuse(V("A", "certain"), (V(None, "unknown")), c4)[0] == RESORT
assert fuse(V("A", "confident"), V("A", "unsure"), c4)[0] == RESORT
print("fusion: 15 checks passed")

# ---- geometric model: two poses under one label become two modes; odd examples are ignored
rng = np.random.default_rng(0)
flat = [dict(L=13 + rng.normal(0, .1), W=11.6 + rng.normal(0, .1), A=114 + rng.normal(0, 2), Dh=0, S=.97, Wm=9.4) for _ in range(20)]
side = [dict(L=12.8 + rng.normal(0, .1), W=5.6 + rng.normal(0, .1), A=70 + rng.normal(0, 2), Dh=0, S=.97, Wm=5.5) for _ in range(8)]
odd = [dict(L=30, W=20, A=400, Dh=0, S=.6, Wm=9)]
other = [dict(L=16.7 + rng.normal(0, .1), W=15.2 + rng.normal(0, .1), A=193 + rng.normal(0, 3), Dh=0, S=.98, Wm=13.1) for _ in range(20)]
gm = GeoModel(cfg); rep = gm.fit({"nut-0.25": flat + side + odd, "nut-0.375": other})
assert sum(m["label"] == "nut-0.25" for m in gm.modes) == 2 and "ignored 1 odd" in rep, rep
assert gm.classify(side[0]).label == "nut-0.25" and gm.classify(flat[0]).level == "certain"
assert gm.classify(odd[0]).level == "unknown" and gm.classify(odd[0]).label is None
mid = dict(L=14.9, W=13.4, A=150, Dh=0, S=.97, Wm=11.2)                     # halfway between the two nuts
assert gm.classify(mid).level == "unknown"
near = dict(flat[0]); near["L"] += 1.2                                      # 4 sigma off in one measurement
assert gm.classify(near).level in ("unsure",), gm.classify(near)
print("geometric model: modes, outliers, gates OK")
