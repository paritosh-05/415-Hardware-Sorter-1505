"""Decide which slot each part gets, given how close together the parts are.

Rule (one line): two parts in a row that need DIFFERENT slots must reach the drop at least
`gap_needed` apart. Re-sort is a slot like any other.

Every part has a landing WINDOW, not a single instant: t_drop +- half. A long part (or a clump of
touching parts) takes a while to leave the belt, so its window is wider. The gap is measured from
the end of one window to the start of the next, and the collector is told to stay until the end. If they are closer, both go to re-sort,
and the check is repeated because that can create a new conflict with the part in front
(the decision ripples backwards through the queue).

A part can no longer be changed once it is "locked": the collector would not have time to react.
If a conflict needs a locked part to change, it is recorded as a MISS (the collector may still be
turning when the next part lands) - that is the one case the camera cannot save.
"""
from dataclasses import dataclass


@dataclass
class Job:
    id: int
    label: str            # what the vision decided ("resort" or a part name)
    slot: int             # slot currently planned
    t_drop: float         # predicted time the part lands (middle of its window)
    half: float = 0.0     # half-width of the landing window (s)
    reason: str = ""      # why it is going to re-sort, if it is
    wanted: int = None    # slot the vision asked for
    locked: bool = False  # the collector has been told and has no time left to change
    sent: bool = False
    dropped: bool = False
    missed: bool = False

    @property
    def t_start(self):
        return self.t_drop - self.half

    @property
    def t_end(self):
        return self.t_drop + self.half


class Scheduler:
    def __init__(self, cfg, link=None):
        self.cfg, self.link = cfg, link
        self.queue = []                 # jobs not yet dropped, in drop order
        self.last = None                # the most recent dropped job
        self.done = []                  # everything that has dropped (for the log)
        self.counts = dict(sorted=0, resort=0, gap_resort=0, missed=0)

    def add(self, part_id, label, t_drop, now, reason="", half=0.0):
        slot = self.cfg.slot_of(label)
        job = Job(part_id, label, slot, t_drop, half, reason, wanted=slot)
        self.queue.append(job)
        self.queue.sort(key=lambda j: j.t_drop)
        self._plan(now)
        return job

    def tick(self, now):
        """Call every frame. Returns jobs that have just dropped."""
        out = []
        self._lock(now)
        while self.queue and now >= self.queue[0].t_end:
            j = self.queue.pop(0)
            j.dropped = True
            self.last = j
            self.done.append(j)
            rs = self.cfg.resort_slot
            self.counts["resort" if j.slot == rs else "sorted"] += 1
            if j.slot == rs and j.wanted != rs:
                self.counts["gap_resort"] += 1
            out.append(j)
        return out

    # ------------------------------------------------------------------------------------------
    def _conflict(self, a, b):
        if a.slot == b.slot:
            return (not self.cfg.same_slot_ok) and a.slot != self.cfg.resort_slot and \
                (b.t_start - a.t_end) * 1000.0 < self.cfg.min_gap_ms - 1e-6
        return (b.t_start - a.t_end) * 1000.0 < self.cfg.gap_needed_ms(a.slot, b.slot) - 1e-6   # tolerance: float rounding

    def _plan(self, now):
        rs = self.cfg.resort_slot
        self._lock(now)
        changed = set()
        for _ in range(4 * len(self.queue) + 4):
            chain = ([self.last] if self.last else []) + self.queue
            again = False
            for a, b in zip(chain, chain[1:]):
                if not self._conflict(a, b):
                    continue
                for j in (a, b):
                    if j.slot != rs and not j.locked and not j.dropped:
                        j.slot = rs
                        j.reason = j.reason or "too close to another part"
                        changed.add(j.id)
                        again = True
                if self._conflict(a, b) and not b.missed:     # could not be fixed: a is committed
                    b.missed = True
                    self.counts["missed"] += 1
                    changed.add(b.id)
            if not again:
                break
        if self.link:
            for j in self.queue:
                if j.id in changed or not j.sent:
                    self.link.send(j.id, j.slot, max(0.0, (j.t_end - now) * 1000.0))
        for j in self.queue:
            j.sent = True

    def _lock(self, now):
        for j in self.queue:
            if j.sent and not j.locked and now >= j.t_start - self.cfg.min_gap_ms / 1000.0:
                j.locked = True
