"""Follow each part across the frame.

A track collects, for one part: where it was at each time stamp (-> its speed and when it will
reach the decision line), its measurements in the measuring zone (middle of the frame, where the
view is least distorted), and a handful of appearance descriptors spread over the zone.

A track is "finished" when the part crosses the decision line (or disappears). Finished tracks are
handed to the pipeline, which classifies them and schedules the collector.
"""
from dataclasses import dataclass, field

import numpy as np

N_DESC = 8          # appearance views kept per part, spread evenly over the measuring zone


@dataclass
class Track:
    id: int
    t: list = field(default_factory=list)
    x: list = field(default_factory=list)        # mm along the belt
    full: list = field(default_factory=list)     # was the whole part in view on that frame?
    y: list = field(default_factory=list)
    geo: list = field(default_factory=list)      # measurements from clean frames inside the zone
    descs: list = field(default_factory=list)
    flags: set = field(default_factory=set)      # touching, appeared, lost, side, oversize, slipping
    ext_mm: float = 0.0                          # longest extent along the belt seen so far
    blob: object = None                          # latest blob (for drawing)
    best_blob: object = None                     # clean blob nearest the middle (for the saved crop)
    missed: int = 0
    last_seen: float = 0.0
    next_desc_x: float = 0.0
    decided: bool = False                        # has crossed the decision line
    done: bool = False                           # has left the view
    # filled in when finished
    speed: float = None                          # mm/s
    speed_alt: float = None                      # second guess when the part is not moving with the belt
    t_handoff: float = None
    verdicts: tuple = None
    decision: str = None
    reason: str = ""

    def landing(self, speed, cfg):
        """Predicted landing time if the part travels at `speed` from its last clear sighting."""
        t_i, x_i, x_hand = self._ref
        return t_i + (x_hand - x_i + cfg.drop_distance_mm) / max(speed, 1.0) + cfg.fall_ms / 1000.0

    def predict(self, t, belt_speed):
        v = self.fit_speed() or belt_speed
        return self.x[-1] + v * (t - self.t[-1])

    def recent_speed(self, window=0.3):
        """Speed over the last `window` seconds of whole-part frames: what the belt is doing right now."""
        k = np.array(self.full, bool)
        if k.sum() < 6:
            return None
        t, x = np.array(self.t)[k], np.array(self.x)[k]
        m = t >= t[-1] - window
        t, x = t[m], x[m]
        if len(t) < 6 or t[-1] - t[0] < 0.1:
            return None
        v = np.polyfit(t - t[0], x, 1)[0]
        return float(v) if v > 1.0 else None

    def fit_speed(self):
        """Speed from frames where the whole part was visible (a part that is still entering has a
        centroid that moves at half speed)."""
        k = np.array(self.full, bool)
        if k.sum() < 4:
            return None
        t, x = np.array(self.t)[k], np.array(self.x)[k]
        if t[-1] - t[0] < 0.03:
            return None
        v = np.polyfit(t - t[0], x, 1)[0]
        return float(v) if v > 1.0 else None


class Tracker:
    def __init__(self, cfg):
        self.cfg = cfg
        self.tracks = []
        self.next_id = 1
        self.belt_speed = cfg.init_speed_mm_s
        self._speeds = []
        self.t_prev = None
        self.counts = dict(parts=0, lost=0)
        self.forward = self.backward = 0      # frame-to-frame moves seen with / against the belt direction
        self.wrong_way = False                # parts are moving against cfg.flow_left_to_right
        self.dt = None                        # typical time between frames (s)
        self.seam = None                      # the belt seam while it is in view: dict(track, lo, hi, t, peak)
        self._seam_areas = []                 # how big the bare seam looked on its last few passes (mm2)

    def update(self, blobs, t, lane_len, describe):
        """blobs: this frame's blobs. describe(blob) -> appearance descriptor.
        Returns the tracks that finished on this frame."""
        cfg = self.cfg
        z0, z1 = cfg.zone[0] * lane_len, cfg.zone[1] * lane_len
        x_hand = cfg.handoff_frac * lane_len
        dt = 0.0 if self.t_prev is None else t - self.t_prev
        self.t_prev = t
        if dt > 0:
            self.dt = dt if self.dt is None else 0.9 * self.dt + 0.1 * dt
        known = len(self._speeds) > 0
        seam_done = self._seam(blobs, t, dt, lane_len, x_hand)
        blobs = [b for b in blobs if not getattr(b, "is_seam", False)]

        # ---- match blobs to tracks: closest to where the track should be by now
        pairs = []
        for ti, tr in enumerate(self.tracks):
            xp = tr.predict(t, self.belt_speed)
            sure = tr.fit_speed() is not None or known
            for bi, b in enumerate(blobs):
                if abs(b.y_mm - tr.y[-1]) > 8.0:
                    continue
                step = b.x_mm - tr.x[-1]
                if sure:
                    ok = abs(b.x_mm - xp) <= cfg.match_gate_mm + 0.25 * abs(xp - tr.x[-1])
                else:   # belt speed not known yet: anything between "didn't move" and "max speed"
                    ok = -2.0 <= step <= cfg.max_speed_mm_s * max(dt, 1e-3) * (tr.missed + 1) + 2.0
                if ok:
                    pairs.append((abs(b.x_mm - xp) + 0.5 * abs(b.y_mm - tr.y[-1]), ti, bi))
        used_t, used_b = set(), set()
        for _, ti, bi in sorted(pairs):
            if ti in used_t or bi in used_b:
                continue
            used_t.add(ti); used_b.add(bi)
            step = blobs[bi].x_mm - self.tracks[ti].x[-1]
            if step > 1.0:
                self.forward += 1
            elif step < -1.0:
                self.backward += 1
            self._observe(self.tracks[ti], blobs[bi], t, z0, z1, describe)

        # ---- blobs nobody claimed: new parts
        for bi, b in enumerate(blobs):
            if bi in used_b:
                continue
            for ti, old in enumerate(self.tracks):        # did an unmatched track just move BACKWARDS to here?
                if ti in used_t or abs(b.y_mm - old.y[-1]) > 8.0:
                    continue
                back = old.x[-1] - b.x_mm
                if 1.0 < back <= cfg.max_speed_mm_s * max(dt, 1e-3) * (old.missed + 1) + 2.0:
                    self.backward += 1
                    break
            tr = Track(self.next_id, next_desc_x=z0)
            self.next_id += 1
            if self._started and not b.touch_up and b.x_mm > 0.30 * lane_len:
                tr.flags.add("appeared")          # showed up mid-belt: dropped in, or split off a clump
            self._observe(tr, b, t, z0, z1, describe)
            self.tracks.append(tr)
        self._started = True
        self.wrong_way = self.backward >= 6 and self.backward > 3 * self.forward

        # ---- tracks nobody matched
        finished = []
        for ti, tr in enumerate(self.tracks):
            if ti in used_t or tr.last_seen == t:
                continue
            tr.missed += 1
            xp = tr.predict(t, self.belt_speed)
            for other in self.tracks:             # swallowed by a neighbour -> the two are touching
                if other is tr or other.last_seen != t or other.blob is None:
                    continue
                bx, by, bw, bh = other.blob.bbox
                ppm = cfg.px_per_mm
                x_lo, x_hi = sorted((other.blob.x_mm - bw / ppm, other.blob.x_mm + bw / ppm))
                if x_lo <= xp <= x_hi and abs(other.y[-1] - tr.y[-1]) < bh / ppm:
                    other.flags.add("touching"); tr.flags.add("touching")
            gone = xp > lane_len + 2.0
            if gone or t - tr.last_seen > cfg.lost_after_s:
                if not gone and not tr.decided:
                    tr.flags.add("lost")
                tr.done = True

        if cfg.follow_belt:                       # unsteady belt: its speed is whatever the parts in view do now
            vs = self._live_speeds(t)
            if vs:
                self.belt_speed = float(np.median(vs))
        if seam_done is not None:
            finished.append(seam_done)
        for tr in list(self.tracks):
            crossing = not tr.decided and tr.x[-1] >= x_hand
            if crossing or (tr.done and not tr.decided):
                tr.decided = True                 # decide once: at the line, or when it vanished early
                if self._finish(tr, x_hand):
                    finished.append(tr)
            if tr.done:
                self.tracks.remove(tr)
        return finished

    _started = False

    def _live_speeds(self, t, skip=None):
        """Current speeds of the parts in view on this frame (not `skip`, not touching ones)."""
        # Something that is not riding the belt must not set the belt speed: a part dragging on the rail,
        # a hand, a part dropped in, or anything standing still. So: no odd tracks, never below 15 mm/s,
        # and within half to double the speed of the last few parts.
        usual = float(np.median(self._speeds)) if len(self._speeds) >= 3 else None
        out = []
        for tr in self.tracks:
            if tr is skip or tr.last_seen != t or tr.flags & {"touching", "oversize", "side", "appeared"}:
                continue
            v = tr.recent_speed()
            if v is None or v < 15.0 or (usual and not 0.5 * usual <= v <= 2.0 * usual):
                continue
            out.append(v)
        return out

    # ------------------------------------------------------------------------------- belt seam
    def _seam(self, blobs, t, dt, lane_len, x_hand):
        """The belt seam reaches from one belt edge to the other, which no single part can. It is followed
        separately from the parts: alone it is ignored; if it looks bigger than it usually does, a part is
        sitting on it and that stretch of belt is sent to re-sort. Returns a track to re-sort, or None."""
        cfg, ppm, sm = self.cfg, self.cfg.px_per_mm, self.seam

        def span_mm(b):
            half = 0.5 * b.bbox[2] / ppm
            return b.x_mm - half, b.x_mm + half

        mine = [b for b in blobs if b.span]
        if sm is not None:                       # pieces of it still entering / leaving: they touch one edge
            step = self.belt_speed * dt
            sm["lo"] += step; sm["hi"] += step
            for b in blobs:
                lo, hi = span_mm(b)
                if not b.span and b.touch_side and lo >= sm["lo"] - 6.0 and hi <= sm["hi"] + 6.0:
                    mine.append(b)
        for b in mine:
            b.is_seam = True
        if not mine:
            if sm is not None and (sm["lo"] > lane_len or t - sm["t"] > cfg.lost_after_s):
                self.seam = None
                return self._seam_verdict(sm, x_hand)
            return None
        if sm is None:                           # a new pass. Drop the stubs that were tracked while it entered
            lo, hi = min(span_mm(b)[0] for b in mine), max(span_mm(b)[1] for b in mine)
            for tr in list(self.tracks):
                if not tr.geo and "side" in tr.flags and lo - 6.0 <= tr.x[-1] <= hi + 6.0:
                    self.tracks.remove(tr)
            sm = self.seam = dict(track=Track(self.next_id, flags={"seam"}), lo=lo, hi=hi, t=t, peak=0.0, out=None)
            self.next_id += 1
        big = max(mine, key=lambda b: b.area_mm2)
        tr = sm["track"]
        sm["lo"], sm["hi"] = min(span_mm(b)[0] for b in mine), max(span_mm(b)[1] for b in mine)
        sm["t"] = t
        sm["peak"] = max(sm["peak"], sum(b.area_mm2 for b in mine))
        tr.t.append(t); tr.x.append(big.x_mm); tr.y.append(big.y_mm); tr.full.append(bool(big.span))
        tr.ext_mm = max(tr.ext_mm, sm["hi"] - sm["lo"])
        tr.blob, tr.last_seen = big, t
        if not tr.decided and big.x_mm >= x_hand:
            return self._seam_verdict(sm, x_hand)
        return None

    def _seam_verdict(self, sm, x_hand):
        """Called once per pass, at the decision line (or when it vanished early)."""
        tr = sm["track"]
        if tr.decided:
            return None
        tr.decided = True
        cfg = self.cfg
        ref = float(np.median(self._seam_areas)) if len(self._seam_areas) >= 2 else None
        loaded = not cfg.seam_ignore or ref is None or sm["peak"] > ref * (1.0 + cfg.seam_extra_frac) + cfg.seam_extra_mm2
        if ref is None or not loaded:
            self._seam_areas = (self._seam_areas + [sm["peak"]])[-7:]
        self.counts["seam"] = self.counts.get("seam", 0) + 1
        if not loaded or len(tr.t) < 3:
            return None                          # bare seam: nothing to do
        return tr if self._finish(tr, x_hand) else None

    def _observe(self, tr, b, t, z0, z1, describe):
        tr.t.append(t); tr.x.append(b.x_mm); tr.y.append(b.y_mm)
        tr.full.append(not (b.touch_up or b.touch_down))
        tr.ext_mm = max(tr.ext_mm, b.bbox[2] / self.cfg.px_per_mm)
        tr.blob, tr.last_seen, tr.missed = b, t, 0
        if b.oversize:
            tr.flags.add("oversize")
        if b.touch_side:
            tr.flags.add("side")
        if b.clean and z0 <= b.x_mm <= z1 and not tr.decided:
            tr.geo.append(b.geo)
            mid = 0.5 * (z0 + z1)
            if tr.best_blob is None or abs(b.x_mm - mid) < abs(tr.best_blob.x_mm - mid):
                tr.best_blob = b
            if b.x_mm >= tr.next_desc_x and len(tr.descs) < N_DESC:
                tr.descs.append(describe(b))
                tr.next_desc_x = b.x_mm + (z1 - z0) / N_DESC

    def _finish(self, tr, x_hand):
        """Work out speed and decision-line time. Returns False for things that are not parts
        (e.g. a mark at the belt edge that never moved)."""
        v = tr.fit_speed()
        moved = tr.x[-1] - tr.x[0]
        if v is None and moved < 3.0 and len(tr.t) > 3:
            return False                                   # stationary -> not on the belt
        if len(tr.t) <= 2 and not tr.geo:
            return False                                   # a flicker
        ref = self.belt_speed if len(self._speeds) >= 4 else None
        follow = self.cfg.follow_belt
        if follow:
            # the belt speed varies, so the last few parts say nothing about this one. Use this part's own
            # current speed, and call it slipping only against other parts that are in view at the same time.
            v = tr.recent_speed() or v
            others = self._live_speeds(tr.last_seen, skip=tr)
            ref = float(np.median(others)) if others else None
        if v is not None and len(tr.t) >= 6 and not tr.flags & {"touching", "seam"}:
            self._speeds = (self._speeds + [v])[-9:]
            if not follow:
                self.belt_speed = float(np.median(self._speeds))
        if follow and not (v is not None and ref and abs(v / ref - 1.0) > self.cfg.slip_tol):
            tr.speed = v or self.belt_speed
        elif v is not None and ref and abs(v / ref - 1.0) > self.cfg.slip_tol:
            tr.flags.add("slipping")               # rolling or sliding: its arrival can't be trusted
            tr.speed, tr.speed_alt = v, ref        # keep both guesses; the landing window covers both
        else:                                      # the belt's speed (many parts) beats one part's fit
            tr.speed = ref or v or self.belt_speed
        # time at the decision line, from the last frame where the WHOLE part was visible (a long part
        # is already sliding out of view at the line, which drags its visible centroid backwards)
        full = [i for i, f in enumerate(tr.full) if f]
        i = full[-1] if full else len(tr.t) - 1
        tr.t_handoff = tr.t[i] + (x_hand - tr.x[i]) / tr.speed
        tr._ref = (tr.t[i], tr.x[i], x_hand)
        self.counts["parts"] += 1
        if "lost" in tr.flags:
            self.counts["lost"] += 1
        return True
