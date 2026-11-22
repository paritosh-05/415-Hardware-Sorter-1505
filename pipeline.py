"""Glue: frame in -> parts tracked, classified, scheduled, logged."""
import csv
import os
import time

import cv2
import numpy as np

from collector import NullLink
from config import RESORT
from models import AppModel, GeoModel, Verdict, fuse, save_sample
from scheduler import Scheduler
from segment import GEO, Segmenter
from tracker import Tracker

MODEL_DIR, TEACH_DIR, BG_FILE = "model", "teach", "background.png"


class Health:
    """Cheap checks on every frame. While something is wrong, every part goes to re-sort."""

    def __init__(self, cfg):
        self.cfg, self.t_prev, self.problem, self.bad_until = cfg, None, "", -1.0
        self.belt_level = 1.0
        self.external = ""          # set by the collector link, e.g. "collector: E,HOME"

    def update(self, seg, t):
        cfg, why = self.cfg, ""
        if self.t_prev is not None and t - self.t_prev > cfg.stale_s:
            why = "camera stalled"
        self.t_prev = t
        m = seg.mask[::4, ::4]
        if (m > 0).mean() > cfg.max_cover:
            why = "view blocked"
        elif seg.bg_mode == "background":
            free = (m == 0) & seg.valid[::4, ::4]
            if free.sum() > 50:
                self.belt_level = float(np.median(seg.ratio[::4, ::4][free]))
                if abs(self.belt_level - 1.0) > cfg.belt_tol:
                    why = f"lighting changed ({self.belt_level:.2f}x)"
        why = self.external or why
        if why:
            self.problem, self.bad_until = why, t + 0.5
        elif t > self.bad_until:
            self.problem = ""
        return self.problem


class Pipeline:
    def __init__(self, cfg, mode="run", label=None, link=None, out_dir=None, truth=None, quiet=False):
        """mode: 'run' (classify + schedule), 'teach' (save examples of `label`), 'view' (setup)."""
        self.cfg, self.mode, self.label, self.truth, self.quiet = cfg, mode, label, truth, quiet
        self.seg = Segmenter(cfg)
        self.tracker = Tracker(cfg)
        self.health = Health(cfg)
        self.geo, self.app = GeoModel(cfg), AppModel(cfg)
        self.link = link or NullLink()
        self.sched = Scheduler(cfg, self.link)
        self.records = {}            # part id -> dict (kept until the part has dropped)
        self.recent = []             # last few decisions, for the display
        self.stats = dict(parts=0, sorted=0, resort=0, wrong=0, correct=0, override=0, reasons={})
        self.taught = 0
        self.notes = {}              # advice shown on screen and printed once
        self.log = []                # every finished part's record (used by `check`)
        self.feed, self.fed = None, 0   # `check`: the feeding order, and how many parts have been seen
        self.blobs = []
        self.out_dir = out_dir
        self._csv = None
        self._crops = {}
        if os.path.exists(BG_FILE):
            self.seg.set_background(cv2.imread(BG_FILE, cv2.IMREAD_GRAYSCALE))
        if mode == "run":
            self.geo.load(os.path.join(MODEL_DIR, "geo.json"))
            self.app.load(os.path.join(MODEL_DIR, "app.npz"))
        if mode == "teach":
            d = os.path.join(TEACH_DIR, label)
            self.taught = len([f for f in os.listdir(d) if f.endswith(".npz")]) if os.path.isdir(d) else 0
        if out_dir:
            os.makedirs(os.path.join(out_dir, "crops"), exist_ok=True)

    # ------------------------------------------------------------------------------------ frame
    def process(self, frame, t, now=None):
        now = t if now is None else now
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        self.blobs = self.seg.process(gray)
        problem = self.health.update(self.seg, t)
        lane_len = self.cfg.lane_len_mm(gray.shape)
        finished = self.tracker.update(self.blobs, t, lane_len, self.seg.describe)
        self._advise(lane_len)
        for tr in self.tracker.tracks:                       # keep a picture of each part mid-zone
            if tr.best_blob is tr.blob and tr.last_seen == t and tr.blob is not None:
                x, y, w, h = tr.blob.bbox
                p = int(2 * self.cfg.px_per_mm)
                tr.crop = gray[max(0, y - p):y + h + p, max(0, x - p):x + w + p].copy()
        if self.mode != "view":
            for tr in finished:
                self._decide(tr, problem, now)
        if hasattr(self.link, "set_time"):
            self.link.set_time(now)
        for job in self.sched.tick(now):
            self._dropped(job)
        return finished

    def _advise(self, lane_len):
        """Spot setup problems that would otherwise show up as every part being rejected."""
        tk, cfg, notes = self.tracker, self.cfg, {}
        if tk.wrong_way:
            notes["way"] = ("Parts are moving AGAINST the configured belt direction. "
                            "Run setup, press f, then w.")
        if tk.dt and len(tk._speeds) >= 3:
            n = (cfg.zone[1] - cfg.zone[0]) * lane_len / max(tk.belt_speed * tk.dt, 1e-6)
            if n < 2 * cfg.min_frames:
                notes["fps"] = (f"Only about {n:.0f} frames per part in the measuring zone "
                                f"({1 / tk.dt:.0f} frames/s at {tk.belt_speed:.0f} mm/s). Need a faster camera "
                                f"mode or a slower belt.")
        for k, text in notes.items():
            if k not in self.notes and not self.quiet:
                print("[advice] " + text)
        self.notes = {**self.notes, **notes} if notes else {k: v for k, v in self.notes.items() if k == "way" and tk.wrong_way}

    # --------------------------------------------------------------------------------- decision
    def _decide(self, tr, problem, now):
        cfg = self.cfg
        med = dict(zip(GEO, np.median([[g[k] for k in GEO] for g in tr.geo], axis=0))) if tr.geo else {}
        gv, av, cand = Verdict(), Verdict(), False
        pre = ""
        if problem:
            pre = "health: " + problem
        elif tr.flags:
            pre = {"touching": "touching another part", "appeared": "appeared mid-belt", "lost": "lost from view",
                   "side": "hanging off the belt edge", "seam": "on the belt seam", "oversize": "too big (pile of parts)",
                   "slipping": "moving differently from the belt"}[sorted(tr.flags)[0]]
        elif len(tr.geo) < cfg.min_frames or not tr.descs:
            pre = "not seen clearly"

        if self.mode == "teach":
            if pre:
                tr.decision, tr.reason = "skipped", pre
            else:
                save_sample(TEACH_DIR, self.label, self.taught, tr.geo, tr.descs, getattr(tr, "crop", None))
                self.taught += 1
                tr.decision, tr.reason = "taught", ""
            self._note(tr)
            if not self.quiet:
                print(f"[teach] part {tr.id}: {tr.decision} {tr.reason}  " +
                      "  ".join(f"{k}={v:.2f}" for k, v in med.items()))
            return

        if pre:
            label, reason = RESORT, pre
        else:
            gv, av = self.geo.classify(med), self.app.classify(tr.descs)
            label, reason, cand = fuse(gv, av, cfg)
        tr.verdicts, tr.decision, tr.reason = (gv, av), label, reason
        t_drop = tr.landing(tr.speed, cfg)
        k = 1.0 if tr.flags else cfg.extent_factor            # clumps: parts land one after another
        half = k * 0.5 * tr.ext_mm / max(min(tr.speed, tr.speed_alt or tr.speed), 1.0)
        if tr.speed_alt:                                      # two speed guesses: cover both landings
            t_alt = tr.landing(tr.speed_alt, cfg)
            t_drop, half = 0.5 * (t_drop + t_alt), half + 0.5 * abs(t_drop - t_alt)
        self.sched.add(tr.id, label, t_drop, now, "" if label != RESORT else reason, half)
        self.records[tr.id] = dict(
            id=tr.id, t_decide=round(tr.t_handoff, 4), t_drop=round(t_drop, 4), window_ms=round(2000 * half),
            speed_mm_s=round(tr.speed, 1),
            frames=len(tr.geo), views=len(tr.descs),
            **{k: round(float(med[k]), 3) if k in med else "" for k in GEO},     # blank when never measured
            geo_label=gv.info.get("nearest"), geo_level=gv.level, geo_z=round(gv.score, 2),
            geo_rival=gv.rival, geo_rival_z=round(gv.rival_score, 2),
            app_label=av.info.get("nearest"), app_level=av.level, app_dist=round(av.score, 3),
            app_rival=av.rival, app_share=av.info.get("share"), app_ratio=av.info.get("ratio"),
            vision=label, vision_reason=reason, override_candidate=int(cand), truth=self.truth or "")
        self._crops[tr.id] = getattr(tr, "crop", None)
        if self.feed and tr.geo:                 # a real part went by (the seam / a hand has no measurements)
            self.fed += 1
            print(self.next_prompt())
        self._note(tr)

    def next_prompt(self):
        if self.fed >= len(self.feed):
            return f"\n>>> ALL {len(self.feed)} PARTS SEEN. Wait for the last one to drop, then press q.\n"
        return f"\n>>> NEXT: #{self.fed + 1} of {len(self.feed)}:  {self.feed[self.fed].upper()}\n"

    def _note(self, tr):
        self.recent = (self.recent + [(tr.id, tr.decision, tr.reason)])[-8:]

    def _dropped(self, job):
        """The part has landed (by the clock): its slot is final now, so write the log line."""
        rec = self.records.pop(job.id, None)
        if rec is None:
            return
        cfg, st = self.cfg, self.stats
        final = RESORT if job.slot == cfg.resort_slot else rec["vision"]
        reason = job.reason if final == RESORT else ""
        rec.update(final=final, slot=job.slot, final_reason=reason, missed=int(job.missed))
        st["parts"] += 1
        if final == RESORT:
            st["resort"] += 1
            st["reasons"][reason.split(" (")[0]] = st["reasons"].get(reason.split(" (")[0], 0) + 1
        else:
            st["sorted"] += 1
            if self.truth:
                st["correct" if final == self.truth else "wrong"] += 1
        st["override"] += rec["override_candidate"]
        self.log.append(rec)
        if self.out_dir:
            if self._csv is None:
                fh = open(os.path.join(self.out_dir, "parts.csv"), "w", newline="")
                self._csv = (fh, csv.DictWriter(fh, fieldnames=list(rec)))
                self._csv[1].writeheader()
            self._csv[1].writerow(rec)
            self._csv[0].flush()
            crop = self._crops.pop(job.id, None)
            if crop is not None:
                cv2.imwrite(os.path.join(self.out_dir, "crops", f"{job.id:05d}_{final}.png"), crop)
        if not self.quiet:
            print(f"[part {job.id}] -> slot {job.slot} ({final})" + (f"  [{reason}]" if reason else "") +
                  ("  MISSED: collector may not have been in place" if job.missed else ""))

    def finish(self):
        """Flush parts that were decided but had not dropped when the run ended."""
        for job in list(self.sched.queue):
            self._dropped(job)
        self.sched.queue.clear()
        if self._csv:
            self._csv[0].close()
        self.link.close()

    def summary(self):
        st, sc = self.stats, self.sched.counts
        n = max(st["parts"], 1)
        lines = [f"parts {st['parts']}:  sorted {st['sorted']} ({100 * st['sorted'] / n:.0f}%)  "
                 f"re-sort {st['resort']} ({100 * st['resort'] / n:.0f}%)  missed {sc['missed']}  "
                 f"belt speed {self.tracker.belt_speed:.0f} mm/s"]
        lead = self.cfg.drop_distance_mm / max(self.tracker.belt_speed, 1.0)
        if lead * 1000.0 < 3 * self.cfg.min_gap_ms:
            lines.append(f"NOTE: only {lead:.1f} s between the decision line and the drop at this belt speed. "
                         f"Below about {3 * self.cfg.min_gap_ms / 1000:.1f} s, close chains of parts can be 'missed'.")
        if self.truth:
            lines.append(f"truth = {self.truth}:  correct bin {st['correct']}  WRONG BIN {st['wrong']}")
        for r, c in sorted(st["reasons"].items(), key=lambda kv: -kv[1]):
            lines.append(f"   re-sort because {r}: {c}")
        if st["override"]:
            lines.append(f"   {st['override']} re-sorted parts would have been sorted with allow_single_certain = true")
        return "\n".join(lines)

    # ---------------------------------------------------------------------------------- drawing
    def draw(self, frame, fps=(0, 0)):
        cfg = self.cfg
        out = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if frame.ndim == 2 else frame.copy()
        x0, y0, x1, y1 = cfg.lane_rect(frame.shape)
        ppm, ll = cfg.px_per_mm, (x1 - x0)
        fx = (lambda f: int(x0 + f * ll)) if cfg.flow_left_to_right else (lambda f: int(x1 - f * ll))
        cv2.rectangle(out, (x0, y0), (x1 - 1, y1 - 1), (255, 150, 0), 1)
        for f in cfg.zone:
            cv2.line(out, (fx(f), y0), (fx(f), y1), (120, 120, 120), 1)
        cv2.line(out, (fx(cfg.handoff_frac), y0), (fx(cfg.handoff_frac), y1), (0, 220, 255), 2)
        cv2.putText(out, "DECIDE", (fx(cfg.handoff_frac) + 4, y0 + 18), 0, 0.5, (0, 220, 255), 1)
        tracked = {id(tr.blob): tr for tr in self.tracker.tracks}
        for b in self.blobs:
            tr = tracked.get(id(b))
            col = (0, 165, 255) if not b.clean else (60, 200, 60)
            txt = f"#{tr.id}" if tr else ""
            if tr is not None and tr.decided:
                col = (160, 160, 160) if tr.decision in (RESORT, "skipped") else (255, 200, 0)
                txt += f" {tr.decision}"
            elif b.clean:
                txt += f" {b.geo['L']:.1f}x{b.geo['W']:.1f}"
            cv2.drawContours(out, [b.contour], -1, col, 2)
            cv2.putText(out, txt, (b.bbox[0], max(b.bbox[1] - 6, 14)), 0, 0.5, (0, 0, 0), 3)
            cv2.putText(out, txt, (b.bbox[0], max(b.bbox[1] - 6, 14)), 0, 0.5, col, 1)
        st = self.stats
        hud = [f"{'SETUP' if self.mode == 'view' else self.mode.upper()}{' ' + self.label if self.label else ''} | cam {fps[0]:.0f} fps, loop {fps[1]:.0f} fps | "
               f"belt {self.tracker.belt_speed:.0f} mm/s | {ppm:.2f} px/mm | bg: {self.seg.bg_mode}"]
        if self.mode == "teach":
            hud.append(f"examples saved for {self.label}: {self.taught}")
        elif self.mode == "run":
            hud.append(f"parts {st['parts']}  sorted {st['sorted']}  re-sort {st['resort']}  "
                       f"missed {self.sched.counts['missed']}  queue {len(self.sched.queue)}")
        if self.health.problem:
            hud.append("PROBLEM: " + self.health.problem + " -> everything goes to re-sort")
        for text in self.notes.values():
            hud.append("PROBLEM: " + text)
        if self.feed:
            hud.append(self.next_prompt().strip())
        for i, (pid, dec, why) in enumerate(self.recent[::-1][:5]):
            hud.append(f"  #{pid}: {dec}" + (f" ({why})" if why else ""))
        bar = np.zeros((20 * 8 + 8, out.shape[1], 3), np.uint8)       # status bar above the picture
        for i, s in enumerate(hud[:8]):
            warn = s.startswith("PROBLEM")
            cv2.putText(bar, s, (8, 20 + 20 * i), 0, 0.5, (0, 0, 255) if warn else (255, 255, 255), 1)
        return np.vstack([bar, out])


def build_models(cfg, quiet=False):
    """Read the teach folder, fit both models, save them, and report how well the parts separate."""
    from models import load_samples
    geo_s, app_s = load_samples(TEACH_DIR)
    if not geo_s:
        raise SystemExit(f"No examples in '{TEACH_DIR}/'. Run teach first.")
    geo, app = GeoModel(cfg), AppModel(cfg)
    lines = ["Geometric model (mm):", geo.fit(geo_s), "", "Appearance model:", app.fit(app_s), ""]
    os.makedirs(MODEL_DIR, exist_ok=True)
    geo.save(os.path.join(MODEL_DIR, "geo.json"))
    app.save(os.path.join(MODEL_DIR, "app.npz"))
    # how would the taught examples themselves be judged?
    lines.append("Taught examples re-checked by the geometric model:")
    worst = {}
    for label, rows in sorted(geo_s.items()):
        lv = {}
        for r in rows:
            v = geo.classify(r)
            key = v.level if v.label in (label, None) else f"WRONG ({v.label})"
            lv[key] = lv.get(key, 0) + 1
            if v.rival:
                k = tuple(sorted((label, v.rival)))
                worst[k] = min(worst.get(k, 99.0), v.rival_score)
        lines.append(f"  {label}: " + ", ".join(f"{k} {n}" for k, n in sorted(lv.items())))
    lines.append("Closest pairs (rival's worst z; below 5 means these two are hard to tell apart by size):")
    for (a, b), z in sorted(worst.items(), key=lambda kv: kv[1])[:6]:
        lines.append(f"  {a} / {b}: {z:.1f}")
    missing = [l for l in geo_s if l not in cfg.slots]
    if missing:
        lines.append(f"NOTE: no slot assigned in config.json for: {', '.join(missing)} (they will go to re-sort)")
    text = "\n".join(lines)
    with open(os.path.join(MODEL_DIR, "report.txt"), "w") as fh:
        fh.write(text + "\n")
    if not quiet:
        print(text)
    return text
