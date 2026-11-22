"""Belt simulator for testing without the rig.

Cuts each part (with its shadows and glints) out of a reference photo and slides it across an
empty-belt picture at a chosen speed, rotation and position, with sensor noise and motion blur.
It exercises tracking, teaching, both classifiers, the gap rule and the collector timing.

What it does NOT reproduce: parts looking different from pass to pass (it has one photo per part,
varied by rotation, mirroring, +-2% size and contrast), perspective on tall parts, rolling parts.
So its accuracy numbers are optimistic; its logic checks (gaps, re-sort rules, timing) are real.
"""
import glob
import os

import cv2
import numpy as np

from config import Config
from segment import Segmenter

HERE = os.path.dirname(os.path.abspath(__file__))
# px/mm of each reference photo, estimated from nominal part sizes (the 1/4" photos are zoomed in more)
PHOTO_PPM = {"bolt-0.25": 19.4, "nut-0.25": 19.4, "washer-0.25": 19.4,
             "bolt-0.375": 16.5, "nut-0.375": 16.5, "washer-0.375": 16.5}


class Sim:
    def __init__(self, ppm=14.3, size=(1280, 800), fps=120, speed=80.0, exposure_ms=1.0, noise=1.5, seed=0):
        self.ppm, self.size, self.fps, self.speed = ppm, size, fps, speed
        self.exposure_ms, self.noise = exposure_ms, noise
        self.rng = np.random.default_rng(seed)
        self.sprites = {}
        w, h = size
        bg = None
        for p in sorted(glob.glob(os.path.join(HERE, "ref", "*.jpg"))):
            name = os.path.basename(p)[:-4]
            g = cv2.imread(p, 0)
            cfg = Config(); cfg.px_per_mm = PHOTO_PPM[name]
            seg = Segmenter(cfg)
            cfg.lane = seg.auto_lane(g, 3.0)
            blobs = [b for b in seg.process(g) if b.clean]
            b = max(blobs, key=lambda b: b.area_mm2)
            s = cfg.detect_scale
            ratio = cv2.resize(seg.ratio, None, fx=1 / s, fy=1 / s, interpolation=cv2.INTER_LINEAR)
            m = np.zeros(ratio.shape, np.uint8)
            cv2.drawContours(m, [b.contour - (0, cfg.lane[1])], -1, 255, -1)
            pad = int(9 * cfg.px_per_mm)
            x, y, bw, bh = cv2.boundingRect(b.contour - (0, cfg.lane[1]))
            half = int(max(bw, bh) / 2 + pad)
            cx, cy = x + bw // 2, y + bh // 2
            win = lambda im, v: cv2.copyMakeBorder(im, half, half, half, half, cv2.BORDER_CONSTANT, value=v)[
                cy:cy + 2 * half, cx:cx + 2 * half]
            r, mk = win(ratio, 1.0), win(m, 0)
            # fade the surroundings to 1.0 (bare belt) away from the part, keeping the near shadows
            dist = cv2.distanceTransform(255 - mk, cv2.DIST_L2, 5) / cfg.px_per_mm
            fade = np.clip((8.0 - dist) / 3.0, 0, 1)
            r = 1.0 + (r - 1.0) * fade
            k = ppm / cfg.px_per_mm
            self.sprites[name] = (cv2.resize(r, None, fx=k, fy=k, interpolation=cv2.INTER_AREA),
                                  cv2.resize(mk, None, fx=k, fy=k, interpolation=cv2.INTER_AREA))
            if bg is None:                               # empty belt: this photo's background estimate
                small = cv2.resize(g, None, fx=0.125, fy=0.125, interpolation=cv2.INTER_AREA)
                bg = cv2.resize(cv2.medianBlur(small, 61), (w, h), interpolation=cv2.INTER_LINEAR)
        self.bg = cv2.GaussianBlur(bg, (0, 0), 3).astype(np.float32)
        self.parts = []
        rows = np.where(np.median(self.bg, axis=1) > 0.6 * np.percentile(self.bg, 80))[0]
        self.belt_mm = (rows.min() / ppm, rows.max() / ppm)
        self._noise = [self.rng.normal(0, noise, (h, w)).astype(np.float32) for _ in range(6)]

    # ---------------------------------------------------------------------------------- script
    def add(self, name, t_enter, y_mm=None, angle=None, scale=1.0, label=None, x_start_mm=None, vary=1.0,
            slip=1.0):
        """Queue a part. t_enter = time its centre crosses the left image edge."""
        rng = self.rng
        r, m = self.sprites[name]
        ang = rng.uniform(0, 360) if angle is None else angle
        sc = scale * (1.0 + vary * rng.uniform(-0.02, 0.02))
        if vary and rng.random() < 0.5:
            r, m = r[:, ::-1], m[:, ::-1]
        if vary:
            r = np.clip(1.0 + (r - 1.0) * (1.0 + vary * rng.uniform(-0.08, 0.08)), 0.02, 2.5)
            if vary > 1.0:                               # stress: slight defocus too
                r = cv2.GaussianBlur(r, (0, 0), rng.uniform(0.3, 0.6 * vary))
        c = (r.shape[1] / 2, r.shape[0] / 2)
        M = cv2.getRotationMatrix2D(c, ang, sc)
        r = cv2.warpAffine(r, M, r.shape[::-1], flags=cv2.INTER_LINEAR, borderValue=1.0)
        m = cv2.warpAffine(m, M, m.shape[::-1], flags=cv2.INTER_LINEAR, borderValue=0)
        blur = int(round(self.speed * self.exposure_ms / 1000.0 * self.ppm))
        if blur >= 2:
            r = cv2.blur(r, (blur, 1)); m = cv2.blur(m, (blur, 1))
        ys = np.where(m.max(axis=1) > 127)[0]
        half = (ys.max() - ys.min()) / 2 / self.ppm + 4.0            # keep the whole part on the belt
        lo, hi = self.belt_mm[0] + half, self.belt_mm[1] - half
        y = rng.uniform(lo, max(lo, hi)) if y_mm is None else y_mm
        mm_ = cv2.moments((m > 127).astype(np.uint8), binaryImage=True)
        off = mm_["m10"] / max(mm_["m00"], 1) - m.shape[1] / 2          # centroid relative to the sprite centre
        self.parts.append(dict(name=name, label=label if label is not None else name, t0=t_enter, y=y * self.ppm, off=off,
                               r=r.astype(np.float32), m=m, x0=(x_start_mm or 0.0) * self.ppm,
                               t_show=t_enter, v=slip))
        return self.parts[-1]

    def x_px(self, part, t):
        return part["x0"] + part["v"] * self.speed * self.ppm * (t - part["t0"])

    def t_cross(self, part, x_mm):
        """Time the part's centroid is x_mm from the left image edge."""
        return part["t0"] + (x_mm * self.ppm - part["off"] - part["x0"]) / (part["v"] * self.speed * self.ppm)

    def duration(self):
        return max(self.t_cross(p, self.size[0] / self.ppm + 30) for p in self.parts) if self.parts else 0.0

    # ---------------------------------------------------------------------------------- render
    def frame(self, i, light=1.0):
        t = i / self.fps
        w, h = self.size
        ratio = np.ones((h, w), np.float32)
        for p in self.parts:
            if t < p["t_show"]:
                continue
            x = self.x_px(p, t)
            hs = p["r"].shape[0] / 2
            if x < -hs or x > w + hs:
                continue
            xi, yi = int(np.floor(x - hs)), int(round(p["y"] - hs))
            M = np.float32([[1, 0, x - hs - xi], [0, 1, 0]])
            n = p["r"].shape[0]
            r = cv2.warpAffine(p["r"], M, (n, n), flags=cv2.INTER_LINEAR, borderValue=1.0)
            m = cv2.warpAffine(p["m"], M, (n, n), flags=cv2.INTER_LINEAR, borderValue=0) > 127
            sx0, sy0 = max(0, -xi), max(0, -yi)
            sx1, sy1 = min(n, w - xi), min(n, h - yi)
            if sx1 <= sx0 or sy1 <= sy0:
                continue
            dst = ratio[yi + sy0:yi + sy1, xi + sx0:xi + sx1]
            rr, mm = r[sy0:sy1, sx0:sx1], m[sy0:sy1, sx0:sx1]
            dst[:] = np.where(mm, rr, dst * rr)             # part on top replaces; shadows multiply
        img = self.bg * ratio * (light * (1.0 + self.rng.normal(0, 0.003)))
        img += self._rand_noise()
        return np.clip(img, 0, 255).astype(np.uint8), t

    def _rand_noise(self):
        n = self._noise[self.rng.integers(len(self._noise))]
        return np.roll(n, (int(self.rng.integers(n.shape[0])), int(self.rng.integers(n.shape[1]))), axis=(0, 1))

    def background(self, n=20):
        acc = sum(self.frame_empty().astype(np.float32) for _ in range(n)) / n
        return acc.astype(np.uint8)

    def frame_empty(self):
        img = self.bg + self._rand_noise()
        return np.clip(img, 0, 255).astype(np.uint8)

    def frames(self, light=None):
        n = int(self.duration() * self.fps) + 2
        for i in range(n):
            yield self.frame(i, 1.0 if light is None else light(i / self.fps))
