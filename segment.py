"""Find parts on the belt and measure them.

Each frame:
  1. brightness / empty-belt brightness  ->  a "ratio" image (1.0 = bare belt). Using a ratio makes
     the result independent of exposure and of the lighting gradient across the belt.
  2. a pixel is part of something if the ratio is clearly darker (< dark_ratio), clearly brighter
     (> bright_ratio, glints on shiny steel), or sits on a sharp edge (edge_grad). Sharp edges catch
     facets that reflect the diffuser and look as bright as the belt. Soft shadows (ratio 0.75-0.9,
     no sharp edge) are deliberately left out.
  3. clean up, trace outlines, measure in mm.

Two kinds of numbers come out per part:
  * geometric measurements (mm)            -> used by the geometric model
  * an appearance descriptor (how it looks inside and around the outline, rotation-independent)
                                           -> used by the trained appearance model
"""
from dataclasses import dataclass, field

import cv2
import numpy as np

GEO = ["L", "W", "A", "Dh", "S", "Wm"]   # length, width, area, hole diameter, solidity, median width

# ---- appearance descriptor layout (all sizes in mm, so it carries real scale) -----------------
R_MM = 20.0          # half-size of the canonical window
Q = 6.0              # px per mm in the canonical window
RING_MM = 0.75
N_RING = int(R_MM / RING_MM)             # 26 rings
N_AX = int(2 * R_MM)                     # 40 one-mm bins along the long axis
N_HIST = 8
BLOCKS = [("ring_mean", N_RING, 0.15), ("ring_std", N_RING, 0.15), ("width", N_AX, 1.5),
          ("axis_int", N_AX, 0.20), ("hist", N_HIST, 0.10)]           # name, length, typical scale
DESC_LEN = sum(n for _, n, _ in BLOCKS)

_side = int(2 * R_MM * Q) + 1
_yy, _xx = np.mgrid[0:_side, 0:_side].astype(np.float32)
_rr = np.hypot(_xx - _side // 2, _yy - _side // 2) / Q
_RING_IDX = np.minimum((_rr / RING_MM).astype(np.int32), N_RING)      # index N_RING = outside
_RING_CNT = np.bincount(_RING_IDX.ravel(), minlength=N_RING + 1).astype(np.float32)
_AX_IDX = np.minimum((_xx[0] / Q).astype(np.int32), N_AX - 1)


def _odd(n, minimum=1):
    n = max(int(round(n)), minimum)
    return n if n % 2 else n + 1


@dataclass
class Blob:
    x_mm: float                 # along the belt, 0 at the upstream lane edge
    y_mm: float                 # across the belt
    cx: float                   # centroid, full-resolution px
    cy: float
    contour: np.ndarray         # outer outline, full-resolution px
    bbox: tuple
    area_mm2: float
    clean: bool                 # fully in view, sane size -> may be measured
    touch_up: bool = False      # touching the upstream / downstream / side lane edge
    touch_down: bool = False
    touch_side: bool = False
    oversize: bool = False
    span: bool = False          # reaches across the whole belt: the belt seam, not a part
    is_seam: bool = False       # set by the tracker: part of the seam, not to be tracked as a part
    geo: dict = field(default_factory=dict)
    _det: dict = field(default_factory=dict, repr=False)    # detection-scale data for the descriptor


class Segmenter:
    def __init__(self, cfg):
        self.cfg = cfg
        self.bg = None            # full-frame grayscale picture of the empty belt
        self.bg_mode = "median"
        self.ratio = None         # last ratio image (detection scale, lane only)
        self.mask = None
        self._bg_cache = None

    # ------------------------------------------------------------------------------- background
    def set_background(self, gray):
        self.bg = None if gray is None else np.asarray(gray, np.uint8)
        self._bg_cache = None

    def auto_lane(self, gray, margin_mm=2.0):
        """Find the belt as the bright band in an empty-belt picture. Returns [x0,y0,x1,y1]."""
        h, w = gray.shape[:2]
        rows = np.median(gray, axis=1)
        level = np.percentile(rows, 80)
        ok = rows > 0.6 * level
        best, start = (0, 0), None                       # longest run of belt rows
        for i, v in enumerate(np.append(ok, False)):
            if v and start is None:
                start = i
            elif not v and start is not None:
                if i - start > best[1] - best[0]:
                    best = (start, i)
                start = None
        m = int(round(margin_mm * self.cfg.px_per_mm))
        y0, y1 = best[0] + m, best[1] - m
        if y1 - y0 < 0.2 * h:
            return [0, 0, w, h]
        return [0, int(y0), int(w), int(y1)]

    def _belt(self, g, lane, ppm):
        """Expected empty-belt brightness for every pixel of the lane image g."""
        cfg = self.cfg
        x0, y0, x1, y1 = lane
        if self.bg is not None and self.bg.shape[:2] == self._shape[:2]:
            key = (lane, g.shape)
            if self._bg_cache is None or self._bg_cache[0] != key:
                b = cv2.resize(self.bg[y0:y1, x0:x1], (g.shape[1], g.shape[0]), interpolation=cv2.INTER_AREA)
                if cfg.blur_mm > 0:
                    b = cv2.GaussianBlur(b, (_odd(cfg.blur_mm * ppm * 2, 3),) * 2, 0)
                self._bg_cache = (key, np.maximum(b.astype(np.float32), 1.0))
            self.bg_mode = "background"
            return self._bg_cache[1]
        k = cfg.bg_kernel_mm * ppm                        # fallback: big median ignores the parts
        s2 = min(1.0, 31.0 / k)
        small = cv2.resize(g, None, fx=s2, fy=s2, interpolation=cv2.INTER_AREA) if s2 < 1 else g
        med = cv2.medianBlur(small, _odd(k * s2, 3))
        if s2 < 1:
            med = cv2.resize(med, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_LINEAR)
        self.bg_mode = "median"
        return np.maximum(med.astype(np.float32), 1.0)

    # ---------------------------------------------------------------------------------- process
    def process(self, gray):
        cfg = self.cfg
        self._shape = gray.shape
        lane = cfg.lane_rect(gray.shape)
        x0, y0, x1, y1 = lane
        s = cfg.detect_scale
        g = gray[y0:y1, x0:x1]
        if s != 1:
            g = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        ppm = cfg.px_per_mm * s
        if cfg.blur_mm > 0:
            g = cv2.GaussianBlur(g, (_odd(cfg.blur_mm * ppm * 2, 3),) * 2, 0)
        belt = self._belt(g, lane, ppm)
        ratio = g.astype(np.float32) / belt
        valid = belt >= cfg.bg_min_level
        strong = (ratio < cfg.dark_ratio) | (ratio > cfg.bright_ratio)
        if cfg.edge_grad > 0:      # sharp edges: catches facets that happen to be as bright as the belt
            gx = cv2.Sobel(ratio, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(ratio, cv2.CV_32F, 0, 1, ksize=3)
            self.grad = cv2.magnitude(gx, gy) / 8.0
            strong |= self.grad > cfg.edge_grad
        mask = (strong & valid).astype(np.uint8) * 255
        if cfg.open_mm > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(cfg.open_mm * ppm, 3),) * 2))
        if cfg.close_mm > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(cfg.close_mm * ppm, 3),) * 2))
        self.ratio, self.mask, self.valid = ratio, mask, valid

        contours, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        blobs = []
        if hier is None:
            return blobs
        hier = hier[0]
        mh, mw = mask.shape
        hole_min_px = np.pi * (cfg.hole_min_mm * ppm / 2) ** 2
        for i, c in enumerate(contours):
            if hier[i][3] != -1:
                continue
            a_px = cv2.contourArea(c)
            a_mm = a_px / (ppm * ppm)
            if a_mm < cfg.min_area_mm2:
                continue
            bx, by, bw, bh = cv2.boundingRect(c)
            left, right = bx <= 1, bx + bw >= mw - 1
            side = by <= 1 or by + bh >= mh - 1
            up, down = (left, right) if cfg.flow_left_to_right else (right, left)
            over = a_mm > cfg.max_area_mm2
            m = cv2.moments(c)
            cx, cy = (m["m10"] / m["m00"], m["m01"] / m["m00"]) if m["m00"] else (bx + bw / 2, by + bh / 2)
            b = Blob(x_mm=(cx if cfg.flow_left_to_right else mw - cx) / ppm, y_mm=cy / ppm,
                     cx=cx / s + x0, cy=cy / s + y0,
                     contour=(c.astype(np.float32) / s + (x0, y0)).astype(np.int32),
                     bbox=(int(bx / s + x0), int(by / s + y0), int(bw / s), int(bh / s)),
                     area_mm2=a_mm, clean=not (up or down or side or over),
                     touch_up=up, touch_down=down, touch_side=side, oversize=over,
                     span=by <= 1 and by + bh >= mh - 1 and mh / ppm > cfg.seam_min_lane_mm)
            if b.clean:
                holes, child = [], hier[i][2]
                while child != -1:
                    ha = cv2.contourArea(contours[child])
                    if ha >= hole_min_px and self._is_through_hole(contours[child], ha):
                        holes.append(ha)
                    child = hier[child][0]
                (_, _), (rw, rh), ang = cv2.minAreaRect(c)
                if rw < rh:
                    rw, rh, ang = rh, rw, ang + 90.0
                hull = max(cv2.contourArea(cv2.convexHull(c)), 1e-9)
                theta = 0.5 * np.arctan2(2 * m["mu11"], m["mu20"] - m["mu02"])   # long axis from moments
                b._det = dict(c=c, cx=cx, cy=cy, theta=theta, ppm=ppm)
                wm = self._median_width(c, cx, cy, theta, ppm)
                b.geo = dict(L=rw / ppm, W=rh / ppm, A=a_mm,
                             Dh=2 * np.sqrt(max(holes) / np.pi) / ppm if holes else 0.0,
                             S=a_px / hull, Wm=wm)
            blobs.append(b)
        blobs.sort(key=lambda b: b.x_mm)
        return blobs

    # ---------------------------------------------------------------------------- measurements
    @staticmethod
    def _canon(img, cx, cy, theta, ppm, flip=False, interp=cv2.INTER_LINEAR, border=1.0):
        """Resample img into the canonical window: centred on the part, long axis horizontal, Q px/mm."""
        k = ppm / Q
        c, s_ = np.cos(theta) * k, np.sin(theta) * k
        f = -1.0 if flip else 1.0
        h = _side // 2
        M = np.array([[c * f, -s_, cx - (c * f * h - s_ * h)],
                      [s_ * f, c, cy - (s_ * f * h + c * h)]], np.float32)   # canonical -> source
        return cv2.warpAffine(img, M, (_side, _side), flags=interp | cv2.WARP_INVERSE_MAP,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=border)

    def _is_through_hole(self, hc, area):
        """A real hole shows bare belt (ratio near 1) and is roundish; a bright facet enclosed by
        edges is not counted."""
        per = cv2.arcLength(hc, True)
        if 4 * np.pi * area / max(per * per, 1e-9) < 0.6:
            return False
        x, y, w, h = cv2.boundingRect(hc)
        m = np.zeros((h, w), np.uint8)
        cv2.drawContours(m, [hc - (x, y)], -1, 255, -1)
        m = cv2.erode(m, np.ones((3, 3), np.uint8))
        vals = self.ratio[y:y + h, x:x + w][m > 0]
        return len(vals) > 0 and 0.85 < float(np.median(vals)) < 1.12

    def _filled(self, c):
        m = np.zeros(self.mask.shape, np.uint8)
        cv2.drawContours(m, [c], -1, 255, -1)
        return m

    def _median_width(self, c, cx, cy, theta, ppm):
        """Median thickness across the long axis (mm): the shank of a bolt, ignoring its head."""
        cm = self._canon(self._filled(c), cx, cy, theta, ppm, interp=cv2.INTER_NEAREST, border=0)
        cols = (cm > 0).sum(axis=0)
        cols = cols[cols > 0]
        return float(np.median(cols)) / Q if len(cols) else 0.0

    def describe(self, blob):
        """Appearance descriptor of a clean blob (see BLOCKS). Rotation-independent by construction."""
        d = blob._det
        c, cx, cy, theta, ppm = d["c"], d["cx"], d["cy"], d["theta"], d["ppm"]
        filled = self._filled(c)
        cm = self._canon(filled, cx, cy, theta, ppm, interp=cv2.INTER_NEAREST, border=0) > 0
        cols = cm.sum(axis=0).astype(np.float32)
        xs = np.arange(_side, dtype=np.float32) - _side // 2
        tot = max(cols.sum(), 1.0)
        mean = (cols * xs).sum() / tot
        sd = np.sqrt(max((cols * (xs - mean) ** 2).sum() / tot, 1e-6))
        skew = (cols * (xs - mean) ** 3).sum() / tot / sd ** 3
        flip = skew < -0.05                                   # put the long tail (shank) on the right
        if flip:
            cm = cm[:, ::-1]
        cr = np.clip(self._canon(self.ratio, cx, cy, theta, ppm, flip=flip), 0.0, 2.0)

        ring_sum = np.bincount(_RING_IDX.ravel(), weights=cr.ravel(), minlength=N_RING + 1)
        ring_sq = np.bincount(_RING_IDX.ravel(), weights=(cr * cr).ravel(), minlength=N_RING + 1)
        ring_mean = ring_sum / _RING_CNT
        ring_std = np.sqrt(np.maximum(ring_sq / _RING_CNT - ring_mean ** 2, 0.0))

        cols = cm.sum(axis=0).astype(np.float32)
        inten = (cr * cm).sum(axis=0)
        w_bin = np.bincount(_AX_IDX, weights=cols, minlength=N_AX) / Q
        i_bin = np.bincount(_AX_IDX, weights=inten, minlength=N_AX)
        n_bin = np.bincount(_AX_IDX, weights=cols, minlength=N_AX)
        axis_int = np.where(n_bin > 0, i_bin / np.maximum(n_bin, 1), 1.0)
        width = w_bin / Q                                      # mean thickness (mm) per 1 mm of length

        inside = cr[cm]
        hist = np.histogram(inside, bins=N_HIST, range=(0.0, 1.6))[0] / max(len(inside), 1)
        return np.concatenate([ring_mean[:N_RING], ring_std[:N_RING], width, axis_int, hist]).astype(np.float32)


def desc_distance(a, B):
    """Distance from descriptor a to each row of B. Every block is scaled so blocks count equally."""
    B = np.atleast_2d(B)
    out = np.zeros(len(B), np.float64)
    i = 0
    for _, n, scale in BLOCKS:
        d = (B[:, i:i + n] - a[i:i + n]) / scale
        out += (d * d).mean(axis=1)
        i += n
    return np.sqrt(out / len(BLOCKS))
