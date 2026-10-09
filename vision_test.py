#!/usr/bin/env python3
import argparse
import csv
import glob
import json
import os
import sys
import time
from datetime import datetime

import cv2
import numpy as np

# ---------------------------------------------------------------- tunables ---
# Classification thresholds - tweak these once you see real numbers on screen.
RULES = {
    "hole_min_frac": 0.02,      # a child contour counts as a hole if > 2% of outer area
    "hole_min_circularity": 0.5,  # ...and is roundish (glare streaks on bolts are long slivers)
    "round_max_aspect": 1.35,   # nuts/washers lying flat are roughly as wide as tall
    "washer_min_circle_fill": 0.90,  # area / min-enclosing-circle area: circle ~1.0, hexagon ~0.83
}

MODES = ["blackhat", "otsu", "adaptive", "manual", "bg_subtract"]
CLASS_COLORS = {  # BGR
    "nut": (0, 165, 255),
    "bolt": (60, 200, 60),
    "washer": (255, 140, 0),
    "unknown": (0, 0, 255),
    "edge": (150, 150, 150),
}
LABEL_KEYS = {ord("1"): "nut", ord("2"): "bolt", ord("3"): "washer"}
IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

DEFAULTS = {
    "blur": 5,          # Gaussian kernel (odd); 0 = off
    "mode": 0,          # index into MODES (0 = blackhat, handles uneven lighting)
    "thresh": 128,      # manual threshold
    "polarity": 0,      # 0 auto, 1 parts darker than background, 2 parts brighter
    "block": 51,        # adaptive threshold block size (odd)
    "adapt_c": 5,       # adaptive threshold constant
    "bg_thresh": 30,    # background-subtraction difference threshold
    "bh_size": 201,     # blackhat kernel (px) - must be WIDER than the thickest solid part (1/2" bolt shank) in px
    "bh_thresh": 25,    # blackhat threshold - lower picks up fainter parts, higher = less noise
    "morph": 7,         # morphology kernel size; 0 = off
    "min_area": 300,    # ignore blobs smaller than this (px^2)
}


# ------------------------------------------------------------ frame source ---
class FrameSource:
    """Camera index, video file, single image, or a folder of images."""

    def __init__(self, src, width=0, height=0):
        self.images, self.idx, self.cap, self.static = [], 0, None, None
        if src.isdigit():
            backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
            self.cap = cv2.VideoCapture(int(src), backend)
            if width:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            if height:
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            if not self.cap.isOpened():
                sys.exit(f"Could not open camera {src}. Run with --list to see indices.")
        elif os.path.isdir(src):
            self.images = sorted(p for p in glob.glob(os.path.join(src, "*"))
                                 if p.lower().endswith(IMG_EXT))
            if not self.images:
                sys.exit(f"No images found in {src}")
        elif src.lower().endswith(IMG_EXT):
            self.images = [src]
        else:
            self.cap = cv2.VideoCapture(src)
            if not self.cap.isOpened():
                sys.exit(f"Could not open video {src}")
        self.is_live = self.cap is not None

    def read(self):
        if self.cap is not None:
            ok, frame = self.cap.read()
            return frame if ok else None
        if self.static is None:
            self.static = cv2.imread(self.images[self.idx])
            if self.static is None:
                sys.exit(f"Could not read image {self.images[self.idx]}")
        return self.static.copy()

    def step(self, d):
        if self.images:
            self.idx = (self.idx + d) % len(self.images)
            self.static = None
            print(f"[image] {self.images[self.idx]}")

    def name(self):
        return os.path.basename(self.images[self.idx]) if self.images else "live"

    def release(self):
        if self.cap is not None:
            self.cap.release()


def list_cameras(max_idx=8):
    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
    print("Probing camera indices (the OBS Virtual Camera shows OBS's output resolution):")
    for i in range(max_idx):
        cap = cv2.VideoCapture(i, backend)
        if cap.isOpened():
            ok, frame = cap.read()
            if ok:
                h, w = frame.shape[:2]
                print(f"  [{i}] {w}x{h}")
            else:
                print(f"  [{i}] opened but returned no frame (OBS Virtual Camera not started?)")
        cap.release()


# -------------------------------------------------------------- controls ----
class Controls:
    """Trackbars in a 'Controls' window, or fixed defaults when headless."""

    SPEC = [("blur", 31), ("mode", len(MODES) - 1), ("thresh", 255), ("polarity", 2),
            ("block", 201), ("adapt_c", 30), ("bg_thresh", 120), ("bh_size", 301), ("bh_thresh", 100), ("morph", 21),
            ("min_area", 5000)]

    def __init__(self, gui):
        self.gui = gui
        self.vals = dict(DEFAULTS)
        if gui:
            cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Controls", 420, 460)
            for name, mx in self.SPEC:
                cv2.createTrackbar(name, "Controls", self.vals[name], mx, lambda v: None)

    def get(self):
        if self.gui:
            for name, _ in self.SPEC:
                self.vals[name] = cv2.getTrackbarPos(name, "Controls")
        return self.vals

    def set(self, name, value):
        self.vals[name] = value
        if self.gui:
            cv2.setTrackbarPos(name, "Controls", value)


def odd(n, minimum=1):
    n = max(n, minimum)
    return n if n % 2 else n + 1


# ---------------------------------------------------------- segmentation ----
def segment(gray, p, bg_gray=None):
    """Return a binary mask with parts = 255, background = 0."""
    g = cv2.GaussianBlur(gray, (odd(p["blur"]),) * 2, 0) if p["blur"] > 0 else gray
    mode = MODES[p["mode"]]

    if mode == "blackhat":
        # Morphological closing with a big kernel "erases" the parts and leaves only the
        # (uneven) lighting; subtracting gives just the parts. Bright spots/vignetting cancel out.
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (odd(p["bh_size"], 15),) * 2)
        bright_parts = p["polarity"] == 2 or (p["polarity"] == 0 and np.median(g) < 100)
        op = cv2.MORPH_TOPHAT if bright_parts else cv2.MORPH_BLACKHAT
        hat = cv2.morphologyEx(g, op, k)
        _, mask = cv2.threshold(hat, max(p["bh_thresh"], 1), 255, cv2.THRESH_BINARY)
    elif mode == "bg_subtract" and bg_gray is not None:
        bg = cv2.GaussianBlur(bg_gray, (odd(p["blur"]),) * 2, 0) if p["blur"] > 0 else bg_gray
        diff = cv2.absdiff(g, bg)
        _, mask = cv2.threshold(diff, p["bg_thresh"], 255, cv2.THRESH_BINARY)
    else:
        if mode == "adaptive":
            mask = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, odd(p["block"], 3), p["adapt_c"])
        elif mode == "manual":
            _, mask = cv2.threshold(g, p["thresh"], 255, cv2.THRESH_BINARY)
        else:  # otsu (also the fallback if bg_subtract has no background yet)
            _, mask = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Polarity: we want parts white. Auto = assume the image border is background.
        if p["polarity"] == 1:
            mask = cv2.bitwise_not(mask)
        elif p["polarity"] == 0:
            border = np.concatenate([mask[0], mask[-1], mask[:, 0], mask[:, -1]])
            if border.mean() > 127:
                mask = cv2.bitwise_not(mask)

    if p["morph"] > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (odd(p["morph"]),) * 2)
        k_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (odd(p["morph"] // 2),) * 2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_small)  # remove specks
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)       # bridge glare gaps
    return mask


# -------------------------------------------------------------- analysis ----
def classify(f):
    if f["touches_edge"]:
        return "edge"
    if f["holes"] >= 1 and f["aspect"] <= RULES["round_max_aspect"]:
        return "washer" if f["circle_fill"] >= RULES["washer_min_circle_fill"] else "nut"
    if f["holes"] == 0:  # parts lying flat: only a bolt (on its side) has no hole
        return "bolt"
    return "unknown"


def analyze(mask, gray, min_area, px_per_mm=None):
    contours, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hier is None:
        return []
    hier = hier[0]
    H, W = mask.shape[:2]
    objects = []
    for i, c in enumerate(contours):
        if hier[i][3] != -1:  # only outer contours here; holes handled below
            continue
        area = cv2.contourArea(c)
        if area < min_area:
            continue

        # holes = child contours of this outer contour
        hole_area, holes, child = 0.0, 0, hier[i][2]
        while child != -1:
            a = cv2.contourArea(contours[child])
            hp = cv2.arcLength(contours[child], True)
            h_circ = 4 * np.pi * a / max(hp * hp, 1e-6)
            if a > RULES["hole_min_frac"] * area and h_circ >= RULES["hole_min_circularity"]:
                holes += 1
                hole_area += a
            child = hier[child][0]

        peri = cv2.arcLength(c, True)
        (cx, cy), (rw, rh), angle = cv2.minAreaRect(c)
        long_px, short_px = max(rw, rh), max(min(rw, rh), 1e-6)
        (_, _), r_enc = cv2.minEnclosingCircle(c)
        hull = cv2.convexHull(c)
        hull_area = max(cv2.contourArea(hull), 1e-6)
        approx = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True)
        x, y, w, h = cv2.boundingRect(c)

        roi = gray[y:y + h, x:x + w]
        f = {
            "bbox": [int(x), int(y), int(w), int(h)],
            "center": [round(cx, 1), round(cy, 1)],
            "area_px": round(area, 1),
            "holes": holes,
            "hole_ratio": round(hole_area / area, 3),
            "circularity": round(4 * np.pi * area / max(peri * peri, 1e-6), 3),
            "aspect": round(long_px / short_px, 3),
            "solidity": round(area / hull_area, 3),
            "circle_fill": round(area / (np.pi * r_enc * r_enc), 3),
            "hull_vertices": int(len(approx)),
            "long_px": round(long_px, 1),
            "short_px": round(short_px, 1),
            "sharpness": round(float(cv2.Laplacian(roi, cv2.CV_64F).var()), 1) if roi.size else 0.0,
            "touches_edge": bool(x <= 1 or y <= 1 or x + w >= W - 1 or y + h >= H - 1),
            "contour": c,
        }
        if px_per_mm:
            f["long_mm"] = round(long_px / px_per_mm, 2)
            f["short_mm"] = round(short_px / px_per_mm, 2)
        f["label"] = classify(f)
        objects.append(f)
    return objects


def frame_stats(gray):
    return {
        "sharpness": round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 1),
        "brightness": round(float(gray.mean()), 1),
        "contrast": round(float(gray.std()), 1),
    }


# --------------------------------------------------------------- drawing ----
def draw_overlay(frame, objects, stats, info):
    out = frame.copy()
    for k, o in enumerate(objects):
        col = CLASS_COLORS[o["label"]]
        cv2.drawContours(out, [o["contour"]], -1, col, 2)
        x, y, w, h = o["bbox"]
        size = (f'{o["long_mm"]:.1f}x{o["short_mm"]:.1f}mm' if "long_mm" in o
                else f'{o["long_px"]:.0f}x{o["short_px"]:.0f}px')
        lines = [f'#{k} {o["label"]}', size,
                 f'h{o["holes"]} ar{o["aspect"]:.2f} cf{o["circle_fill"]:.2f}']
        above = y > 100  # put labels below the part if it's near the top (HUD lives there)
        for j, t in enumerate(lines):
            ty = (y - 8 - 16 * (len(lines) - 1 - j)) if above else (y + h + 16 + 16 * j)
            cv2.putText(out, t, (x, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
            cv2.putText(out, t, (x, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)

    hud = [f'mode {info["mode"]} | {info["fps"]:.0f} fps | height {info["height"]}',
           f'sharp {stats["sharpness"]:.0f} | bright {stats["brightness"]:.0f} | '
           f'contrast {stats["contrast"]:.0f}',
           f'px/mm {info["px_per_mm"]:.2f}' if info["px_per_mm"] else "px/mm: not calibrated (c)",
           f'objects {len(objects)}' + ("  [PAUSED]" if info.get("paused") else "")]
    cv2.rectangle(out, (0, 0), (430, 18 * len(hud) + 8), (0, 0, 0), -1)
    for i, t in enumerate(hud):
        cv2.putText(out, t, (6, 18 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return out


def side_by_side(overlay, mask, max_w=1500):
    m = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    combo = np.hstack([overlay, m])
    if combo.shape[1] > max_w:
        s = max_w / combo.shape[1]
        combo = cv2.resize(combo, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return combo


# ----------------------------------------------------------------- saving ---
def stamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]


def save_crops(frame, objects, label, root, pad=12):
    d = os.path.join(root, "dataset", label)
    os.makedirs(d, exist_ok=True)
    H, W = frame.shape[:2]
    n = 0
    for o in objects:
        if o["label"] == "edge":
            continue
        x, y, w, h = o["bbox"]
        crop = frame[max(y - pad, 0):min(y + h + pad, H), max(x - pad, 0):min(x + w + pad, W)]
        cv2.imwrite(os.path.join(d, f"{stamp()}_{n}.png"), crop)
        n += 1
    print(f"[dataset] saved {n} crop(s) as '{label}' -> {d}")


def log_row(root, info, stats, objects, frame, overlay):
    rdir = os.path.join(root, "results")
    os.makedirs(rdir, exist_ok=True)
    path = os.path.join(rdir, "height_log.csv")
    ts = stamp()
    valid = [o for o in objects if o["label"] != "edge"]
    row = {
        "timestamp": ts,
        "height": info["height"],
        "mode": info["mode"],
        "px_per_mm": round(info["px_per_mm"], 3) if info["px_per_mm"] else "",
        "frame_sharpness": stats["sharpness"],
        "brightness": stats["brightness"],
        "contrast": stats["contrast"],
        "n_objects": len(valid),
        "mean_object_sharpness": round(float(np.mean([o["sharpness"] for o in valid])), 1) if valid else "",
        "labels": " ".join(o["label"] for o in valid),
        "sizes_mm": " ".join(f'{o.get("long_mm", "")}x{o.get("short_mm", "")}' for o in valid),
    }
    new = not os.path.exists(path)
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
    cv2.imwrite(os.path.join(rdir, f"{ts}_{info['height']}_raw.png"), frame)
    cv2.imwrite(os.path.join(rdir, f"{ts}_{info['height']}_overlay.png"), overlay)
    print(f"[log] {row}")


def load_calib(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def save_calib(path, calib):
    with open(path, "w") as fh:
        json.dump(calib, fh, indent=2)


# ------------------------------------------------------------------- main ---
def run_headless(args):
    """Process one image (or each image in a folder) once and dump features. Used for testing."""
    src = FrameSource(args.source)
    p = dict(DEFAULTS)
    p["mode"] = MODES.index(args.mode)
    os.makedirs(args.out, exist_ok=True)
    for _ in range(len(src.images) or 1):
        frame = src.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = segment(gray, p)
        objs = analyze(mask, gray, p["min_area"], args.px_per_mm)
        stats = frame_stats(gray)
        info = {"mode": args.mode, "fps": 0, "height": args.heights[0], "px_per_mm": args.px_per_mm}
        ov = draw_overlay(frame, objs, stats, info)
        base = os.path.splitext(src.name())[0]
        cv2.imwrite(os.path.join(args.out, f"{base}_view.png"), side_by_side(ov, mask))
        print(f"== {src.name()}  {stats}")
        for k, o in enumerate(objs):
            print(f"  #{k} {o['label']:7s}", {kk: v for kk, v in o.items()
                                             if kk not in ("contour", "bbox", "center", "label")})
        src.step(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="0", help="camera index, image, image folder, or video file")
    ap.add_argument("--list", action="store_true", help="list camera indices and exit")
    ap.add_argument("--width", type=int, default=0, help="request capture width")
    ap.add_argument("--height-px", type=int, default=0, help="request capture height")
    ap.add_argument("--heights", default="slot1,slot2,slot3,slot4,slot5,slot6,slot7",
                    help="comma-separated labels for the plate positions you test, e.g. 25mm,50mm,75mm")
    ap.add_argument("--ref-mm", type=float, default=24.26,
                    help="diameter of round calibration object in mm (US quarter = 24.26)")
    ap.add_argument("--px-per-mm", type=float, default=None, help="override calibration")
    ap.add_argument("--mode", default="blackhat", choices=MODES, help="starting segmentation mode")
    ap.add_argument("--out", default=".", help="folder for dataset/, results/, calibration.json")
    ap.add_argument("--headless", action="store_true", help="no GUI; process images once")
    args = ap.parse_args()
    args.heights = [h.strip() for h in args.heights.split(",") if h.strip()]

    if args.list:
        list_cameras()
        return
    if args.headless:
        run_headless(args)
        return

    os.makedirs(args.out, exist_ok=True)
    calib_path = os.path.join(args.out, "calibration.json")
    calib = load_calib(calib_path)

    src = FrameSource(args.source, args.width, args.height_px)
    ctl = Controls(gui=True)
    ctl.set("mode", MODES.index(args.mode))
    cv2.namedWindow("Vision", cv2.WINDOW_NORMAL)

    h_idx, bg_gray, paused, frame = 0, None, False, None
    t_prev, fps = time.time(), 0.0
    print(__doc__)

    while True:
        if not paused or frame is None:
            new = src.read()
            if new is None:
                print("No frame (end of video or camera dropped).")
                break
            frame = new
        p = ctl.get()
        height = args.heights[h_idx]
        px_per_mm = args.px_per_mm or calib.get(height)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if bg_gray is not None and bg_gray.shape != gray.shape:
            bg_gray = None  # crop changed in OBS -> background no longer valid
        mask = segment(gray, p, bg_gray)
        objs = analyze(mask, gray, max(p["min_area"], 1), px_per_mm)
        stats = frame_stats(gray)

        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-6))
        t_prev = now
        mode_name = MODES[p["mode"]] + ("" if MODES[p["mode"]] != "bg_subtract" or bg_gray is not None
                                        else " (press b!)")
        info = {"mode": mode_name, "fps": fps, "height": height,
                "px_per_mm": px_per_mm, "paused": paused}
        overlay = draw_overlay(frame, objs, stats, info)
        cv2.imshow("Vision", side_by_side(overlay, mask))

        key = cv2.waitKey(1 if src.is_live else 30) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord(" "):
            paused = not paused
        elif key == ord("m"):
            ctl.set("mode", (p["mode"] + 1) % len(MODES))
        elif key == ord("b"):
            bg_gray = gray.copy()
            ctl.set("mode", MODES.index("bg_subtract"))
            print("[bg] background captured - switched to bg_subtract mode")
        elif key == ord("h"):
            h_idx = (h_idx + 1) % len(args.heights)
            print(f"[height] {args.heights[h_idx]}")
        elif key == ord("g"):
            h_idx = (h_idx - 1) % len(args.heights)
            print(f"[height] {args.heights[h_idx]}")
        elif key == ord("c"):
            cands = [o for o in objs if o["label"] != "edge"]
            if not cands:
                print("[calib] no object found - put the reference coin in view")
            else:
                ref = max(cands, key=lambda o: o["area_px"])
                (_, _), r = cv2.minEnclosingCircle(ref["contour"])
                calib[height] = round(2 * r / args.ref_mm, 4)
                save_calib(calib_path, calib)
                print(f"[calib] {height}: {calib[height]} px/mm  "
                      f"(field of view ~ {frame.shape[1] / calib[height]:.0f} x "
                      f"{frame.shape[0] / calib[height]:.0f} mm)")
        elif key == ord("l"):
            log_row(args.out, info, stats, objs, frame, overlay)
        elif key == ord("s"):
            d = os.path.join(args.out, "dataset", "raw")
            os.makedirs(d, exist_ok=True)
            cv2.imwrite(os.path.join(d, f"{stamp()}_{height}.png"), frame)
            print(f"[dataset] raw frame saved -> {d}")
        elif key in LABEL_KEYS:
            save_crops(frame, objs, LABEL_KEYS[key], args.out)
        elif key == ord("n"):
            src.step(1)
        elif key == ord("p"):
            src.step(-1)

    src.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
