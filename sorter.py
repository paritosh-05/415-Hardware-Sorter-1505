#!/usr/bin/env python3
"""Team 1505 hardware sorter - vision system v2.

  python sorter.py cams                      list cameras and serial ports
  python sorter.py setup                     camera settings, empty-belt background, lane, px/mm
  python sorter.py teach bolt-0.25           run ONLY that part through; saves examples
  python sorter.py build                     fit both models from the examples, print the report
  python sorter.py run                       sort. Add --port COM5 (Mac: /dev/cu.usbmodem...) to drive the collector
  python sorter.py run --truth nut-0.375     test run with one known part type: counts wrong bins
  python sorter.py check                     feed 20 parts in the order it lists; scores every one
                                             (check 30, --seed 4, --order "nut-0.25, bolt-0.375" or a file)
  python sorter.py record clips/mix1.avi     save raw footage (+ time stamps) for replay
  python sorter.py selftest                  simulated belt, no camera needed (about 2 minutes)

Any mode can replay footage instead of the camera:   --source clips/mix1.avi   (add --headless for no window)

Keys (click the video window first)
  all modes   q quit   space pause   m mask window   s save this frame to captures/
  setup       b grab EMPTY belt (background, lane, noise check)     c calibrate px/mm with a quarter,
              or any flat round object:  python sorter.py setup --ref-mm 18.6
              l draw lane by hand   f flip belt direction   w write config.json
              [ ] exposure shorter/longer   , . gain down/up   a auto exposure   k camera dialog   i camera info
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np

from camera import CameraSource, FileSource, Recorder, list_cameras
from collector import NullLink, SerialLink, list_ports
from config import QUARTER_MM, Config
from pipeline import BG_FILE, Pipeline, build_models


def fit(img, max_w=1500):
    if img.shape[1] <= max_w:
        return img
    s = max_w / img.shape[1]
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)


def contact_sheet(out_dir, tile=230, per_row=6):
    """One picture of every part in the run with the sorter's answer under it, in the order they passed."""
    folder = os.path.join(out_dir, "crops")
    names = sorted(n for n in os.listdir(folder) if n.endswith(".png")) if os.path.isdir(folder) else []
    if not names:
        return None
    tiles = []
    imgs = [cv2.imread(os.path.join(folder, n), cv2.IMREAD_GRAYSCALE) for n in names]
    s = (tile - 10) / max(max(i.shape) for i in imgs)       # one scale for all, so sizes can be compared
    for k, (n, img) in enumerate(zip(names, imgs)):
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        t = np.full((tile + 34, tile, 3), 255, np.uint8)
        y, x = (tile - img.shape[0]) // 2, (tile - img.shape[1]) // 2
        t[y:y + img.shape[0], x:x + img.shape[1]] = img[:, :, None]
        label = n[:-4].split("_", 1)[1]
        colour = (0, 0, 200) if label == "resort" else (0, 110, 0)
        cv2.putText(t, f"{k + 1}. {label}", (6, tile + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, colour, 2, cv2.LINE_AA)
        cv2.rectangle(t, (0, 0), (tile - 1, tile + 33), (190, 190, 190), 1)
        tiles.append(t)
    while len(tiles) % per_row:
        tiles.append(np.full_like(tiles[0], 255))
    sheet = np.vstack([np.hstack(tiles[i:i + per_row]) for i in range(0, len(tiles), per_row)])
    path = os.path.join(out_dir, "sheet.png")
    cv2.imwrite(path, sheet)
    return path


def grab_background(src, pipe, cfg, last_idx, n=30):
    """Average n frames of the empty running belt; set the lane if it is not set; check belt noise."""
    acc, k, idx = None, 0, last_idx
    t_end = time.perf_counter() + 5.0
    while k < n and time.perf_counter() < t_end:
        f = src.read(idx)
        if f is None:
            break
        idx = f.index
        g = cv2.cvtColor(f.image, cv2.COLOR_BGR2GRAY) if f.image.ndim == 3 else f.image
        acc = g.astype(np.float32) if acc is None else acc + g
        k += 1
    if not k:
        print("[background] no frames")
        return idx
    bg = (acc / k).astype(np.uint8)
    cv2.imwrite(BG_FILE, bg)
    pipe.seg.set_background(bg)
    if cfg.lane == [0, 0, 0, 0]:
        cfg.lane = pipe.seg.auto_lane(bg)
        print(f"[lane] found the belt automatically: {cfg.lane}  (press l to draw it by hand instead)")
    blobs = pipe.seg.process(g)
    noise = float(np.percentile(pipe.seg.grad[pipe.seg.valid], 99.9))
    print(f"[background] averaged {k} frames -> {BG_FILE}.  Empty-belt edge noise {noise:.4f} "
          f"(edge threshold {cfg.edge_grad}).")
    if noise * 2.5 > cfg.edge_grad:
        cfg.edge_grad = round(noise * 2.5, 3)
        print(f"   belt is noisier than expected -> edge threshold raised to {cfg.edge_grad}. More light / less gain helps.")
    if blobs:
        print(f"   WARNING: {len(blobs)} thing(s) detected on the 'empty' belt. Clear the belt and press b again.")
    return idx


def loop(args, cfg, mode, label=None):
    if args.source:
        src = FileSource(args.source, args.file_fps)
    else:
        if args.camera is not None:
            cfg.camera = args.camera
        if args.fps:
            cfg.fps = args.fps
        src = CameraSource(cfg)
        warn = src.warm_up()
        if warn:
            print(warn)
    link = NullLink()
    if mode == "run" and (args.port or cfg.port):
        link = SerialLink(args.port or cfg.port, cfg.baud)
    out_dir = None
    if mode == "run":
        out_dir = args.out or os.path.join("runs", time.strftime("%Y%m%d_%H%M%S"))
    order = getattr(args, "order_list", None)
    if order:
        print("\nFEED THESE PARTS IN THIS ORDER, one at a time, at least a hand-width apart. Press q when done.\n")
        for i in range(0, len(order), 5):
            print("   " + "   ".join(f"{i + k + 1:>2}. {name:<13}" for k, name in enumerate(order[i:i + 5])))
        print()
    pipe = Pipeline(cfg, mode="view" if mode == "record" else mode, label=label, link=link, out_dir=out_dir,
                    truth=getattr(args, "truth", None), quiet=False)
    rec = None
    print(__doc__ if not args.headless and not order else "", src.describe())
    if order:
        pipe.feed = order
        print(pipe.next_prompt())
    if not args.headless:
        cv2.namedWindow("Sorter", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    last_idx, f, paused, show_mask, mask_open = 0, None, False, False, False
    loop_fps, t_prev = 0.0, time.perf_counter()
    while True:
        if not paused or f is None:
            f = src.read(last_idx)
            if f is None:
                print("No more frames.")
                break
            last_idx = f.index
            if mode == "record":
                if rec is None:
                    os.makedirs(os.path.dirname(args.file) or ".", exist_ok=True)
                    rec = Recorder(args.file, f.image.shape, cfg.fps)
                rec.write(f.image, f.t)
            pipe.process(f.image, f.t, time.perf_counter() if src.live else f.t)
            for line in link.poll():
                if line.startswith("E"):
                    print("[collector]", line, "-> everything goes to re-sort until it reports READY")
                    pipe.health.external = "collector " + line
                elif line.startswith("READY"):
                    pipe.health.external = ""
        now = time.perf_counter()
        loop_fps = 0.9 * loop_fps + 0.1 / max(now - t_prev, 1e-6)
        t_prev = now
        if args.headless:
            continue
        cv2.imshow("Sorter", fit(pipe.draw(f.image, (src.capture_fps, loop_fps))))
        if show_mask and pipe.seg.mask is not None:
            if not mask_open:
                cv2.namedWindow("Mask", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                mask_open = True
            cv2.imshow("Mask", pipe.seg.mask)
        elif mask_open:
            cv2.destroyWindow("Mask")
            mask_open = False
        key = cv2.waitKey(1 if src.live else 15) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord(" "):
            paused = not paused
        elif key == ord("m"):
            show_mask = not show_mask
        elif key == ord("s"):
            os.makedirs("captures", exist_ok=True)
            p = os.path.join("captures", f"{time.strftime('%Y%m%d_%H%M%S')}_{last_idx}.png")
            cv2.imwrite(p, f.image)
            print("[saved]", p)
        elif mode == "view":
            last_idx = setup_key(key, src, pipe, cfg, f, last_idx, args)
    pipe.finish()
    if rec:
        rec.close()
        print(f"[record] {rec.n} frames -> {args.file}")
    src.release()
    cv2.destroyAllWindows()
    if mode == "run":
        print("\n" + pipe.summary())
        with open(os.path.join(out_dir, "summary.txt"), "w") as fh:
            fh.write(pipe.summary() + "\n")
        if order:
            import check
            text, _ = check.report(order, pipe.log)
            print("\n" + text)
            with open(os.path.join(out_dir, "check.txt"), "w") as fh:
                fh.write(text + "\n")
        sheet = contact_sheet(out_dir)
        if sheet:
            print(f"picture of every part with the sorter's answer: {sheet}")
        print(f"log: {out_dir}")
    if mode == "teach":
        print(f"\n{pipe.taught} examples of {label} saved. Run `python sorter.py build` when all parts are taught.")
    return pipe


def setup_key(key, src, pipe, cfg, f, last_idx, args):
    cam = src.live
    if key == ord("b"):
        print("[background] keep the belt EMPTY (running is fine)...")
        last_idx = grab_background(src, pipe, cfg, last_idx)
    elif key == ord("l"):
        r = cv2.selectROI("Draw lane, Enter to accept", f.image, showCrosshair=False)
        cv2.destroyWindow("Draw lane, Enter to accept")
        if r[2] > 0 and r[3] > 0:
            cfg.lane = [int(r[0]), int(r[1]), int(r[0] + r[2]), int(r[1] + r[3])]
            pipe.seg.set_background(pipe.seg.bg)
            print("[lane]", cfg.lane)
    elif key == ord("f"):
        cfg.flow_left_to_right = not cfg.flow_left_to_right
        print("[flow] left to right" if cfg.flow_left_to_right else "[flow] right to left")
    elif key == ord("c"):
        clean = [b for b in pipe.blobs if b.clean]
        if clean:
            b = max(clean, key=lambda b: b.area_mm2)
            (_, _), r = cv2.minEnclosingCircle(b.contour)
            ref = args.ref_mm or QUARTER_MM
            cfg.px_per_mm = round(2 * r / ref, 3)
            print(f"[calibrate] using a round object {ref} mm across")
            h, w = f.image.shape[:2]
            print(f"[calibrate] {cfg.px_per_mm} px/mm -> view is {w / cfg.px_per_mm:.0f} x {h / cfg.px_per_mm:.0f} mm")
        else:
            print("[calibrate] put the round reference object flat in the lane first")
    elif key == ord("w"):
        cfg.save(args.config)
        print(f"[config] saved to {args.config}")
    elif cam and key == ord("k"):
        if not src.open_settings():
            print("[camera] no settings dialog with this driver (it only exists on Windows). On macOS use a UVC "
                  "control app such as 'CameraController' or 'uvc-util'.")
    elif cam and key in (ord("["), ord("]")):
        cur = cfg.exposure if cfg.exposure is not None else round(src.cap.get(cv2.CAP_PROP_EXPOSURE))
        new = cur + (1 if key == ord("]") else -1)
        got = src.set_exposure(new)
        if got is None:
            print("[exposure] the camera driver refused it (usual on macOS): set exposure with a UVC control app")
        else:
            cfg.exposure = new
            print(f"[exposure] {got:.0f}  (~{1000 * 2 ** cfg.exposure:.1f} ms)")
    elif cam and key == ord("a"):
        cfg.exposure = None
        src.auto_exposure()
        print("[exposure] auto (not recommended for sorting: brightness will drift)")
    elif cam and key in (ord(","), ord(".")):
        cur = cfg.gain if cfg.gain is not None else src.cap.get(cv2.CAP_PROP_GAIN)
        new = max(0, cur + (8 if key == ord(".") else -8))
        got = src.set_gain(new)
        if got is None:
            print("[gain] the camera driver refused it (usual on macOS): set gain with a UVC control app")
        else:
            cfg.gain = new
            print(f"[gain] {got:.0f}")
    elif cam and key == ord("i"):
        print(src.describe())
    return last_idx


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["cams", "setup", "teach", "build", "run", "check", "record", "selftest"])
    ap.add_argument("arg", nargs="?", help="part label for teach, file name for record, number of parts for check")
    ap.add_argument("--order", help="check: your own feeding order, comma-separated or a text file")
    ap.add_argument("--seed", type=int, default=1, help="check: a different number gives a different order")
    ap.add_argument("--camera", type=int)
    ap.add_argument("--fps", type=int)
    ap.add_argument("--source", help="video file / image folder instead of the camera")
    ap.add_argument("--file-fps", type=float)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--port", help="collector serial port, e.g. COM5 on Windows, /dev/cu.usbmodem1101 on macOS")
    ap.add_argument("--ref-mm", type=float, help="setup: diameter in mm of the round object used for 'c' "
                                                 "(default: a US quarter, 24.26)")
    ap.add_argument("--truth", help="run mode: every part in this run is this label (for scoring)")
    ap.add_argument("--out", help="run mode: log folder (default runs/<time>)")
    args = ap.parse_args()
    cfg = Config.load(args.config)

    if args.mode == "cams":
        print("Cameras:")
        list_cameras()
        print("Serial ports (for --port):")
        list_ports()
    elif args.mode == "build":
        build_models(cfg)
    elif args.mode == "selftest":
        import selftest
        selftest.main()
    elif args.mode == "setup":
        loop(args, cfg, "view")
    elif args.mode == "teach":
        if not args.arg:
            ap.error("teach needs a label, e.g. teach nut-0.25")
        if args.arg not in cfg.slots:
            print(f"NOTE: '{args.arg}' has no slot in {args.config} yet (known: {', '.join(cfg.slots)}).")
        loop(args, cfg, "teach", args.arg)
    elif args.mode == "record":
        if not args.arg:
            ap.error("record needs a file name, e.g. record clips/mix1.avi")
        args.file = args.arg
        loop(args, cfg, "record")
    elif args.mode == "check":
        if not os.path.exists(os.path.join("model", "geo.json")):
            sys.exit("No model yet. Teach each part, then run: python sorter.py build")
        import check
        args.order_list = (check.read_order(args.order, cfg.slots) if args.order
                           else check.make_order(cfg.slots, int(args.arg or 20), args.seed))
        loop(args, cfg, "run")
    elif args.mode == "run":
        if not os.path.exists(os.path.join("model", "geo.json")):
            sys.exit("No model yet. Teach each part, then run: python sorter.py build")
        loop(args, cfg, "run")


if __name__ == "__main__":
    main()
