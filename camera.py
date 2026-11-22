"""Frame sources: live camera (background thread, every frame time-stamped), recorded files, recorder."""
import csv
import glob
import os
import sys
import threading
import time

import cv2

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp")
IS_WIN, IS_MAC = sys.platform.startswith("win"), sys.platform == "darwin"


def _backend():
    """DirectShow on Windows (exposure/gain/settings dialog work), AVFoundation on macOS."""
    if IS_WIN:
        return cv2.CAP_DSHOW
    if IS_MAC:
        return cv2.CAP_AVFOUNDATION
    return cv2.CAP_ANY


def _no_camera_help(index):
    msg = f"Could not open camera {index}. Run: python sorter.py cams"
    if IS_MAC:
        msg += ("\nOn macOS the app you run this from (Terminal, iTerm, VS Code...) needs camera access: "
                "System Settings > Privacy & Security > Camera. Quit and reopen it after allowing.")
    return msg


class Frame:
    __slots__ = ("image", "t", "index")

    def __init__(self, image, t, index):
        self.image, self.t, self.index = image, t, index


class CameraSource:
    """USB camera read in a background thread. Only the newest frame is kept, stamped with the time
    it arrived, so slow processing never shifts the timing used for the collector."""
    live = True

    def __init__(self, cfg):
        self.cap = cv2.VideoCapture(cfg.camera, _backend())
        if not self.cap.isOpened():
            sys.exit(_no_camera_help(cfg.camera))
        if cfg.mjpg:      # must be set before the size: 1280x800 @ 120 fps only exists in MJPG
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
        self.cap.set(cv2.CAP_PROP_FPS, cfg.fps)
        if cfg.mjpg:      # ask again: some drivers only accept the format once the size is known
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.requested_fps = cfg.fps
        self.notes = []
        if cfg.exposure is not None and self.set_exposure(cfg.exposure) is None:
            self.notes.append(f"NOTE: the camera driver ignored exposure {cfg.exposure} from config.json.")
        if cfg.gain is not None and self.set_gain(cfg.gain) is None:
            self.notes.append(f"NOTE: the camera driver ignored gain {cfg.gain} from config.json.")
        if IS_MAC and self.notes:
            self.notes.append("      macOS (AVFoundation) usually cannot set exposure or gain from OpenCV. Lock them with "
                              "a UVC control app (e.g. 'CameraController' or 'uvc-util') before running, "
                              "or the belt brightness may drift.")
        self._lock, self._latest, self._count, self._running = threading.Lock(), None, 0, True
        self.capture_fps = 0.0
        self._stamps = []
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        t_prev = time.perf_counter()
        while self._running:
            ok, img = self.cap.read()
            t = time.perf_counter()
            if not ok:
                time.sleep(0.005)
                continue
            self._stamps = (self._stamps + [t])[-40:]          # frame rate over the last 40 frames
            if len(self._stamps) >= 5:
                self.capture_fps = (len(self._stamps) - 1) / (self._stamps[-1] - self._stamps[0])
            t_prev = t
            with self._lock:
                self._count += 1
                self._latest = Frame(img, t, self._count)

    def read(self, after_index=0, timeout=1.0):
        t_end = time.perf_counter() + timeout
        while time.perf_counter() < t_end:
            with self._lock:
                f = self._latest
            if f is not None and f.index > after_index:
                return f
            time.sleep(0.0005)
        return None

    def describe(self):
        g = self.cap.get
        fcc = int(g(cv2.CAP_PROP_FOURCC))
        fcc = "".join(chr((fcc >> 8 * i) & 0xFF) for i in range(4)) if fcc > 0 else "?"
        return (f"camera {int(g(cv2.CAP_PROP_FRAME_WIDTH))}x{int(g(cv2.CAP_PROP_FRAME_HEIGHT))} {fcc}, "
                f"exposure {g(cv2.CAP_PROP_EXPOSURE):.0f}, gain {g(cv2.CAP_PROP_GAIN):.0f}, "
                f"measured {self.capture_fps:.0f} fps")

    def warm_up(self, timeout=3.0):
        """Wait until the frame rate can be measured, then say so if it is well below what was asked."""
        t_end = time.perf_counter() + timeout
        while time.perf_counter() < t_end and len(self._stamps) < 30:
            time.sleep(0.05)
        notes = list(self.notes)
        if self.capture_fps < 0.6 * self.requested_fps:
            notes.append(f"NOTE: the camera is delivering {self.capture_fps:.0f} frames/s, not the {self.requested_fps} "
                         f"asked for. " + ("Shorter exposure helps (press ] / [ in setup); so does closing other camera apps."
                                           if not IS_MAC else "Close other camera apps (FaceTime, Zoom, Photo Booth)."))
            if IS_MAC:
                notes.append("      On macOS OpenCV cannot ask for MJPG, so high frame rates at full size may not be "
                             "available. Try a smaller --fps or a lower width/height in config.json.")
        return "\n".join(notes)

    def open_settings(self):
        """The driver's settings dialog. Only DirectShow (Windows) has one. Returns False if unavailable."""
        if not IS_WIN:
            return False
        return bool(self.cap.set(cv2.CAP_PROP_SETTINGS, 1))

    def set_exposure(self, value):
        """DirectShow units: log2(seconds). -7 ~ 8 ms, -8 ~ 4 ms, -9 ~ 2 ms, -10 ~ 1 ms.
        Returns the value the driver reports, or None if the driver refused it (usual on macOS)."""
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        if not self.cap.set(cv2.CAP_PROP_EXPOSURE, float(value)):
            return None
        return self.cap.get(cv2.CAP_PROP_EXPOSURE)

    def auto_exposure(self):
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)

    def set_gain(self, value):
        """Returns the value the driver reports, or None if the driver refused it (usual on macOS)."""
        if not self.cap.set(cv2.CAP_PROP_GAIN, float(value)):
            return None
        return self.cap.get(cv2.CAP_PROP_GAIN)

    def release(self):
        self._running = False
        time.sleep(0.05)
        self.cap.release()


class FileSource:
    """Video file, image folder or single image. Time stamps come from <file>.csv if it exists
    (written by `record`), otherwise from the frame rate."""
    live = False

    def __init__(self, path, fps=None):
        self.images, self.cap, self.times = [], None, None
        if os.path.isdir(path):
            self.images = sorted(p for p in glob.glob(os.path.join(path, "*")) if p.lower().endswith(IMG_EXT))
            if not self.images:
                sys.exit(f"No images in {path}")
            self.fps = fps or 30.0
        elif path.lower().endswith(IMG_EXT):
            self.images, self.fps = [path], fps or 30.0
        else:
            self.cap = cv2.VideoCapture(path)
            if not self.cap.isOpened():
                sys.exit(f"Could not open {path}")
            self.fps = fps or self.cap.get(cv2.CAP_PROP_FPS) or 30.0
            if os.path.exists(path + ".csv"):
                with open(path + ".csv") as fh:
                    self.times = [float(r[1]) for r in list(csv.reader(fh))[1:]]
        self.index, self.capture_fps = 0, self.fps

    def read(self, after_index=0, timeout=None):
        if self.cap is not None:
            ok, img = self.cap.read()
            if not ok:
                return None
        else:
            if self.index >= len(self.images):
                return None
            img = cv2.imread(self.images[self.index])
        t = self.times[self.index] if self.times and self.index < len(self.times) else self.index / self.fps
        self.index += 1
        return Frame(img, t, self.index)

    def describe(self):
        return f"file source, {self.fps:.0f} fps"

    def release(self):
        if self.cap is not None:
            self.cap.release()


class Recorder:
    """Saves raw frames to an .avi plus a .csv of time stamps, for replay and regression tests."""

    def __init__(self, path, shape, fps):
        self.path = path
        self.vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), fps, (shape[1], shape[0]), True)
        self.fh = open(path + ".csv", "w", newline="")
        self.w = csv.writer(self.fh)
        self.w.writerow(["frame", "t"])
        self.n, self.t0 = 0, None

    def write(self, image, t):
        self.t0 = t if self.t0 is None else self.t0
        self.vw.write(image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))
        self.w.writerow([self.n, f"{t - self.t0:.5f}"])
        self.n += 1

    def close(self):
        self.vw.release()
        self.fh.close()


def list_cameras(max_idx=8):
    found = 0
    for i in range(max_idx):
        cap = cv2.VideoCapture(i, _backend())
        opened = cap.isOpened()
        if opened:
            found += 1
            ok, frame = cap.read()
            print(f"  [{i}] " + (f"{frame.shape[1]}x{frame.shape[0]}" if ok else "opened, no frame"))
        cap.release()
        if IS_MAC and not opened:
            break                      # AVFoundation numbers cameras 0..n-1 with no gaps
    if not found:
        print("  no cameras found")
    if IS_MAC:
        print("  macOS: the built-in FaceTime camera and an iPhone (Continuity Camera) also appear here, so the\n"
              "  USB camera's number is often not the same as on Windows. If nothing opens, give your terminal\n"
              "  app camera access in System Settings > Privacy & Security > Camera.")
