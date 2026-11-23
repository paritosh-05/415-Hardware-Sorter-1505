"""All settings for the sorter, in one place. Saved to / loaded from config.json."""
import json
import os
from dataclasses import asdict, dataclass, field

QUARTER_MM = 24.26   # US quarter, used to calibrate px/mm
RESORT = "resort"


@dataclass
class Config:
    # ---- camera (None = leave the driver's value alone) -------------------------------------
    camera: int = 0
    width: int = 1280
    height: int = 800
    fps: int = 120
    mjpg: bool = True
    exposure: float = None            # DirectShow log2(seconds): -7 ~ 8 ms, -8 ~ 4 ms, -9 ~ 2 ms
    gain: float = None

    # ---- geometry of the view ---------------------------------------------------------------
    px_per_mm: float = 14.3           # full-resolution pixels per mm at belt level ('c' in setup)
    lane: list = field(default_factory=lambda: [0, 0, 0, 0])   # x0,y0,x1,y1 px; zeros = whole frame
    flow_left_to_right: bool = True   # belt direction in the image

    # ---- segmentation -----------------------------------------------------------------------
    detect_scale: float = 0.5         # work on a downscaled copy (0.5 -> about 7 px/mm)
    blur_mm: float = 0.25
    dark_ratio: float = 0.75          # pixel belongs to a part if brightness/belt is below this...
    bright_ratio: float = 1.22        # ...or above this (glints). Soft shadows sit at 0.75-0.9.
    edge_grad: float = 0.02           # ratio change per pixel that counts as a sharp edge (0 = off).
                                      # Empty belt measured 0.008 at worst; setup re-measures it ('b').
    open_mm: float = 0.3              # removes specks
    close_mm: float = 1.6             # bridges thread valleys and glint gaps (>= coarsest thread pitch)
    min_area_mm2: float = 25.0        # smaller = dust
    max_area_mm2: float = 1500.0      # bigger = pile of parts
    seam_ignore: bool = True          # False = send the seam's stretch of belt to re-sort on every pass (safest)
    seam_extra_frac: float = 0.06     # seam looks this much bigger than usual (+ seam_extra_mm2) = a part is on it
    seam_extra_mm2: float = 10.0
    seam_min_lane_mm: float = 36.0    # on a lane at least this wide, anything reaching edge to edge is the belt seam
    hole_min_mm: float = 2.0          # smallest hole (equivalent diameter) that counts as a hole
    bg_kernel_mm: float = 45.0        # only used when no empty-belt background has been grabbed
    bg_min_level: int = 40            # background darker than this is not belt (hood floor, etc.)

    # ---- measuring and tracking -------------------------------------------------------------
    zone: list = field(default_factory=lambda: [0.22, 0.78])   # measure only in this part of the lane
    min_frames: int = 5               # fewer clean frames in the zone -> re-sort
    handoff_frac: float = 0.88        # decision line, as a fraction of lane length
    max_speed_mm_s: float = 400.0
    init_speed_mm_s: float = 60.0     # first guess before any part has been tracked
    match_gate_mm: float = 5.0
    lost_after_s: float = 0.20
    slip_tol: float = 0.06            # a part this much faster/slower than the belt is rolling or sliding
    follow_belt: bool = False         # True = the belt does not hold a steady speed. Belt speed is then taken from
                                      # the parts in view right now (not the median of the last few), each queued
                                      # part's landing time keeps being corrected and re-sent as the speed changes,
                                      # and "moving differently from the belt" only compares parts that are in
                                      # view together.
    speed_tol: float = 0.10           # follow_belt: the belt may change this much while nothing is in view to
                                      # measure it. Every landing window is widened by this share of the travel time.
    retime_ms: float = 40.0           # follow_belt: re-send a landing time once it has moved by this much

    # ---- hand-off to the collector ----------------------------------------------------------
    drop_distance_mm: float = 270.0   # from the decision line to where parts leave the belt
    fall_ms: float = 150.0            # belt end -> landing in the slot
    extent_factor: float = 0.5        # landing window = +- extent_factor * (half the part's length along the
                                      # belt) / speed. Clumps and odd tracks always use the full length.
    min_gap_ms: float = 600.0         # two parts needing different slots must be this far apart
    step_ms: float = 0.0              # optional: time per slot step. 0 = always use min_gap_ms
    settle_ms: float = 100.0          # used with step_ms: gap = settle_ms + steps * step_ms
    same_slot_ok: bool = True         # two close parts for the same slot are both sorted
    n_slots: int = 7
    resort_slot: int = 0
    slots: dict = field(default_factory=lambda: {
        "bolt-0.25": 1, "nut-0.25": 2, "washer-0.25": 3,
        "bolt-0.375": 4, "nut-0.375": 5, "washer-0.375": 6})
    port: str = None                  # serial port of the collector controller, e.g. "COM5"
    baud: int = 115200

    # ---- decision rule ----------------------------------------------------------------------
    allow_single_certain: bool = False   # one model 'certain' + the other merely unsure may sort.
                                         # Off by default: such cases are logged as 'override?' first.
    geo_gate: list = field(default_factory=lambda: [2.5, 3.5, 4.5])   # certain / confident / unsure (worst z)
    geo_rival: list = field(default_factory=lambda: [7.0, 5.0])       # rival must be at least this far (certain / confident)
    app_k: int = 5

    # ---- health -----------------------------------------------------------------------------
    belt_tol: float = 0.12            # empty-belt brightness may drift this much from the background
    max_cover: float = 0.35           # more of the lane than this "is a part" -> something is wrong
    stale_s: float = 0.5

    # -----------------------------------------------------------------------------------------
    def save(self, path="config.json"):
        with open(path, "w") as fh:
            json.dump(asdict(self), fh, indent=2)

    @classmethod
    def load(cls, path="config.json"):
        cfg = cls()
        if path and os.path.exists(path):
            with open(path) as fh:
                for k, v in json.load(fh).items():
                    if hasattr(cfg, k):
                        setattr(cfg, k, v)
        return cfg

    def lane_rect(self, shape):
        h, w = shape[:2]
        x0, y0, x1, y1 = self.lane
        if x1 <= x0 or y1 <= y0:
            return 0, 0, w, h
        return max(0, x0), max(0, y0), min(w, x1), min(h, y1)

    def lane_len_mm(self, shape):
        x0, _, x1, _ = self.lane_rect(shape)
        return (x1 - x0) / self.px_per_mm

    def slot_of(self, label):
        return self.slots.get(label, self.resort_slot)

    def gap_needed_ms(self, a, b):
        """Time the collector needs between a part for slot a and the next one for slot b."""
        if a == b:
            return 0.0
        if self.step_ms <= 0:
            return self.min_gap_ms
        d = abs(a - b) % self.n_slots
        return self.settle_ms + min(d, self.n_slots - d) * self.step_ms
