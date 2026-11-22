# Hardware sorter vision, v2

Team 1505, MIE 415. Camera-based sorting for the automatic hardware sorter.

Scope: 1/4" and 3/8" hex bolts, nuts and washers (six parts), plus a re-sort slot. Seven slots on the pie.
The labels are just names: `bolt-0.25 nut-0.25 washer-0.25 bolt-0.375 nut-0.375 washer-0.375`.

Design rule: **a re-sort is cheap, a wrong bin is expensive.** Every stage can say "not sure", and anything
unsure goes to re-sort.

## How it decides

1. **Find and measure.** Each frame is divided by a picture of the empty belt, so lighting and exposure cancel
   out. A pixel belongs to a part if it is clearly darker or brighter than the belt, or sits on a sharp edge.
   Soft shadows are ignored. Outlines are measured in mm.
2. **Track.** Each part is followed across the frame. It is measured only in the middle of the view. Its speed
   and the time it will land are worked out from its own track and the belt speed.
3. **Two independent opinions.**
   - *Geometric model*: length, width, area, hole size, solidity and shank width, compared with what each part
     measured when it was taught on this rig.
   - *Appearance model*: how the part looks inside and around its outline (brightness in rings around the
     centre and along its length), compared with the taught examples by nearest neighbour.
4. **Agreement rule.** A part is sorted only if both models are confident and name the same part. Anything
   else goes to re-sort. `allow_single_certain` (off) would let one certain model carry a part when the other
   is merely unsure about the same part; those cases are counted in the log first so you can see if it is safe.
5. **Gap rule.** Two parts in a row that need different slots must land at least `min_gap_ms` apart. If not,
   both go to re-sort, and the check ripples back through the queue. Two close parts for the same slot are
   both sorted.
   - Each part has a landing *window*, not an instant. A long part, a clump, or a part that is not moving
     with the belt gets a wider window. The gap is measured from the end of one window to the start of the
     next, and the collector is told to stay until the window ends.
   - A part can only be changed until the collector is committed to it. If a chain of close parts reaches
     back to a committed part, that is logged as **missed**: the collector may still be turning when the
     next part lands. Keep the time from the DECIDE line to the drop at about three times `min_gap_ms`
     or more (1.8 s for 600 ms, which is 150 mm/s with 270 mm of belt). The run summary warns if it is less.

Straight to re-sort without asking the models: touching parts, a part that appears mid-belt, a part lost from
view, a part hanging off the belt edge, a part sitting on the belt seam, a part moving more than 6% faster or
slower than the belt (rolling or sliding), too few clear frames, and anything seen while a health check is
failing (lighting changed, view blocked, camera stalled).

### The belt seam

The stitched seam comes past once per belt revolution. It reaches from one belt edge to the other, which no
single part can, so it is followed separately from the parts.

- Seam alone: ignored. It is not counted as a part and does not move the collector.
- Seam that looks bigger than usual: a part is sitting on it, so that stretch of belt goes to re-sort
  ("on the belt seam").
- The first two passes after start-up go to re-sort while the normal size of the seam is learned. Wait about
  two belt revolutions before feeding.

Leave the seam bare. Covering it with tape made it worse: the software could neither recognise it nor ignore
it. A small part lying right on the stitching adds little to the seam's size and could go unnoticed; set
`"seam_ignore": false` in `config.json` to send the seam's stretch of belt to re-sort on every pass instead.
The empty-belt background (`b` in setup) must be taken with the seam out of view.

## Install

```
pip install opencv-python numpy pyserial
python sorter.py selftest
```
The self-test runs a simulated belt for about two minutes and should end with `ALL CHECKS PASSED`.

## macOS

Everything except the camera controls works the same as on Windows. The self-test passes on an Apple-silicon Mac.

```
python3 -m venv .venv
source .venv/bin/activate          # do this in every new terminal
pip install -r requirements.txt
python sorter.py selftest
```

Differences from Windows:

- **Camera permission.** The first time, macOS asks whether your terminal app (Terminal, iTerm, VS Code) may use
  the camera. If you refused, or nothing opens: System Settings > Privacy & Security > Camera, allow it, then
  quit and reopen the terminal app.
- **Camera numbers change.** The built-in camera and an iPhone (Continuity Camera) also count, so the USB
  camera is probably not `1` as on Windows. Run `python sorter.py cams` and use `--camera N`, then `w` in setup.
- **No exposure, gain or MJPG control.** OpenCV's macOS driver (AVFoundation) ignores these settings, so
  `[ ] , . a` and `k` in setup say "refused" and do nothing. The `exposure`/`gain` values in `config.json` are
  Windows (DirectShow) values and are ignored here. Lock exposure and gain *before* starting the sorter with a
  UVC control app, e.g. CameraController (free, App Store / GitHub) or `uvc-util`. Then grab the background
  (`b`). Auto exposure lets the belt brightness drift, and the health check sends parts to re-sort when it does.
- **Frame rate.** Without MJPG the camera may not reach 1280x800 at 120 fps. The start-up line prints the
  real rate. If it is low, the belt must run slower (see the exposure rule above), or lower `width`/`height`
  in `config.json`. **px/mm and teaching depend on the resolution**: if you change it, redo `c` and teach again.
- **Serial port.** `python sorter.py cams` also lists serial ports. The collector shows up as
  `/dev/cu.usbmodem…` or `/dev/cu.usbserial…`. Use the `cu.` name, not `tty.`:
  `python sorter.py run --port /dev/cu.usbmodem1101`. Your own Bluetooth devices also appear in that list.
- Windows uses `python`; on macOS it is `python3`, or plain `python` while the venv is active.

## On the rig

Do these in order, from this folder. Close OBS first.

1. **Find the camera:** `python sorter.py cams`, then put the number in `--camera N` (saved by `w`).
2. **Setup:** `python sorter.py setup --camera N`
   - Lock the exposure with `[` and `]`, and the gain with `,` and `.`. Aim for a bright belt that is not
     pure white. For sharp moving parts the exposure should be about `0.2 mm / belt speed` or shorter:
     2 ms at 100 mm/s, which is exposure `-9`. If that is too dark, add light before adding gain.
   - With the belt running and **empty**, press `b`. It saves the background, finds the belt lane, and checks
     belt noise.
   - Put a quarter flat on the belt and press `c` to set px/mm. No quarter: use any flat round object,
     measure it across with calipers, and start setup with its size, e.g. `python sorter.py setup --ref-mm 18.6`.
   - Press `f` if the belt runs right to left. Press `w` to save.
   - Optional: press `l` to draw the lane by hand. Keep it the full width of the picture and about 1 mm
     inside the belt edges, never onto the rails.
3. **Teach each part:** `python sorter.py teach nut-0.25` and feed only that part, 30 or more passes, in every
   way it can lie (nut flat and on its side, bolt in any rotation). Repeat for all six labels:
   `bolt-0.25 nut-0.25 washer-0.25 bolt-0.375 nut-0.375 washer-0.375`.
   Keep the belt at the speed you will sort at. Teaching again adds to the examples already saved; to start
   a part over, delete its folder first (`teach/nut-0.25`).
4. **Build:** `python sorter.py build`. Read the report: every part should re-check as "certain", and the
   closest pairs should be above 5.
5. **Test.** No collector is needed for any of these. Wrong bin must be zero.
   - *Known order:* `python sorter.py check`. It lists 20 parts in a mixed order and prints which one to
     feed next. At the end it prints a table: what you fed, what the sorter said, and the reason for any
     re-sort. Options: `check 30`, `--seed 4` for a different order, `--order "nut-0.25, bolt-0.375"` or a
     text file for your own. A missed part or an extra detection does not shift the rest of the table.
   - *Blind:* `python sorter.py run` and feed anything. Check the answers afterwards on `sheet.png`.
   - *One part type:* `python sorter.py run --truth nut-0.375` counts correct bin, re-sort and WRONG BIN.
6. **Sort:** `python sorter.py run --port COM5` (macOS: `--port /dev/cu.usbmodem…`).

If every part is rejected, read the terminal: it prints `[advice]` when parts are moving against the
configured belt direction (press `f` in setup, then `w`), and when the camera is too slow for the belt speed.
The camera line at start-up shows the real frame rate; it should be near 120 in MJPG.

Re-do step 2 (`b`) if the lights, exposure or camera position change. Re-do teaching if the camera height or
focus changes.

## Numbers to measure on the rig (config.json)

| Setting | Meaning | How to get it |
|---|---|---|
| `drop_distance_mm` | from the yellow DECIDE line to where parts leave the belt | tape measure (about 270 now) |
| `fall_ms` | belt end to landing in the slot | slow-motion phone video |
| `min_gap_ms` | collector's worst move plus settle | from the collector test (0.5 s move -> 600) |
| `extent_factor` | how much of a part's length counts as landing-time uncertainty (0.5) | compare slow-motion video of bolts dropping with `t_drop` in the log |
| `slots` | which slot each label goes to; `resort_slot` | agree with the collector layout |
| `step_ms`, `settle_ms` | optional: time per slot step, so neighbours need less gap | leave `step_ms` at 0 to start |

Other settings worth knowing: `zone` (where parts are measured), `slip_tol` (rolling detection),
`dark_ratio`, `bright_ratio`, `edge_grad` (segmentation), `geo_gate`, `geo_rival` (how strict the geometric
model is), `seam_ignore`, `seam_extra_frac`, `seam_extra_mm2` (belt seam). A setting that is not in
`config.json` uses the default in `config.py`.

## Logs

Each `run` writes `runs/<time>/`:
- `parts.csv`: one line per part with its measurements, both models' answers and confidence, the decision,
  the reason for any re-sort, the final slot, and `missed` (collector may not have been in place).
- `crops/`: a picture of every part, named with its decision.
- `sheet.png`: every part in the order it passed, at one scale, with the sorter's answer under it.
- `summary.txt`: counts, re-sort reasons, belt speed.
- `check.txt`: the scored table, when the run was a `check`.

`python sorter.py record clips/name.avi` saves raw footage with time stamps. Any mode replays it with
`--source clips/name.avi`, so you can re-test after a change without the rig.

## Files

| File | Job |
|---|---|
| `sorter.py` | command line and windows |
| `config.py` | all settings |
| `camera.py` | camera thread, file replay, recorder |
| `segment.py` | finding parts, measurements, appearance descriptor |
| `models.py` | geometric model, appearance model, agreement rule |
| `tracker.py` | tracks, speed, rolling check, belt seam |
| `check.py` | the known-order test: feeding order, lining up, scoring |
| `scheduler.py` | gap rule and re-sort ripple |
| `collector.py` | serial link and a simulated collector |
| `pipeline.py` | ties it together, health checks, logging, overlay |
| `sim.py`, `selftest.py`, `ref/` | simulated belt built from the six reference photos |
| `unit_tests.py` | quick checks of the gap rule, agreement rule and geometric model (`python unit_tests.py`) |
| `PROTOCOL.md` | messages for the collector controller |

## What has and has not been tested

**On the rig (November 2026), camera only, parts fed by hand, belt at about 94 mm/s, camera at 120 frames/s:**

- All six parts taught, 31 to 48 passes each. In the build report the washers are far from everything else;
  the two bolt sizes (5.1) and the two nut sizes (4.5) are the close pairs.
- Three mixed runs of 20 parts each. Every part that was sorted went to the correct bin, about 50 in total.
  15 to 20% of parts went to re-sort per run.
- Most re-sorts were the small nut lying on its side, which needs more taught examples. The others were
  parts at the belt edge, washers still catching up to belt speed, and parts next to the seam before the
  seam handling was added.

This is a small sample, enough to show it works and not enough to quote an accuracy figure.

The hardware labelled 3/8" measures closer to 5/16"-18 (bolt shank about 7.5 mm, nut about 13 mm across
flats). This has not been checked with calipers. It does not affect sorting.

**On a simulated belt** made from the six reference photos (40 to 200 mm/s): teaching, mixed sorting, the
gap rule against a simulated 500 ms collector, stacked / touching / foreign / dropped-in / rolling parts, a
lighting change, and the known-order check with deliberate feeding faults. No part reached a wrong bin. The
seam handling was tested by replaying photos of the real seam as a moving belt.

**Not tested:** anything with the collector connected. `drop_distance_mm`, `fall_ms`, `min_gap_ms` and
`extent_factor` in `config.json` are still placeholders. If they are wrong, a correctly identified part can
still land in the wrong bin. Also untested: parts arriving from the feeder instead of by hand.
