# Collector: open questions

Status, 22 November 2026. Bench testing of `collector/collector.ino` started. Pick this up next time.

## 1. When should the pie turn towards a part? (blocking: needs a code change)

**What happens now:** when a part is identified, the laptop sends `Q,<id>,<slice>,<ms>`, where `<ms>` is the
time until the part *lands* (from `drop_distance_mm`, `fall_ms` and the belt speed). The Arduino turns to
the part's slice **as soon as it can**: straight away if nothing is queued, otherwise the moment the part
before it has landed. It then waits there. `<ms>` only decides when it may turn *away*. On the bench this
looks like "the part passes and the pie turns almost immediately".

**What's wanted:** the part passes the camera, the pie **waits a tunable delay (e.g. 1000 ms)**, then turns,
and **holds** that position until the next part needs a turn. No setting does this today.

Options for the delay:

- **(a) Fixed delay after identification:** `TURN_DELAY_MS = 1000` in the sketch, counted from when the `Q`
  line arrives. Simple and matches the original requirement ("after identification, there is a timed
  delay"), but it does not follow changes in belt speed.
- **(b) From the predicted landing:** turn at the last moment before the part arrives; the setting is "how
  early before landing" (e.g. 100 ms). It follows the belt speed automatically.

Either way the pie must never turn away before the previous part's predicted landing, so a quick follow-up
part cannot make it drop the one ahead.

**Also to decide:** after a part lands, should the pie stay put ("holds this position"), or go back to
Rework (7) as it does now? Holding means a part the camera missed lands in whatever slice was used last;
returning to Rework catches it there.

## 2. Gap between parts: 300 ms or 400 ms

Every move takes 300 ms (1 to 4 slices). Two parts for different slices landing less than `min_gap_ms`
apart both go to Rework. `config.json` has **300**, the rule as stated.

On the simulated belt (rig settings: 8 slices, 300 ms moves) 300 ms left no margin: 1 part landed while the
pie was turning (76 ms after its predicted window ended). With **400** (300 ms move + 100 ms spare, the same
spare the original design had: 600 for a 500 ms collector) no part landed mid-turn, and the closest came
14.6 ms clear. The cost: more parts go to Rework when parts are close together (31 sorted vs 47 in that test).
Recommendation: 400.

## 3. Unfinished on the laptop side

- `selftest.py`'s "landing-time prediction" pass mark still assumes a 500 ms collector
  (`0.25 * (min_gap_ms - 500)`). It is fine with the default settings, but fails with the rig's 300 ms
  collector. It should use the actual collector move time.
- `fall_ms` (150) and `drop_distance_mm` (270) are still placeholders: measure them on the rig.

## Things that are done

- `collector.ino`: protocol queue, 300 ms moves using the example's `runMove()`, EEPROM position across
  resets, test commands `G` / `Z` / `P`, LEDs on pins 7 and 8. Pinout and bench tests: `collector_req.md`.
- Only checked off the Uno (compiled on the Mac with stand-ins for the Arduino parts): step counts and
  300 ms moves for 1 to 4 slices. **Not yet compiled with the Arduino IDE for the Uno** at the time of writing.
