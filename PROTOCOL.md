# Collector controller protocol

Text lines over USB serial, 115200 baud, each ended by a newline (`\n`). The laptop decides *what* and *when*;
the controller does the motion and the timing.

## Laptop to controller

| Line | Meaning |
|---|---|
| `Q,<id>,<slot>,<ms>` | Part `<id>` belongs in `<slot>` (slice 1 to 8) and will have landed within `<ms>` milliseconds from now: be at the slot and stay until then. If `<id>` is already queued, this replaces it (the slot can change to re-sort when another part turns up too close). |
| `H` | Go home and re-zero on the home sensor. Sent once when the laptop connects. Early build (no home sensor): just answers `READY`. |
| `S` | Stop: clear the queue (the pie then goes back to Rework). Sent when the laptop program exits. |
| `A` | Alive. Sent about once a second while the sorter program is running (not while it is paused). No answer. Only used for the status light. |

## Controller to laptop

| Line | Meaning |
|---|---|
| `OK,<id>` | Entry accepted (optional). |
| `D,<id>,<slot>` | The landing time of `<id>` passed while the pie was standing at `<slot>`, or `0` if it was still turning (optional, useful for logs). |
| `POS,<slot>` | Current slice (at start-up, and in answer to `P` / `Z`). |
| `?,<text>` | A line it did not understand, e.g. `?,BAD Q`. Not an error for the sorter. |
| `E,<text>` | Something is wrong, for example `E,HOME` when the home sensor did not appear where expected. The laptop shows it and sends every part to re-sort until `READY`. |
| `READY` | Homed and ready again. |

## What the controller must do

1. On `Q`: convert `<ms>` to its own clock (`millis() + ms`) and store `(id, slot, time)`, keeping entries
   sorted by time. Replace the entry if the id exists.
2. Always stand at the slot of the **earliest** entry that has not landed yet. When that entry's time has
   passed, remove it and turn to the next entry's slot straight away, by the shorter direction.
3. With an empty queue, go to slice 7 (Rework), so anything that was not announced lands there.
4. (Later build, with a home sensor.) Every time the pie passes the home position, compare the step count with the home sensor. If they
   disagree, stop, send `E,HOME`, re-home, then send `READY`.

The laptop guarantees that, for two entries in a row with different slots, the second part cannot land
earlier than the collector needs to turn between them (`settle_ms + slices * step_ms`) after the first entry's time, except when a part turns up too
late to plan for. It counts those as "missed".

Slices, as on the pie (`collector_req.md`, and `slots` / `resort_slot` in `config.json`): 1 bolt 1/4, 2 nut 1/4,
3 washer 1/4, 4 bolt 5/16, 5 nut 5/16, 6 washer 5/16, 7 Rework (everything the sorter is not sure about),
8 Misc (not used yet). The software's labels `bolt-0.375` etc. are the 5/16" parts.

Every move of `collector/collector.ino` takes 300 ms, whether 1 slice or 4. The laptop uses `min_gap_ms` in
`config.json` (with `step_ms` 0) to keep parts for different slices far enough apart.

## Status light

Green LED on D11 (`collector/PINOUT.md`). Fast blink: starting up, about half a second after power-up or a
reset. Slow blink (1 s cycle): ready, waiting for the sorter program. Solid: the sorter program is connected and
running, meaning the controller has received `H`, `Q` or `A` within the last 3 seconds and no `S` since.
It drops back to slow blink within 3 s if the program is paused, freezes, exits, or the cable is pulled.
"Running" means the laptop program's frame loop is turning. It does not prove the camera sees the belt or
that parts are being sorted; the laptop window shows that. Commands typed in the Serial Monitor (`G`, `Z`,
`P`) do not turn it solid, `H` or `A` do.

## Test without the laptop program

Arduino Serial Monitor at 115200 baud, line ending "Newline". `Q,1,3,2000` then `Q,2,5,3000` should turn to
slice 3, wait until 2 s has passed, turn to slice 5, then go back to 7. Test-only commands:

| Line | Meaning |
|---|---|
| `G,<slot>` | Clear the queue and go to `<slot>` now; back to Rework after 3 s. |
| `Z,<slot>` | "This slice is catching now": set the position without moving (after turning the pie by hand). |
| `P` | Print the position. |

The full test list is at the end of `collector_req.md`.
