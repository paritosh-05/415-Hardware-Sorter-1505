# Collector controller protocol

Text lines over USB serial, 115200 baud, each ended by a newline (`\n`). The laptop decides *what* and *when*;
the controller does the motion and the timing.

## Laptop to controller

| Line | Meaning |
|---|---|
| `Q,<id>,<slot>,<ms>` | Part `<id>` belongs in `<slot>` (0 to 6) and will have landed within `<ms>` milliseconds from now: be at the slot and stay until then. If `<id>` is already queued, this replaces it (the slot can change to re-sort when another part turns up too close). |
| `H` | Go home and re-zero on the home sensor. Sent once when the laptop connects. |
| `S` | Stop: hold position and clear the queue. Sent when the laptop program exits. |

## Controller to laptop

| Line | Meaning |
|---|---|
| `OK,<id>` | Entry accepted (optional). |
| `D,<id>,<slot>` | The landing time of `<id>` passed while the pie was standing at `<slot>` (optional, useful for logs). |
| `E,<text>` | Something is wrong, for example `E,HOME` when the home sensor did not appear where expected. The laptop shows it and sends every part to re-sort until `READY`. |
| `READY` | Homed and ready again. |

## What the controller must do

1. On `Q`: convert `<ms>` to its own clock (`millis() + ms`) and store `(id, slot, time)`, keeping entries
   sorted by time. Replace the entry if the id exists.
2. Always stand at the slot of the **earliest** entry that has not landed yet. When that entry's time has
   passed, remove it and turn to the next entry's slot straight away, by the shorter direction.
3. With an empty queue, stay where it is.
4. Every time the pie passes the home position, compare the step count with the home sensor. If they
   disagree, stop, send `E,HOME`, re-home, then send `READY`.

The laptop guarantees that, for two entries in a row with different slots, the second part cannot land
earlier than `min_gap_ms` (600 ms by default) after the first entry's time, except when a part turns up too
late to plan for. It counts those as "missed".

Slots (default, change in `config.json`): 0 re-sort, 1 bolt 1/4, 2 nut 1/4, 3 washer 1/4, 4 bolt 3/8,
5 nut 3/8, 6 washer 3/8.

## Test without the laptop program

Type lines into a serial monitor: `Q,1,3,2000` then `Q,2,5,3000` should turn to slot 3, wait until 2 s has
passed, then turn to slot 5.
