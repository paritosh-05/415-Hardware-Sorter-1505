# Arduino Uno pinout (collector and hood lights)

Taken from `collector/collector.ino` as it is in this branch (24 November 2026). One Uno does both jobs: it
turns the pie and holds the two hood LED panels at full white. If a pin changes in the sketch, change it here.

## Pins the sketch uses

| Uno pin | Name in the sketch | Connects to | Notes |
|---|---|---|---|
| D0 (RX), D1 (TX) | `Serial`, 115200 baud | USB to the laptop | Leave unconnected. Anything wired here breaks the laptop link and uploads. |
| D2 | `PUL_PIN` | TB6600 **PUL+** | Step pulses, 5 us high / 5 us low. |
| D3 | `DIR_PIN` | TB6600 **DIR+** | Direction. Pie turns the wrong way: set `INVERT_DIR = true` (do not swap wires). |
| D7 | `LED_PIN_A` | LED panel A, **Din** | 14 LEDs, `NEO_GRB + NEO_KHZ800`. |
| D8 | `LED_PIN_B` | LED panel B, **Din** | 14 LEDs, same. |
| GND | | TB6600 **PUL-** and **DIR-**, and the LED supply's GND | All grounds joined. |

Not used by the sketch, so free: D4, D5, D6, D9 to D13, A0 to A5. D13 also drives the on-board LED.
`collector_req.md` says to keep D9 and D10 free because Timer1 does the stepping; the sketch as it stands
times its steps with `micros()` and does not use Timer1, so that only matters if the stepping is changed.

No home sensor is wired or read yet (the sketch answers `H` with `READY`). It has no pin assigned.

## Stepper driver (TB6600)

| TB6600 terminal | Goes to |
|---|---|
| PUL+ | Uno D2 |
| PUL- | Uno GND |
| DIR+ | Uno D3 |
| DIR- | Uno GND |
| ENA+, ENA- | not connected (driver always enabled) |
| A+, A-, B+, B- | the NEMA 23's two coils |
| VCC, GND | its own DC supply, 9 to 42 V (not from the Uno) |

DIP switches: **16 microsteps** (must match `MICROSTEP` in the sketch), current to suit the motor.
The sketch assumes 200 steps/rev, a 1:4 belt and 8 slices: 1600 microsteps per slice.

## Hood LED panels

| Strip pad | Goes to |
|---|---|
| +5V | separate **5 V** supply, 2 A or more (not the Uno's 5V pin: 28 LEDs at full white draw about 1.4 to 1.7 A) |
| GND | that supply's GND **and** Uno GND |
| Din | Uno D7 (panel A) or D8 (panel B) |
| DO | not connected |

- The strips on the rig are **WS2812E, 5 V. Never 12 V.** (`collector_req.md` calls them WS2812B; the wiring
  and the sketch are the same for both.)
- `collector_req.md` asks for a 330 ohm resistor in series with each data line and a 1000 uF capacitor across
  +5V / GND at the strip. On the rig as built by Paritosh the data lines are wired **direct**: the resistor
  he had turned out to be 330 kilohm, which lit only the first LED. It works without.
- The LEDs are written once at power-up (full white, `LED_LEVEL` 255) and never again, so they stay on while
  the Uno has power and do not disturb the stepping.

## Power and USB

- Uno: powered from the laptop's USB. Port: `COM5` on Paritosh's laptop (`python sorter.py cams` lists it; ignore the Bluetooth COM ports).
- Three supplies in all: USB (Uno), 5 V (LEDs), 9 to 42 V (TB6600). Their grounds must be joined.
- Opening the serial port resets the Uno. The sketch keeps the pie's position in EEPROM so it carries on from
  the slice it was at.

## Check it

Arduino Serial Monitor, 115200 baud, line ending "Newline" (close it again before running the sorter):

1. After a reset it prints `POS,<n>` and `READY`, and both panels go white.
2. `G,2`: the pie turns so slice 2 catches, waits 3 s, goes back to 7 (Rework).
3. `P`: prints the slice it thinks is catching. `Z,<n>`: tells it slice n is catching, after moving the pie by hand.

Then: `python sorter.py run --port COM5`

Not verified against the physical wiring from here: this is what the code expects. More detail and the
reasons behind the choices are in `collector_req.md`.
