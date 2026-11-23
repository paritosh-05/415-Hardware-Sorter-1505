COLLECTOR PROGRAM REQUIREMENTS

The collector is a cylinder with 8 radial bins, like a pizza. It works by rotating the correct slice to the conveyor to catch the hardware. After identification, there is a timed delay, before the collector rotates to the corresponding slice. 

Bin mapping, clockwise:
Slice No.   Category
1. 1/4in bolt
2. 1/4in nut
3. 1/4in washer
4. 5/16in bolt
5. 5/16in nut
6. 5/16in washer
7. Rework
8. Misc

The collector system is controlled by an Arduino Uno. The Uno needs to do the following:
1. Collection
- Read identified target from the computer over serial (115200)
- Rotate to that target. The Arduino is connected to a TB6600 Stepper driver. Example code for movements is shown in stepper_serial_moves.ino, use that as reference for slice movement, and ask any questions if needed.
- Wait for next target
2. Backlighting
- The Arduino is also a lighting controller, it sends full-white signal to 28 WD2812b LEDs that light up the camera capture area.
- Currently, this occupies pin 7 & 8 for 14 LEDs each. This can be moved around if needed, should be noted in the pinout.

Early build:
For now, there is no stepper calibration/zeroing. Assume that on startup, the pie is aligned so that slice 1 is catching.


DECISIONS AND DETAILS (added with the first version of collector/collector.ino)

Drive
- NEMA 23 (200 steps/rev) on a TB6600 at 16 microsteps, 1:4 timing belt (small pulley on the motor).
- One pie revolution = 4 motor revolutions = 12800 microsteps. One slice (45 deg of pie) = 1600 microsteps
  = 180 deg of motor, the same move stepper_serial_moves.ino was tested with.
- The pie always turns the shorter way round, so the longest move is 4 slices.
- Every move takes MOVE_TIME_MS = 300 ms, whether 1 slice or 4 (runMove() of stepper_serial_moves.ino,
  25% ramps, TIME_SCALES_WITH_LENGTH = false). Parts fall off the belt one after another, so two parts for
  different slices that land less than min_gap_ms apart both go to Rework (config.json: min_gap_ms 300,
  step_ms 0). A 4-slice move in 300 ms is 720 deg of motor (about 530 rpm at the peak): check on the bench
  that it does not skip steps. See collector_open_questions.md for the margin.
- Direction: "forward" means the catching slice goes from 1 to 2. If it turns the wrong way, flip INVERT_DIR.

Timing (who decides when)
- The laptop sends Q,<id>,<slice>,<ms> as soon as a part is identified: which slice and how many ms until it
  has landed (PROTOCOL.md). The Arduino does the timing: it keeps a queue sorted by landing time, stands at
  the slice of the earliest part that has not landed, and turns to the next one once that time has passed.
- Slices are numbered 1-8 in the protocol, the same as the bin mapping above.
- Parts the vision system is not sure about go to slice 7 (Rework). Slice 8 (Misc) is not used yet.
- With nothing queued the pie goes back to slice 7, so anything the camera missed lands in Rework.
  (Under review: see collector_open_questions.md.)
  On power-up it first turns from slice 1 to slice 7.
- A part is announced when it crosses the yellow DECIDE line, about 2.9 s before it lands at 94 mm/s.
  That must stay longer than two moves (0.6 s: back to Rework and out again), which it easily does.

Start-up position
- The Uno resets every time the laptop opens the serial port. So the sketch saves the current slice in
  EEPROM after every move (and a "turning" marker before each move), and after a reset it carries on from
  the saved slice instead of assuming slice 1.
- Slice 1 is assumed only on the very first start, or after a reset in the middle of a move. In the second
  case it prints a warning: turn the pie by hand so slice 1 is catching, then send Z,1.
- Z,<n> in the Serial Monitor tells it slice n is catching now, whenever the pie has been moved by hand.
  Set REMEMBER_POSITION to false to always assume slice 1 at start-up, as in the original requirement.
- The EEPROM cell is written about 4 times per sorted part. It is good for about 100,000 writes, so about
  25,000 parts. Fine for this build; the home sensor in a later build replaces it.

Backlighting
- Both strips are set to full white (LED_LEVEL 255) once at start-up and never written again. Writing
  WS2812B LEDs stops interrupts for about 0.4 ms per strip, which would lose serial bytes and upset the
  stepping. To change the brightness, change LED_LEVEL and upload again.
- 28 LEDs at full white draw about 1.7 A at 5 V. That is more than the Uno's 5V pin can give: power the
  strips from a separate 5 V supply of at least 2 A, with its ground joined to the Uno's ground.


PINOUT (Arduino Uno)

| Uno pin | Connects to                  | Notes                                                        |
|---------|------------------------------|--------------------------------------------------------------|
| D0, D1  | USB serial                   | Used by the laptop link. Leave unconnected.                  |
| D2      | TB6600 PUL+                  | Step pulses (5 us). PUL- to GND.                             |
| D3      | TB6600 DIR+                  | Direction. DIR- to GND.                                      |
| D7      | LED strip A, data in (DIN)   | 14 x WS2812B. 330 ohm resistor in series, close to the strip.|
| D8      | LED strip B, data in (DIN)   | 14 x WS2812B. 330 ohm resistor in series, close to the strip.|
| GND     | TB6600 PUL-, DIR-; LED supply GND | All grounds joined.                                     |
| D9, D10 | (keep free)                  | Timer1 is used for stepping, so PWM on these pins won't work.|

TB6600 ENA+/ENA- are left unconnected (driver always enabled). TB6600 power: its own 9-42 V DC supply.
LED strips: +5V and GND from the separate 5 V supply, with a 1000 uF capacitor across +5V/GND at the strip.
DIP switches on the TB6600: 16 microsteps (must match MICROSTEP in the sketch), current to suit the motor.


TESTING WITHOUT THE LAPTOP PROGRAM (Arduino Serial Monitor, 115200 baud, line ending "Newline")

1. After upload it prints POS,1 and READY, the LEDs come on, and the pie turns from slice 1 to slice 7.
2. G,2 : the pie should turn so slice 2 catches, wait 3 s, then go back to 7. Wrong way? Flip INVERT_DIR.
3. G,3 then G,6 etc.: each slice should line up exactly. If it drifts, check MICROSTEP vs the DIP switches
   and GEAR_RATIO.
4. Q,1,3,2000 then quickly Q,2,5,3000 : turns to 3, stays until 2 s, turns to 5, stays until 3 s, then 7.
   It answers OK,1 OK,2 and later D,1,3 and D,2,5.
5. Press the reset button and send P: it should print the slice it is actually on.
