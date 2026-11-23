// Team 1505 hardware sorter - collector controller (Arduino Uno), early build.
//
// Turns the 8-slice pie so the right slice is under the end of the conveyor when each part lands,
// and drives the backlight LEDs. Serial protocol: ../PROTOCOL.md (115200 baud, newline-terminated).
// Requirements and pinout: ../collector_req.md
//
//   laptop -> Arduino   Q,<id>,<slice>,<ms>   part <id> lands in <slice> (1-8) within <ms> ms from now
//                       H                     home. Early build: no home sensor, just answers READY
//                       S                     stop: clear the queue (the pie then goes back to Rework)
//   test commands       G,<slice>             clear the queue and go to <slice> now
//   (Serial Monitor)    Z,<slice>             "this slice is catching now": set the position, no motion
//                       P                     print the position
//   Arduino -> laptop   OK,<id> | D,<id>,<slice or 0 if turning> | READY | POS,<slice> | ?,<problem>
//
// Needs the "Adafruit NeoPixel" library (Library Manager).
//
// How it works: the queue holds (id, slice, landing time) sorted by time. The pie always heads for the
// slice of the earliest part that has not landed yet; once that time has passed the entry is removed
// and the pie turns on, by the shorter way round. With an empty queue it goes to IDLE_SLICE (Rework),
// so anything the camera did not announce lands there.
//
// Every move takes MOVE_TIME_MS (300 ms), whether it is 1 slice or 4, using the move code of
// stepper_serial_moves.ino (trapezoidal profile, TIME_SCALES_WITH_LENGTH = false). The laptop knows this
// (min_gap_ms = 300 in config.json): two parts for different slices that land less than 300 ms apart
// both go to Rework. A move blocks for its 300 ms; serial bytes arriving meanwhile wait in the 64-byte
// receive buffer and are read straight after. The LEDs are written once in setup() and never again:
// writing WS2812 LEDs blocks interrupts, which would lose serial bytes.

#include <Adafruit_NeoPixel.h>
#include <EEPROM.h>

// ---------------- Pins (see the pinout in collector_req.md) ----------------
const uint8_t PUL_PIN = 2;          // -> TB6600 PUL+
const uint8_t DIR_PIN = 3;          // -> TB6600 DIR+
const uint8_t LED_PIN_A = 7;        // -> data in of LED strip A
const uint8_t LED_PIN_B = 8;        // -> data in of LED strip B

// ---------------- Mechanics ----------------
const long MOTOR_STEPS = 200;       // 1.8 deg motor
const long MICROSTEP = 16;          // MUST match the TB6600 DIP switches
const long GEAR_RATIO = 4;          // 1:4 timing belt, small pulley on the motor
const int  N_SLICES = 8;
const long STEPS_PER_SLICE = MOTOR_STEPS * MICROSTEP * GEAR_RATIO / N_SLICES;   // 1600 (= 180 deg of motor)
const bool INVERT_DIR = false;      // flip if G,2 from slice 1 turns the pie the wrong way

// Motion, as in stepper_serial_moves.ino: every move takes MOVE_TIME_MS in total, however many slices.
// If MOVE_TIME_MS changes, change min_gap_ms in config.json to match.
const unsigned long MOVE_TIME_MS = 300;
const float ACCEL_FRACTION = 0.25;  // share of the move spent speeding up (and the same slowing down)
const unsigned int PULSE_US = 5;    // step pulse high/low width

// ---------------- Behaviour ----------------
const uint8_t START_SLICE = 1;      // slice catching at power-up when no position was saved
const uint8_t IDLE_SLICE = 7;       // Rework: where the pie waits with nothing queued
const bool REMEMBER_POSITION = true;   // keep the position in EEPROM across resets (see below)
const uint8_t LED_LEVEL = 255;      // 0-255, all three colours (full white)
const uint8_t LEDS_PER_STRIP = 14;

// The Uno resets every time the laptop opens the serial port. Without REMEMBER_POSITION it would then
// believe slice 1 is catching while the pie is really waiting at Rework. So the slice is saved after
// every move, and a "turning" marker before every move; if the Arduino restarts mid-move the position
// is unknown and it falls back to START_SLICE with a warning (re-align the pie and send Z,1).
const int EE_POS = 0;
const uint8_t EE_TURNING = 0;

// ---------------------------------------------------------------------------------------------------
Adafruit_NeoPixel stripA(LEDS_PER_STRIP, LED_PIN_A, NEO_GRB + NEO_KHZ800);
Adafruit_NeoPixel stripB(LEDS_PER_STRIP, LED_PIN_B, NEO_GRB + NEO_KHZ800);

unsigned long moveStart = 0, moveEnd = 0;   // millis() of the last move, for D reports
uint8_t pos = START_SLICE;          // slice catching now

struct Entry { long id; uint8_t slice; unsigned long due; };
const int QMAX = 16;
Entry queue[QMAX];
int qn = 0;

char line[40];
uint8_t lineLen = 0;

// ---------------------------------------------------------------- stepping
inline void stepPulse() {
  digitalWrite(PUL_PIN, HIGH);
  delayMicroseconds(PULSE_US);
  digitalWrite(PUL_PIN, LOW);
  delayMicroseconds(PULSE_US);
}

// Move n slices in one continuous trapezoidal motion lasting MOVE_TIME_MS (runMove() of
// stepper_serial_moves.ino with TIME_SCALES_WITH_LENGTH = false). Blocks until the move is over.
void runMove(bool forward, int n) {
  long D = (long)n * STEPS_PER_SLICE;                     // total steps
  unsigned long totalUs = MOVE_TIME_MS * 1000UL;

  float accelFrac = constrain(ACCEL_FRACTION, 0.01, 0.5);
  float Tf = (float)totalUs;
  float Ta = Tf * accelFrac;                  // accel time (= decel time)
  float v  = (float)D / (Tf - Ta);            // cruise speed, steps per microsecond
  float pAccel = 0.5f * v * Ta;               // steps covered by end of accel

  digitalWrite(DIR_PIN, (forward != INVERT_DIR) ? HIGH : LOW);
  delayMicroseconds(100);                     // let DIR settle

  long done = 0;
  unsigned long start = micros();

  while (done < D) {
    float t = (float)(micros() - start);
    float target;

    if (t >= Tf) {
      target = (float)D;
    } else if (t < Ta) {                      // accelerating
      target = v * t * t / (2.0f * Ta);
    } else if (t < Tf - Ta) {                 // cruising
      target = pAccel + v * (t - Ta);
    } else {                                  // decelerating
      float r = Tf - t;
      target = (float)D - v * r * r / (2.0f * Ta);
    }

    long tgt = (long)target;
    if (tgt > D) tgt = D;
    while (done < tgt) {                      // catch up with the profile
      stepPulse();
      done++;
    }
  }
  while ((unsigned long)(micros() - start) < totalUs) {}   // the move lasts the full time
}

void savePos(uint8_t v) {
  if (REMEMBER_POSITION) EEPROM.update(EE_POS, v);
}

// Signed slice difference, the shorter way round: +1 = next slice number.
int shortest(uint8_t from, uint8_t to) {
  int d = ((int)to - (int)from) % N_SLICES;
  if (d < 0) d += N_SLICES;
  if (d > N_SLICES / 2) d -= N_SLICES;
  return d;
}

void moveTo(uint8_t target) {
  int d = shortest(pos, target);
  if (d == 0) return;
  savePos(EE_TURNING);
  moveStart = millis();
  runMove(d > 0, abs(d));
  moveEnd = millis();
  pos = target;
  savePos(pos);
}

// ---------------------------------------------------------------- queue
void queuePut(long id, uint8_t slice, unsigned long due) {
  for (int i = 0; i < qn; i++) {
    if (queue[i].id == id) {        // same id again: replace (the slice can change to Rework)
      for (int j = i; j < qn - 1; j++) queue[j] = queue[j + 1];
      qn--;
      break;
    }
  }
  if (qn >= QMAX) {
    Serial.print(F("?,FULL,")); Serial.println(id);
    return;
  }
  int i = qn;
  while (i > 0 && (long)(queue[i - 1].due - due) > 0) {   // keep sorted by landing time
    queue[i] = queue[i - 1];
    i--;
  }
  queue[i] = {id, slice, due};
  qn++;
  Serial.print(F("OK,")); Serial.println(id);
}

void landPassed(unsigned long now) {
  while (qn > 0 && (long)(now - queue[0].due) >= 0) {
    Serial.print(F("D,")); Serial.print(queue[0].id); Serial.print(',');
    // landed during the last move (reported just after it): the pie was turning
    bool during = (long)(queue[0].due - moveStart) >= 0 && (long)(moveEnd - queue[0].due) > 0;
    Serial.println(during ? 0 : pos);
    for (int j = 0; j < qn - 1; j++) queue[j] = queue[j + 1];
    qn--;
  }
}

// ---------------------------------------------------------------- serial
bool validSlice(long s) { return s >= 1 && s <= N_SLICES; }

void handleLine(char *s) {
  char *cmd = strtok(s, ",");
  if (!cmd) return;
  char c = toupper(cmd[0]);
  char *a = strtok(NULL, ","), *b = strtok(NULL, ","), *m = strtok(NULL, ",");
  if (c == 'Q') {
    if (!a || !b || !m || !validSlice(atol(b)) || atol(m) < 0) { Serial.println(F("?,BAD Q")); return; }
    queuePut(atol(a), (uint8_t)atol(b), millis() + (unsigned long)atol(m));
  } else if (c == 'H') {
    Serial.println(F("READY"));     // early build: no home sensor
  } else if (c == 'S') {
    qn = 0;
  } else if (c == 'G') {
    if (!a || !validSlice(atol(a))) { Serial.println(F("?,BAD G")); return; }
    qn = 0;
    queuePut(-1, (uint8_t)atol(a), millis() + 3000);    // stay there 3 s, then back to Rework
  } else if (c == 'Z') {
    if (!a || !validSlice(atol(a))) { Serial.println(F("?,BAD Z")); return; }
    pos = (uint8_t)atol(a);
    savePos(pos);
    Serial.print(F("POS,")); Serial.println(pos);
  } else if (c == 'P') {
    Serial.print(F("POS,")); Serial.println(pos);
  } else {
    Serial.println(F("?,UNKNOWN"));
  }
}

void readSerial() {
  while (Serial.available()) {
    char ch = Serial.read();
    if (ch == '\r') continue;
    if (ch == '\n') {
      line[lineLen] = 0;
      if (lineLen) handleLine(line);
      lineLen = 0;
    } else if (lineLen < sizeof(line) - 1) {
      line[lineLen++] = ch;
    } else {
      lineLen = 0;                  // too long: drop it
      Serial.println(F("?,LONG"));
    }
  }
}

// ---------------------------------------------------------------- main
void setup() {
  pinMode(PUL_PIN, OUTPUT);
  pinMode(DIR_PIN, OUTPUT);
  digitalWrite(PUL_PIN, LOW);
  Serial.begin(115200);

  stripA.begin(); stripB.begin();
  for (uint8_t i = 0; i < LEDS_PER_STRIP; i++) {
    stripA.setPixelColor(i, LED_LEVEL, LED_LEVEL, LED_LEVEL);
    stripB.setPixelColor(i, LED_LEVEL, LED_LEVEL, LED_LEVEL);
  }
  stripA.show(); stripB.show();     // the only LED writes (see the top of the file)

  uint8_t saved = REMEMBER_POSITION ? EEPROM.read(EE_POS) : 0xFF;
  if (validSlice(saved)) {
    pos = saved;
  } else {
    pos = START_SLICE;
    if (saved == EE_TURNING)
      Serial.println(F("?,RESTARTED WHILE TURNING: assuming slice 1 is catching. Re-align the pie, then send Z,1"));
    savePos(pos);
  }
  Serial.print(F("POS,")); Serial.println(pos);
  Serial.println(F("READY"));
}

void loop() {
  readSerial();
  landPassed(millis());
  uint8_t target = qn > 0 ? queue[0].slice : IDLE_SLICE;
  if (target != pos) moveTo(target);
}
