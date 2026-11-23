// TB6600 + NEMA 23 (1.8 deg): serial-controlled 180-degree moves.
//
// Type a string of l and r in the Serial Monitor and press Enter
// (set the monitor's line ending to "Newline" or "Both NL & CR").
//   r = 180 deg clockwise, l = 180 deg counter-clockwise
// Consecutive identical letters merge into one continuous move, up to
// MAX_STACK letters. A longer run is split into several moves.
//   rrr    -> 540 CW
//   rrrrr  -> 720 CW, then 180 CW
//   rrlrr  -> 360 CW, 180 CCW, 360 CW
// Each continuous move uses a trapezoidal velocity profile and takes
// MOVE_TIME_MS in total.

const uint8_t PUL_PIN = 2;   // -> TB6600 PUL+
const uint8_t DIR_PIN = 3;   // -> TB6600 DIR+

// ---------------- Settings ----------------
const int MICROSTEP = 16;                 // MUST match the driver DIP switches
const unsigned long MOVE_TIME_MS = 300;   // total time of one continuous move
const int MAX_STACK = 4;                  // max letters merged into one move
const float ACCEL_FRACTION = 0.25;        // share of the move spent speeding up
                                          // (and the same again slowing down).
                                          // 0.5 = triangle, smaller = flatter top.
const unsigned long PAUSE_BETWEEN_MOVES_MS = 0;
const bool INVERT_DIR = false;            // flip if 'r' turns the wrong way

// false: every continuous move takes MOVE_TIME_MS, however long it is.
// true:  each 180 deg takes MOVE_TIME_MS (rrr takes 3x as long).
const bool TIME_SCALES_WITH_LENGTH = false;

const unsigned int PULSE_US = 5;          // step pulse high/low width
// ------------------------------------------

const long STEPS_PER_180 = 100L * MICROSTEP;   // 200 steps/rev / 2

inline void stepPulse() {
  digitalWrite(PUL_PIN, HIGH);
  delayMicroseconds(PULSE_US);
  digitalWrite(PUL_PIN, LOW);
  delayMicroseconds(PULSE_US);
}

// Move n * 180 degrees in one continuous trapezoidal motion.
void runMove(bool cw, int n) {
  long D = (long)n * STEPS_PER_180;                       // total steps
  unsigned long timeMs = TIME_SCALES_WITH_LENGTH ? MOVE_TIME_MS * n : MOVE_TIME_MS;
  unsigned long totalUs = timeMs * 1000UL;

  float accelFrac = constrain(ACCEL_FRACTION, 0.01, 0.5);
  float Tf = (float)totalUs;
  float Ta = Tf * accelFrac;                  // accel time (= decel time)
  float v  = (float)D / (Tf - Ta);            // cruise speed, steps per microsecond
  float pAccel = 0.5f * v * Ta;               // steps covered by end of accel

  digitalWrite(DIR_PIN, (cw != INVERT_DIR) ? HIGH : LOW);
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

    // Emit however many steps are needed to catch up with the profile
    while (done < tgt) {
      stepPulse();
      done++;
    }
  }

  // Make sure the move lasts the full time
  while ((unsigned long)(micros() - start) < totalUs) {}

  if (PAUSE_BETWEEN_MOVES_MS > 0) delay(PAUSE_BETWEEN_MOVES_MS);
}

void announceAndRun(char dir, int count) {
  Serial.print(dir == 'r' ? F("CW  ") : F("CCW "));
  Serial.print(count * 180);
  Serial.println(F(" deg"));
  runMove(dir == 'r', count);
}

void setup() {
  pinMode(PUL_PIN, OUTPUT);
  pinMode(DIR_PIN, OUTPUT);
  digitalWrite(PUL_PIN, LOW);
  Serial.begin(115200);
  Serial.println(F("Enter a string of l and r, then press Enter (e.g. llrl, rrr)."));
}

void loop() {
  if (!Serial.available()) return;

  String line = Serial.readStringUntil('\n');
  line.toLowerCase();

  char dir = 0;
  int count = 0;

  for (unsigned int i = 0; i < line.length(); i++) {
    char c = line[i];
    if (c != 'l' && c != 'r') continue;       // ignore spaces, \r, other characters

    if (c == dir && count < MAX_STACK) {
      count++;                                // extend the current move
    } else {
      if (count > 0) announceAndRun(dir, count);
      dir = c;                                // start a new move
      count = 1;
    }
  }
  if (count > 0) announceAndRun(dir, count);

  Serial.println(F("Done."));
}
