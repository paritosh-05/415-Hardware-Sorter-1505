"""Link to the collector (the rotating pie) and a simulated collector for testing.

Serial protocol (text lines, 115200 baud, newline terminated). See PROTOCOL.md.
  laptop -> controller   Q,<id>,<slot>,<ms>    part <id> will have landed within <ms> milliseconds: be at
                                               <slot> and stay there until then.
                                               Sending the same id again replaces the earlier entry.
                         H                     go home and re-zero
                         S                     stop (hold position, clear the queue)
                         A                     alive: about once a second while the sorter runs (status light)
  controller -> laptop   OK,<id>               entry accepted
                         D,<id>,<slot>         drop time of <id> passed while standing at <slot>
                         E,<text>              error, e.g. E,HOME (home sensor did not match)
                         READY                 homed and ready again

The controller does the timing: it keeps the entries sorted by landing time, and after each
landing it turns to the slot of the next entry.
"""
import time


class NullLink:
    """No collector attached: just remember what would have been sent."""

    def __init__(self):
        self.sent = []

    def send(self, part_id, slot, ms):
        self.sent.append((part_id, slot, ms))

    def poll(self):
        return []

    def close(self):
        pass


class SerialLink:
    def __init__(self, port, baud=115200):
        import serial                      # pip install pyserial
        try:
            self.ser = serial.Serial(port, baud, timeout=0)
        except serial.SerialException as e:
            print(f"Could not open serial port {port}: {e}\nPorts found:")
            list_ports()
            raise SystemExit(1)
        self.buf = b""
        self.t_alive = 0.0
        time.sleep(2.0)                    # most boards reset when the port opens
        self.ser.write(b"H\n")

    def send(self, part_id, slot, ms):
        self.ser.write(f"Q,{part_id},{slot},{int(round(ms))}\n".encode())

    def poll(self):
        """Lines received from the controller since the last call. Called on every frame, so it also
        tells the controller about once a second that the sorter is running (its status light: PROTOCOL.md)."""
        now = time.monotonic()
        if now - self.t_alive >= 1.0:
            self.t_alive = now
            self.ser.write(b"A\n")
        self.buf += self.ser.read(4096)
        *lines, self.buf = self.buf.split(b"\n")
        return [ln.decode(errors="replace").strip() for ln in lines if ln.strip()]

    def close(self):
        try:
            self.ser.write(b"S\n")
            self.ser.close()
        except Exception:
            pass


def list_ports():
    """Print the serial ports (COM5 etc. on Windows; on macOS use the /dev/cu.* name, not /dev/tty.*)."""
    try:
        from serial.tools import list_ports as lp
    except ImportError:
        print("  pyserial is not installed: pip install pyserial")
        return
    ports = sorted(lp.comports(), key=lambda p: p.device)
    for p in ports:
        if "Bluetooth" in p.device or "debug-console" in p.device:
            continue                       # macOS built-ins, never the collector
        print(f"  {p.device}  {p.description if p.description != 'n/a' else ''}".rstrip())
    if not any("Bluetooth" not in p.device and "debug-console" not in p.device for p in ports):
        print("  none (is the collector's USB cable plugged in?)")


class SimCollector:
    """Pretend collector for tests: obeys the same messages, on a simulated clock, like collector.ino.

    It always heads for the slot of the earliest entry that has not landed; after each landing it turns
    towards the next entry's slot, taking move_ms (or settle + steps * step_ms). It cannot start a move
    before the message for it was sent. With idle_slot set (the real collector: Rework), it goes back
    there whenever its queue is empty, and finishes that move before turning to a new part.
    landed() reports, for every part, the slot it really fell into, or None if the pie was still turning.
    """

    def __init__(self, cfg, move_ms=500.0, idle_slot=None):
        self.cfg, self.move_ms, self.idle = cfg, move_ms, idle_slot
        self.entries = {}          # id -> (t_land, slot, t_first_sent)
        self.now = 0.0

    def set_time(self, now):
        self.now = now

    def send(self, part_id, slot, ms):
        t_sent = self.entries[part_id][2] if part_id in self.entries else self.now
        self.entries[part_id] = (self.now + ms / 1000.0, slot, t_sent)

    def poll(self):
        return []

    def close(self):
        pass

    def _move_time(self, a, b):
        if a == b:
            return 0.0
        if self.cfg.step_ms > 0:
            d = abs(a - b) % self.cfg.n_slots
            return (self.cfg.settle_ms + min(d, self.cfg.n_slots - d) * self.cfg.step_ms) / 1000.0
        return self.move_ms / 1000.0

    def _stands(self):
        """-> [(id or None, slot, standing_from, standing_until)]: every time the pie is at rest.
        id None = waiting at the start / idle slot. The last stand lasts forever."""
        start = self.idle if self.idle is not None else self.cfg.resort_slot
        stands, pos, t_free = [], start, -1e18
        for pid, (t_until, slot, t_sent) in sorted(self.entries.items(), key=lambda kv: kv[1][0]):
            if t_sent > t_free:                                  # queue was empty when this was sent
                if self.idle is not None and pos != self.idle:   # ...so the pie went back to idle first
                    t_idle = t_free + self._move_time(pos, self.idle)
                    stands.append((None, self.idle, t_idle, max(t_sent, t_idle)))
                    pos, t_free = self.idle, max(t_sent, t_idle)
                else:
                    t_free = t_sent
            arrive = t_free + self._move_time(pos, slot)
            stands.append((pid, slot, arrive, t_until))
            pos, t_free = slot, max(t_until, arrive)
        if self.idle is not None and pos != self.idle:
            stands.append((None, self.idle, t_free + self._move_time(pos, self.idle), 1e18))
        return stands

    def schedule(self):
        """-> [(id, slot, standing_from, standing_until)] for each part."""
        return [s for s in self._stands() if s[0] is not None]

    def slot_at(self, t):
        """Slot under the drop at time t, or None while the pie is turning."""
        stands = self._stands()
        start = self.idle if self.idle is not None else self.cfg.resort_slot
        prev_slot, prev_from = start, -1e18
        for pid, slot, t_from, t_until in stands:
            leave = t_from - self._move_time(prev_slot, slot)     # when the pie left the previous stand
            if prev_from <= t <= leave + 1e-9:
                return prev_slot
            if leave < t < t_from:
                return None
            prev_slot, prev_from = slot, t_from
        return prev_slot if t >= prev_from else None

    def landed(self):
        """-> {id: slot it fell into if it lands exactly when predicted}."""
        return {pid: (slot if t_from <= t_until else None) for pid, slot, t_from, t_until in self.schedule()}
