"""Link to the collector (the rotating pie) and a simulated collector for testing.

Serial protocol (text lines, 115200 baud, newline terminated). See PROTOCOL.md.
  laptop -> controller   Q,<id>,<slot>,<ms>    part <id> will have landed within <ms> milliseconds: be at
                                               <slot> and stay there until then.
                                               Sending the same id again replaces the earlier entry.
                         H                     go home and re-zero
                         S                     stop (hold position, clear the queue)
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
        self.ser = serial.Serial(port, baud, timeout=0)
        self.buf = b""
        time.sleep(2.0)                    # most boards reset when the port opens
        self.ser.write(b"H\n")

    def send(self, part_id, slot, ms):
        self.ser.write(f"Q,{part_id},{slot},{int(round(ms))}\n".encode())

    def poll(self):
        """Lines received from the controller since the last call."""
        self.buf += self.ser.read(4096)
        *lines, self.buf = self.buf.split(b"\n")
        return [ln.decode(errors="replace").strip() for ln in lines if ln.strip()]

    def close(self):
        try:
            self.ser.write(b"S\n")
            self.ser.close()
        except Exception:
            pass


class SimCollector:
    """Pretend collector for tests: obeys the same messages, on a simulated clock.

    After each landing it turns towards the next entry's slot, taking move_ms (or settle + steps *
    step_ms). landed() reports, for every part, the slot it really fell into, or None if the pie
    was still turning when it landed.
    """

    def __init__(self, cfg, move_ms=500.0):
        self.cfg, self.move_ms = cfg, move_ms
        self.entries = {}          # id -> (t_land, slot)
        self.now = 0.0

    def set_time(self, now):
        self.now = now

    def send(self, part_id, slot, ms):
        self.entries[part_id] = (self.now + ms / 1000.0, slot)

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

    def schedule(self):
        """-> [(id, slot, standing_from, standing_until)]: when the pie is at rest for each entry."""
        out, pos, t_free = [], self.cfg.resort_slot, -1e9
        for pid, (t_until, slot) in sorted(self.entries.items(), key=lambda kv: kv[1][0]):
            arrive = t_free + self._move_time(pos, slot)
            out.append((pid, slot, arrive, t_until))
            pos, t_free = slot, max(t_until, arrive)
        return out

    def slot_at(self, t):
        """Slot under the drop at time t, or None while the pie is turning."""
        sched = self.schedule()
        for i, (pid, slot, t_from, t_until) in enumerate(sched):
            nxt = sched[i + 1][2] if i + 1 < len(sched) else 1e18
            if t_from <= t <= t_until + 1e-9:
                return slot
            if t_until < t < nxt:
                # between entries: still standing if the next entry is the same slot, else turning
                return slot if i + 1 >= len(sched) or sched[i + 1][1] == slot else None
        return self.cfg.resort_slot if sched and t < sched[0][2] else None

    def landed(self):
        """-> {id: slot it fell into if it lands exactly when predicted}."""
        return {pid: (slot if t_from <= t_until else None) for pid, slot, t_from, t_until in self.schedule()}
