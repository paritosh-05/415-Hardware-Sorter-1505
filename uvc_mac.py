"""Exposure and gain for a USB (UVC) camera on macOS.

On Windows, OpenCV's DirectShow driver passes CAP_PROP_EXPOSURE / CAP_PROP_GAIN to the camera. On macOS
the AVFoundation driver ignores them, so this module sends the same standard UVC requests to the camera
itself over USB (pyusb + libusb), while AVFoundation keeps streaming the video. Exposure is kept in the
same units as on Windows (DirectShow: log2 of seconds, -11 ~ 0.5 ms), so one config.json works on both.

Needs: pip install pyusb libusb-package
"""
import math

# UVC 1.1 constants
CC_VIDEO, SC_VIDEOCONTROL = 0x0E, 0x01
CS_INTERFACE, VC_INPUT_TERMINAL, VC_PROCESSING_UNIT = 0x24, 0x02, 0x05
ITT_CAMERA = 0x0201
SET_CUR, GET_CUR, GET_MIN, GET_MAX, GET_RES = 0x01, 0x81, 0x82, 0x83, 0x84
CT_AE_MODE, CT_EXPOSURE_ABS = 0x02, 0x04      # camera terminal controls
PU_GAIN = 0x04                                 # processing unit control
AE_MANUAL, AE_AUTO, AE_APERTURE_PRIORITY = 1, 2, 8


def dshow_to_uvc(value):
    """DirectShow exposure (log2 seconds) -> UVC exposure time (units of 100 us)."""
    return max(1, int(round(10000.0 * 2.0 ** float(value))))


def uvc_to_dshow(units):
    return math.log2(max(units, 1) / 10000.0)


def parse_vc_descriptors(extra):
    """Class-specific VideoControl descriptors -> (camera terminal id, processing unit id)."""
    ct = pu = None
    i, extra = 0, bytes(extra)
    while i + 2 < len(extra):
        n = extra[i]
        if n < 3:
            break
        d = extra[i:i + n]
        if d[1] == CS_INTERFACE and d[2] == VC_INPUT_TERMINAL and n >= 6:
            if int.from_bytes(d[4:6], "little") == ITT_CAMERA and ct is None:
                ct = d[3]
        elif d[1] == CS_INTERFACE and d[2] == VC_PROCESSING_UNIT and n >= 4 and pu is None:
            pu = d[3]
        i += n
    return ct, pu


def _backend():
    try:
        import libusb_package
        return libusb_package.get_libusb1_backend()
    except ImportError:
        return None                              # let pyusb look for a system libusb


def _name(dev):
    import usb.util
    try:
        return f"{usb.util.get_string(dev, dev.iManufacturer) or ''} {usb.util.get_string(dev, dev.iProduct) or ''}".strip()
    except Exception:
        return ""


class UvcControls:
    """Exposure/gain of one UVC camera. Use UvcControls.find(); it raises RuntimeError with a reason."""

    def __init__(self, dev, intf, ct, pu, name):
        self.dev, self.intf, self.ct, self.pu, self.name = dev, intf, ct, pu, name

    @classmethod
    def find(cls, match=""):
        """match: '' = the only / the OV9281 UVC camera; 'vvvv:pppp' (hex) or part of its USB name."""
        try:
            import usb.core
        except ImportError:
            raise RuntimeError("pyusb is not installed: pip install pyusb libusb-package")
        try:
            devs = list(usb.core.find(find_all=True, backend=_backend()))
        except usb.core.NoBackendError:
            raise RuntimeError("no libusb found: pip install libusb-package")
        cams = []
        for dev in devs:
            try:
                cfg = dev.get_active_configuration()
            except Exception:
                try:
                    cfg = dev[0]
                except Exception:
                    continue
            for intf in cfg:
                if intf.bInterfaceClass == CC_VIDEO and intf.bInterfaceSubClass == SC_VIDEOCONTROL:
                    ct, pu = parse_vc_descriptors(getattr(intf, "extra_descriptors", None) or [])
                    cams.append((dev, intf.bInterfaceNumber, ct, pu, _name(dev)))
                    break
        if not cams:
            raise RuntimeError("no USB (UVC) camera found. Is it plugged in?")
        label = lambda c: f"{c[0].idVendor:04x}:{c[0].idProduct:04x} {c[4]}".strip()
        if match:
            m = match.lower()
            pick = [c for c in cams if m == f"{c[0].idVendor:04x}:{c[0].idProduct:04x}" or m in c[4].lower()]
        elif len(cams) == 1:
            pick = cams
        else:
            pick = [c for c in cams if "9281" in c[4]] or []
        if len(pick) != 1:
            raise RuntimeError(f"{'no' if not pick else 'more than one'} USB camera matches; found: "
                               f"{', '.join(label(c) for c in cams)}. Set \"uvc_device\" in config.json to one of them.")
        dev, intf, ct, pu, name = pick[0]
        if ct is None:
            raise RuntimeError(f"camera {label(pick[0])} has no exposure control in its USB descriptors")
        return cls(dev, intf, ct, pu, label(pick[0]))

    # ---- raw requests ---------------------------------------------------------------------
    def _get(self, req, unit, sel, n):
        data = self.dev.ctrl_transfer(0xA1, req, sel << 8, (unit << 8) | self.intf, n, timeout=500)
        return int.from_bytes(bytes(data), "little", signed=False)

    def _set(self, unit, sel, value, n):
        self.dev.ctrl_transfer(0x21, SET_CUR, sel << 8, (unit << 8) | self.intf,
                               int(value).to_bytes(n, "little", signed=False), timeout=500)

    def _range(self, unit, sel, n):
        return self._get(GET_MIN, unit, sel, n), self._get(GET_MAX, unit, sel, n)

    # ---- the same calls the Windows (DirectShow) version makes --------------------------------
    def set_exposure(self, value):
        """value in DirectShow units (log2 seconds). Switches to manual exposure. Returns the value now set."""
        self._set(self.ct, CT_AE_MODE, AE_MANUAL, 1)
        lo, hi = self._range(self.ct, CT_EXPOSURE_ABS, 4)
        self._set(self.ct, CT_EXPOSURE_ABS, min(max(dshow_to_uvc(value), lo), hi), 4)
        return self.get_exposure()

    def get_exposure(self):
        return uvc_to_dshow(self._get(GET_CUR, self.ct, CT_EXPOSURE_ABS, 4))

    def exposure_ms(self):
        return self._get(GET_CUR, self.ct, CT_EXPOSURE_ABS, 4) / 10.0

    def auto_exposure(self):
        try:
            modes = self._get(GET_RES, self.ct, CT_AE_MODE, 1)   # bitmap of supported modes
        except Exception:
            modes = AE_APERTURE_PRIORITY
        self._set(self.ct, CT_AE_MODE, AE_APERTURE_PRIORITY if modes & AE_APERTURE_PRIORITY or not modes & AE_AUTO
                  else AE_AUTO, 1)

    def set_gain(self, value):
        if self.pu is None:
            raise RuntimeError("camera has no gain control")
        lo, hi = self._range(self.pu, PU_GAIN, 2)
        self._set(self.pu, PU_GAIN, min(max(int(round(value)), lo), hi), 2)
        return self.get_gain()

    def get_gain(self):
        return float(self._get(GET_CUR, self.pu, PU_GAIN, 2)) if self.pu is not None else -1.0
