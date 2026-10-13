"""Pure logic (no camera, no model) so it can be tested on its own."""
from collections import deque
from dataclasses import dataclass

CLASS_NAMES = [
    "hammer", "screwdriver", "boxcutter", "adjustable_spanner", "pliers",
    "d_clamp", "hand_saw", "compact_blower", "m18_hammer_drill", "m18_battery_cp2",
]
DRILL = "m18_hammer_drill"
BATTERY = "m18_battery_cp2"


@dataclass
class Det:
    name: str
    conf: float
    box: tuple  # x1, y1, x2, y2


def _area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _intersection(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    return _area((x1, y1, x2, y2))


def battery_in_drill(battery: Det, drill: Det, min_cover: float = 0.6) -> bool:
    """Battery counts as installed when most of its box lies inside the drill box
    and its centre is inside the drill box. A battery sitting next to the drill
    overlaps little, so it is reported as loose."""
    b, d = battery.box, drill.box
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    centre_inside = d[0] <= cx <= d[2] and d[1] <= cy <= d[3]
    cover = _intersection(b, d) / _area(b) if _area(b) > 0 else 0.0
    return centre_inside and cover >= min_cover


def battery_status(dets, min_cover: float = 0.6):
    """Return a dict: batteries_total, installed, loose."""
    drills = [d for d in dets if d.name == DRILL]
    batteries = [d for d in dets if d.name == BATTERY]
    installed = 0
    used_drills = set()
    for bat in batteries:
        for i, dr in enumerate(drills):
            if i not in used_drills and battery_in_drill(bat, dr, min_cover):
                installed += 1
                used_drills.add(i)
                break
    return {
        "batteries_total": len(batteries),
        "installed": installed,
        "loose": len(batteries) - installed,
    }


class Stabilizer:
    """An item is 'present' if seen in at least `need` of the last `window` frames.
    Stops one missed frame from looking like a removed tool."""

    def __init__(self, window: int = 15, need: int = 8):
        self.window, self.need = window, need
        self.hist = {n: deque(maxlen=window) for n in CLASS_NAMES}

    def update(self, seen_names):
        seen = set(seen_names)
        for n in CLASS_NAMES:
            self.hist[n].append(n in seen)
        return {n for n in CLASS_NAMES if sum(self.hist[n]) >= self.need}


class Inventory:
    """Tracks which of the expected tools are on the board and emits
    REMOVED / RETURNED events when the stable state changes."""

    def __init__(self, expected=None):
        self.expected = list(expected or CLASS_NAMES)
        self.state = None  # unknown until first update

    def update(self, present):
        present = set(present) & set(self.expected)
        events = []
        if self.state is None:
            self.state = present
            return events
        for n in sorted(self.state - present):
            events.append(("REMOVED", n))
        for n in sorted(present - self.state):
            events.append(("RETURNED", n))
        self.state = present
        return events

    def missing(self):
        return sorted(set(self.expected) - (self.state or set()))
