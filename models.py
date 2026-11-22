"""The two classifiers and the rule that combines them.

GeoModel  - geometric model. Learns, from taught examples, what each part measures on THIS rig
            (mean and spread of length, width, area, hole size...). A part can have several
            "modes" (e.g. nut lying flat / nut on its side); they are found automatically.
AppModel  - appearance model. Nearest-neighbour on the appearance descriptor (how the part looks
            inside and around its outline). Adding a part = adding examples, no retraining.
fuse()    - sort only when both agree; anything else goes to re-sort.

Both models answer with a label and a level:
    certain > confident > unsure > unknown
"unknown" means "this looks like nothing I was taught" and always leads to re-sort.
"""
import json
import os
from dataclasses import dataclass, field

import numpy as np

from config import RESORT
from segment import GEO, desc_distance

LEVELS = ["unknown", "unsure", "confident", "certain"]

# smallest spread the geometric model will believe, per measurement: (absolute, relative)
SIGMA_FLOOR = {"L": (0.30, 0.025), "W": (0.30, 0.025), "Wm": (0.30, 0.025),
               "A": (6.0, 0.05), "Dh": (0.5, 0.0), "S": (0.025, 0.0)}
# how far apart two taught examples of the same part may be and still count as the same mode
MODE_TOL = {"L": (1.5, 0.08), "W": (1.5, 0.08), "Wm": (1.5, 0.10), "A": (25.0, 0.15), "Dh": (2.0, 0.0), "S": (0.10, 0.0)}


@dataclass
class Verdict:
    label: str = None           # best matching part (None if nothing is close)
    level: str = "unknown"
    score: float = 0.0          # geometric: worst z of the best match. appearance: nearest distance
    rival: str = None           # closest other part
    rival_score: float = 0.0
    info: dict = field(default_factory=dict)

    def at_least(self, level):
        return LEVELS.index(self.level) >= LEVELS.index(level)


def _tol(table, key, value):
    a, r = table[key]
    return max(a, r * abs(value))


# ===================================================================================== geometric
class GeoModel:
    def __init__(self, cfg):
        self.cfg = cfg
        self.modes = []          # dicts: label, n, mean[], sigma[]

    # ---- teaching ----
    def fit(self, samples):
        """samples: {label: [ {L:..,W:..,...}, ... ]}. Returns a text report."""
        self.modes, report = [], []
        for label, rows in sorted(samples.items()):
            X = np.array([[r[k] for k in GEO] for r in rows], float)
            groups = []                                   # leader clustering -> modes
            for x in X:
                for g in groups:
                    c = np.mean(g, axis=0)
                    if all(abs(x[j] - c[j]) <= _tol(MODE_TOL, k, c[j]) for j, k in enumerate(GEO)):
                        g.append(x)
                        break
                else:
                    groups.append([x])
            keep_min = max(3, int(np.ceil(0.05 * len(X))))
            for g in sorted(groups, key=len, reverse=True):
                g = np.array(g)
                if len(g) < keep_min:
                    report.append(f"  {label}: ignored {len(g)} odd example(s) (not enough to form a mode)")
                    continue
                mean = g.mean(axis=0)
                sd = g.std(axis=0, ddof=1) if len(g) > 1 else np.zeros(len(GEO))
                sigma = np.array([max(sd[j], _tol(SIGMA_FLOOR, k, mean[j])) for j, k in enumerate(GEO)])
                self.modes.append(dict(label=label, n=int(len(g)), mean=mean.tolist(), sigma=sigma.tolist()))
                report.append(f"  {label}: mode with {len(g)} examples  " +
                              "  ".join(f"{k}={m:.2f}±{s:.2f}" for k, m, s in zip(GEO, mean, sigma)))
        return "\n".join(report)

    # ---- classifying ----
    def classify(self, geo):
        if not self.modes:
            return Verdict()
        x = np.array([geo[k] for k in GEO], float)
        best = {}                                           # label -> (worst z, which measurement)
        for m in self.modes:
            z = np.abs(x - np.array(m["mean"])) / np.array(m["sigma"])
            j = int(np.argmax(z))
            if m["label"] not in best or z[j] < best[m["label"]][0]:
                best[m["label"]] = (float(z[j]), GEO[j])
        order = sorted(best.items(), key=lambda kv: kv[1][0])
        (lab, (z1, why1)) = order[0]
        rival, z2 = (order[1][0], order[1][1][0]) if len(order) > 1 else (None, 99.0)
        g_cert, g_conf, g_unsure = self.cfg.geo_gate
        r_cert, r_conf = self.cfg.geo_rival
        if z1 <= g_cert and z2 >= r_cert:
            level = "certain"
        elif z1 <= g_conf and z2 >= r_conf:
            level = "confident"
        elif z1 <= g_unsure:
            level = "unsure"
        else:
            level = "unknown"
        return Verdict(lab if level != "unknown" else None, level, z1, rival, z2,
                       dict(worst=why1, nearest=lab))

    def save(self, path):
        with open(path, "w") as fh:
            json.dump(dict(features=GEO, modes=self.modes), fh, indent=1)

    def load(self, path):
        with open(path) as fh:
            self.modes = json.load(fh)["modes"]
        return self


# ==================================================================================== appearance
class AppModel:
    def __init__(self, cfg):
        self.cfg = cfg
        self.X = np.zeros((0, 1), np.float32)   # descriptors
        self.y = np.array([], dtype=object)     # labels
        self.g = np.array([], dtype=int)        # which taught part (track) each row came from
        self.novel = {}                         # label -> largest distance still considered familiar

    def fit(self, samples):
        """samples: {label: [ [descriptor, ...] per taught part ]}. Returns a text report."""
        X, y, g, gid = [], [], [], 0
        for label, tracks in sorted(samples.items()):
            for descs in tracks:
                for d in descs:
                    X.append(d); y.append(label); g.append(gid)
                gid += 1
        self.X, self.y, self.g = np.array(X, np.float32), np.array(y, dtype=object), np.array(g)
        # how far is a taught part from its nearest neighbour of the same label (other passes only)?
        report, loo_ok, loo_n = [], 0, 0
        near = {lab: [] for lab in set(y)}
        for i in range(len(self.X)):
            d = desc_distance(self.X[i], self.X)
            other = self.g != self.g[i]
            if not other.any():
                continue
            same = other & (self.y == self.y[i])
            if same.any():
                near[self.y[i]].append(d[same].min())
            j = np.where(other)[0][np.argmin(d[other])]
            loo_ok += self.y[j] == self.y[i]; loo_n += 1
        for lab in sorted(near):
            v = np.array(near[lab]) if near[lab] else np.array([0.3])
            self.novel[lab] = float(max(1.6 * np.percentile(v, 95), 0.30))
            report.append(f"  {lab}: {int((self.y == lab).sum())} views, typical distance {np.median(v):.2f}, "
                          f"unfamiliar beyond {self.novel[lab]:.2f}")
        if loo_n:
            report.append(f"  leave-one-part-out check: {loo_ok}/{loo_n} views matched their own label")
        return "\n".join(report)

    def classify(self, descs):
        """descs: descriptors of one part seen at several positions."""
        if len(self.X) == 0 or len(descs) == 0:
            return Verdict()
        k = self.cfg.app_k
        votes, d_same, d_other, tops = {}, [], [], []
        for a in descs:
            d = desc_distance(a, self.X)
            idx = np.argsort(d)[:k]
            w = 1.0 / (d[idx] + 0.05)
            tally = {}
            for i, wi in zip(idx, w):
                tally[self.y[i]] = tally.get(self.y[i], 0.0) + wi
            top = max(tally, key=tally.get)
            tops.append(top)
            for lab, v in tally.items():
                votes[lab] = votes.get(lab, 0.0) + v / w.sum() / len(descs)
        label = max(votes, key=votes.get)
        for a in descs:
            d = desc_distance(a, self.X)
            same = self.y == label
            d_same.append(d[same].min())
            d_other.append(d[~same].min() if (~same).any() else 99.0)
        share = votes[label]
        d1, d2 = float(np.median(d_same)), float(np.median(d_other))
        ratio = d2 / max(d1, 1e-6)
        others = {l: v for l, v in votes.items() if l != label}
        rival = max(others, key=others.get) if others else None
        lim = self.novel.get(label, 0.5)
        if d1 > lim:
            level, out = "unknown", None
        elif share >= 0.95 and ratio >= 2.0 and d1 <= 0.7 * lim:
            level, out = "certain", label
        elif share >= 0.80 and ratio >= 1.4:
            level, out = "confident", label
        else:
            level, out = "unsure", label
        return Verdict(out, level, d1, rival, d2, dict(share=round(share, 2), ratio=round(ratio, 2), nearest=label))

    def save(self, path):
        np.savez_compressed(path, X=self.X, y=self.y.astype(str), g=self.g,
                            novel=json.dumps(self.novel))

    def load(self, path):
        z = np.load(path, allow_pickle=False)
        self.X, self.y, self.g = z["X"], z["y"].astype(object), z["g"]
        self.novel = json.loads(str(z["novel"]))
        return self


# ========================================================================================= fusion
def fuse(geo: Verdict, app: Verdict, cfg):
    """Returns (label or RESORT, reason, override_candidate)."""
    if geo.at_least("confident") and app.at_least("confident"):
        if geo.label == app.label:
            return geo.label, "agree", False
        return RESORT, f"disagree ({geo.label} vs {app.label})", False
    if geo.level == "unknown" or app.level == "unknown":
        return RESORT, "unknown to " + ("both" if geo.level == app.level else
                                        "geometry" if geo.level == "unknown" else "appearance"), False
    # one or both are only 'unsure'
    strong, weak = (geo, app) if LEVELS.index(geo.level) >= LEVELS.index(app.level) else (app, geo)
    candidate = strong.level == "certain" and weak.level == "unsure" and weak.label == strong.label
    if candidate and cfg.allow_single_certain:
        return strong.label, "one certain, other unsure", True
    if geo.label != app.label:
        return RESORT, f"unsure and split ({geo.label} vs {app.label})", False
    return RESORT, "not confident enough", candidate


# ================================================================================ teach data files
def save_sample(folder, label, index, geo_rows, descs, crop=None):
    import cv2
    d = os.path.join(folder, label)
    os.makedirs(d, exist_ok=True)
    np.savez_compressed(os.path.join(d, f"{index:05d}.npz"),
                        geo=np.array([[r[k] for k in GEO] for r in geo_rows], np.float32),
                        desc=np.array(descs, np.float32))
    if crop is not None:
        cv2.imwrite(os.path.join(d, f"{index:05d}.png"), crop)


def load_samples(folder):
    """-> geo {label: [median measurements per part]}, app {label: [[descriptors] per part]}"""
    geo, app = {}, {}
    if not os.path.isdir(folder):
        return geo, app
    for label in sorted(os.listdir(folder)):
        d = os.path.join(folder, label)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.endswith(".npz"):
                z = np.load(os.path.join(d, f))
                if len(z["geo"]) == 0:
                    continue
                geo.setdefault(label, []).append(dict(zip(GEO, np.median(z["geo"], axis=0).tolist())))
                app.setdefault(label, []).append(list(z["desc"]))
    return geo, app
