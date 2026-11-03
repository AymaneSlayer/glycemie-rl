import json

import numpy as np

from . import config

CGM_FEATURES = [f"cgm_t-{i * config.SAMPLE_TIME}" for i in range(config.HISTORY - 1, -1, -1)]
FEATURE_NAMES = CGM_FEATURES + ["cgm_slope", "iob_rel", "cob_rel", "meal_now_rel", "pid_rel", "hour_sin", "hour_cos"]


def raw_features(sim, pid_basal):
    cgm = np.array(sim.cgm[-config.HISTORY:], dtype=np.float64)
    if cgm.size < config.HISTORY:
        cgm = np.concatenate([np.full(config.HISTORY - cgm.size, cgm[0]), cgm])
    slope = (cgm[-1] - cgm[-4]) / (3 * config.SAMPLE_TIME)
    hourly_basal = sim.profile["basal"] * 60
    iob_rel = sim.iob() / hourly_basal
    cob_rel = sim.cob() / sim.profile["cr"] / hourly_basal
    meal_now_rel = sim.pending_meal / sim.profile["cr"] / hourly_basal
    pid_rel = pid_basal / sim.profile["basal"]
    t = sim.times[-1]
    hour = t.hour + t.minute / 60
    angle = 2 * np.pi * hour / 24
    return np.concatenate([cgm, [slope, iob_rel, cob_rel, meal_now_rel, pid_rel, np.sin(angle), np.cos(angle)]])


class Scaler:
    def __init__(self, mean=None, std=None):
        self.mean = None if mean is None else np.asarray(mean, dtype=np.float64)
        self.std = None if std is None else np.asarray(std, dtype=np.float64)

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        self.std[self.std < 1e-8] = 1.0
        return self

    def transform(self, X):
        return ((np.asarray(X, dtype=np.float64) - self.mean) / self.std).astype(np.float32)

    def inverse(self, X):
        return np.asarray(X) * self.std + self.mean

    def save(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"features": FEATURE_NAMES, "mean": self.mean.tolist(), "std": self.std.tolist()}, indent=2))

    @classmethod
    def load(cls, path):
        d = json.loads(path.read_text())
        if d["features"] != FEATURE_NAMES:
            raise ValueError("feature mismatch, rerun python -m src.data")
        return cls(d["mean"], d["std"])


SCALER_PATH = config.DATA_DIR / "scaler.json"
