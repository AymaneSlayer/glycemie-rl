import numpy as np

from . import config
from .features import SCALER_PATH, Scaler, raw_features


def apply_action(action, pid_basal, basal_ref):
    a = np.clip(np.asarray(action, dtype=np.float64).reshape(-1), -1.0, 1.0)
    basal = float(np.clip(pid_basal + config.RESIDUAL_SCALE * a[0] * basal_ref, 0.0, config.MAX_BASAL_FACTOR * basal_ref))
    factor = float(2.0 ** (config.BOLUS_LOG2_RANGE * a[1])) if a.size > 1 else 1.0
    return basal, factor


class Controller:
    name = "controller"

    def reset(self, sim):
        pass

    def act(self, sim):
        raise NotImplementedError

    def bolus_factor(self):
        return 1.0


class BasalBolus(Controller):
    name = "basal_bolus"

    def act(self, sim):
        return sim.profile["basal"]


class PID(Controller):
    name = "pid"

    def __init__(self, kp=1e-4, ki=2e-7, kd=3e-4, target=config.BG_TARGET):
        self.kp, self.ki, self.kd, self.target = kp, ki, kd, target

    def reset(self, sim):
        self.integral = 0.0
        self.prev = sim.cgm[-1]

    def act(self, sim):
        g = sim.cgm[-1]
        err = g - self.target
        self.integral = float(np.clip(self.integral + err * config.SAMPLE_TIME, -5e4, 5e4))
        deriv = (g - self.prev) / config.SAMPLE_TIME
        self.prev = g
        scale = sim.profile["basal"] / 0.02
        u = sim.profile["basal"] + scale * (self.kp * err + self.ki * self.integral + self.kd * deriv)
        if g < config.BG_LOW + 10:
            u = 0.0
        return float(np.clip(u, 0.0, config.MAX_BASAL_FACTOR * sim.profile["basal"]))


class RL(Controller):
    def __init__(self, model_path, name=None):
        from stable_baselines3 import PPO, SAC
        algo = SAC if "sac" in str(model_path).lower() else PPO
        self.model = algo.load(str(model_path), device="cpu")
        self.scaler = Scaler.load(SCALER_PATH)
        self.name = name or model_path.stem
        self.pid = PID()
        self._factor = 1.0

    def reset(self, sim):
        self.pid.reset(sim)

    def act(self, sim):
        pid_basal = self.pid.act(sim)
        self.last_features = raw_features(sim, pid_basal)
        obs = self.scaler.transform(self.last_features[None, :])[0]
        action, _ = self.model.predict(obs, deterministic=True)
        basal, self._factor = apply_action(action, pid_basal, sim.profile["basal"])
        return basal

    def bolus_factor(self):
        return self._factor


def baselines():
    return [BasalBolus(), PID()]


def load_rl_models():
    controllers = []
    for path in sorted(config.MODELS_DIR.glob("*_best.zip")):
        controllers.append(RL(path, name=path.stem.replace("_best", "")))
    return controllers
