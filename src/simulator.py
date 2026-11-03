from datetime import datetime

import numpy as np
import pandas as pd
from simglucose.actuator.pump import InsulinPump
from simglucose.analysis.risk import risk_index
from simglucose.controller.base import Action
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.simulation.env import T1DSimEnv
from simglucose.simulation.scenario import CustomScenario
from simglucose.simulation.scenario_gen import RandomScenario

from . import config

_QUEST = None
_PARAMS = None


def _tables():
    global _QUEST, _PARAMS
    if _QUEST is None:
        from simglucose.controller.basal_bolus_ctrller import CONTROL_QUEST, PATIENT_PARA_FILE
        _QUEST = pd.read_csv(CONTROL_QUEST).set_index("Name")
        _PARAMS = pd.read_csv(PATIENT_PARA_FILE).set_index("Name")
    return _QUEST, _PARAMS


def patient_profile(name):
    quest, params = _tables()
    return {
        "basal": params.loc[name, "u2ss"] * params.loc[name, "BW"] / 6000,
        "cr": float(quest.loc[name, "CR"]),
        "cf": float(quest.loc[name, "CF"]),
        "bw": float(params.loc[name, "BW"]),
    }


def make_scenario(seed, meals=None):
    start = datetime(2026, 1, 1, config.START_HOUR, 0, 0)
    if meals is None:
        return RandomScenario(start_time=start, seed=seed)
    return CustomScenario(start_time=start, scenario=[(h - config.START_HOUR, g) for h, g in meals])


class Simulator:
    def __init__(self, patient, seed=0, meals=None, meal_bolus=True):
        self.patient_name = patient
        self.seed = int(seed)
        self.meals = meals
        self.meal_bolus = meal_bolus
        self.profile = patient_profile(patient)
        self.reset()

    def reset(self):
        rng = np.random.RandomState(self.seed)
        patient = T1DPatient.withName(self.patient_name, random_init_bg=True, seed=int(rng.randint(1e6)))
        sensor = CGMSensor.withName("Dexcom", seed=int(rng.randint(1e6)))
        pump = InsulinPump.withName("Insulet")
        scenario = make_scenario(int(rng.randint(1e6)), self.meals)
        self.env = T1DSimEnv(patient, sensor, pump, scenario)
        step = self.env.reset()
        self.t = 0
        self.cgm = [float(step.observation.CGM)]
        self.bg = [float(step.info["bg"])]
        self.insulin = []
        self.basal = []
        self.bolus = []
        self.cho = []
        self.times = [step.info["time"]]
        self.pending_meal = 0.0
        self.done = False
        return self

    def bolus_for(self, carbs):
        if not self.meal_bolus or carbs <= 0:
            return 0.0
        glucose = self.cgm[-1]
        correction = max(glucose - config.BG_TARGET, 0) / self.profile["cf"] if glucose > 150 else 0.0
        return carbs / self.profile["cr"] + correction

    def step(self, basal_rate):
        basal_rate = float(np.clip(basal_rate, 0.0, self.profile["basal"] * config.MAX_BASAL_FACTOR))
        bolus_units = self.bolus_for(self.pending_meal)
        action = Action(basal=basal_rate, bolus=bolus_units / config.SAMPLE_TIME)
        step = self.env.step(action, reward_fun=lambda h: 0.0)
        info = step.info
        carbs = float(info["meal"]) * config.SAMPLE_TIME
        self.pending_meal = carbs
        self.t += 1
        self.cgm.append(float(step.observation.CGM))
        self.bg.append(float(info["bg"]))
        self.basal.append(basal_rate * config.SAMPLE_TIME)
        self.bolus.append(bolus_units)
        self.insulin.append(basal_rate * config.SAMPLE_TIME + bolus_units)
        self.cho.append(carbs)
        self.times.append(info["time"])
        bg = self.bg[-1]
        self.terminated = bg < config.BG_TERMINATE_LOW or bg > config.BG_TERMINATE_HIGH
        self.truncated = self.t >= config.EPISODE_STEPS
        self.done = self.terminated or self.truncated
        return bg

    def iob(self):
        n = config.IOB_HOURS * 60 // config.SAMPLE_TIME
        recent = np.array(self.insulin[-n:])
        if recent.size == 0:
            return 0.0
        age = np.arange(recent.size)[::-1]
        weights = 1 - age / n
        return float(np.sum(recent * weights))

    def cob(self):
        n = config.COB_HOURS * 60 // config.SAMPLE_TIME
        recent = np.array(self.cho[-n:])
        if recent.size == 0:
            return 0.0
        age = np.arange(recent.size)[::-1]
        weights = 1 - age / n
        return float(np.sum(recent * weights))

    def risk(self):
        return float(risk_index([max(self.bg[-1], 1.0)], 1)[2])

    def history(self):
        n = len(self.insulin)
        lbgi, hbgi = zip(*[risk_index([max(b, 1.0)], 1)[:2] for b in self.bg[1:]]) if n else ([], [])
        return pd.DataFrame({
            "time": self.times[1:],
            "patient": self.patient_name,
            "seed": self.seed,
            "bg": self.bg[1:],
            "cgm": self.cgm[1:],
            "cho": self.cho,
            "basal": self.basal,
            "bolus": self.bolus,
            "insulin": self.insulin,
            "lbgi": lbgi,
            "hbgi": hbgi,
        })
