import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")

from src import config
from src.controllers import PID, BasalBolus, apply_action
from src.env import GlucoseEnv, reward_fn
from src.features import FEATURE_NAMES, Scaler, raw_features
from src.metrics import episode_metrics
from src.rollout import run_episode
from src.simulator import Simulator


def test_splits_are_disjoint():
    train, val, test = map(set, (config.TRAIN_PATIENTS, config.VAL_PATIENTS, config.TEST_PATIENTS))
    assert not train & val and not train & test and not val & test
    assert len(train | val | test) == 30


def test_seeds_are_disjoint():
    assert not set(config.TRAIN_SEEDS) & set(config.TEST_SEEDS)
    assert not set(config.VAL_SEEDS) & set(config.TEST_SEEDS)


def test_zero_action_reproduces_pid():
    assert apply_action([0.0, 0.0], 0.7, 1.0) == pytest.approx((0.7, 1.0))
    assert apply_action([-1.0, 0.0], 0.5, 1.0)[0] == 0.0
    assert apply_action([1.0, 0.0], 2.9, 1.0)[0] == pytest.approx(config.MAX_BASAL_FACTOR)
    lo, hi = apply_action([0, -1], 1, 1)[1], apply_action([0, 1], 1, 1)[1]
    assert lo == pytest.approx(2 ** -config.BOLUS_LOG2_RANGE)
    assert hi == pytest.approx(2 ** config.BOLUS_LOG2_RANGE)


def test_env_with_zero_action_matches_pid():
    env = GlucoseEnv(["adult#003"], [7])
    env.reset(seed=0)
    bg_env = []
    done = False
    while not done:
        _, _, term, trunc, info = env.step(np.zeros(2, dtype=np.float32))
        bg_env.append(info["bg"])
        done = term or trunc
    bg_pid = run_episode(PID(), "adult#003", 7)["bg"].to_numpy()
    np.testing.assert_allclose(bg_env, bg_pid)


def test_reward_sign():
    sim = Simulator("adult#001", seed=0)
    sim.bg[-1] = 120
    sim.terminated = False
    assert reward_fn(sim) > 0
    sim.bg[-1] = 45
    assert reward_fn(sim) < 0


def test_features_shape():
    sim = Simulator("adult#001", seed=0)
    for _ in range(5):
        sim.step(sim.profile["basal"])
    f = raw_features(sim, sim.profile["basal"])
    assert f.shape == (len(FEATURE_NAMES),)
    assert np.isfinite(f).all()


def test_scaler_roundtrip(tmp_path):
    X = np.random.default_rng(0).normal(size=(100, len(FEATURE_NAMES)))
    s = Scaler().fit(X)
    s.save(tmp_path / "s.json")
    s2 = Scaler.load(tmp_path / "s.json")
    np.testing.assert_allclose(s2.inverse(s2.transform(X)), X, atol=1e-5)


def test_simulation_is_deterministic():
    a = run_episode(BasalBolus(), "adult#002", 5)
    b = run_episode(BasalBolus(), "adult#002", 5)
    np.testing.assert_allclose(a["bg"], b["bg"])


@pytest.mark.parametrize("ctrl", [BasalBolus(), PID()])
def test_baselines_run_one_day(ctrl):
    df = run_episode(ctrl, "adolescent#001", 0)
    m = episode_metrics(df)
    assert len(df) == config.EPISODE_STEPS
    assert 0 <= m["tir"] <= 100
    assert m["tir"] > 50


def test_custom_meals():
    sim_df = run_episode(BasalBolus(), "adult#001", 0, meals=[(8, 50), (13, 70)])
    assert sim_df["cho"].sum() == pytest.approx(120, rel=0.05)
    assert (sim_df["bolus"] > 0).sum() == 2
