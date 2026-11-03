import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")

from src import config
from src.controllers import PID, BasalBolus, action_to_basal
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


def test_action_mapping():
    assert action_to_basal([0.0], 1.0) == pytest.approx(1.0)
    assert action_to_basal([-1.0], 1.0) == 0.0
    assert action_to_basal([1.0], 1.0) == pytest.approx(config.MAX_BASAL_FACTOR)


def test_features_shape():
    sim = Simulator("adult#001", seed=0)
    for _ in range(5):
        sim.step(sim.profile["basal"])
    f = raw_features(sim)
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
