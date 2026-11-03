import argparse
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from . import config
from .controllers import BasalBolus
from .features import FEATURE_NAMES, SCALER_PATH, Scaler
from .rollout import run_episode

SIM_PATH = config.DATA_DIR / "simulations.csv"
FEATURES_PATH = config.DATA_DIR / "features_train.csv"

SEED_OFFSET = {"train": 0, "val": 1000, "test": 2000}


def _job(args):
    warnings.filterwarnings("ignore")
    patient, seed = args
    df, feats = run_episode(BasalBolus(), patient, seed, with_features=True)
    df["split"] = config.split_of(patient)
    df["day"] = seed
    return df, np.array(feats)


def generate(days=3, workers=None):
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    jobs = [(p, SEED_OFFSET[config.split_of(p)] + d) for p in config.ALL_PATIENTS for d in range(days)]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        out = list(ex.map(_job, jobs))
    sims = pd.concat([df for df, _ in out], ignore_index=True)
    sims.to_csv(SIM_PATH, index=False)
    train_feats = np.concatenate([f for (df, f) in out if df["split"].iloc[0] == "train"])
    pd.DataFrame(train_feats, columns=FEATURE_NAMES).to_csv(FEATURES_PATH, index=False)
    Scaler().fit(train_feats).save(SCALER_PATH)
    return sims


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=3)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    sims = generate(args.days, args.workers)
    print(f"{len(sims)} lignes -> {SIM_PATH}")
    print(f"normalisation -> {SCALER_PATH}")


if __name__ == "__main__":
    main()
