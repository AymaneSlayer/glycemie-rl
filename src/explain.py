import argparse
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from . import config
from .controllers import RL
from .data import FEATURES_PATH
from .features import FEATURE_NAMES, SCALER_PATH, Scaler
from .rollout import run_episode


def _collect(args):
    warnings.filterwarnings("ignore")
    import torch
    torch.set_num_threads(1)
    model_path, patient, seed = args
    _, feats = run_episode(RL(Path(model_path)), patient, seed, with_features=True)
    return np.array(feats)


def policy_fn(model, scaler):
    def f(X):
        obs = scaler.transform(X)
        actions, _ = model.predict(obs, deterministic=True)
        return np.clip(1 + 2 * np.clip(actions[:, 0], -1, 1), 0, config.MAX_BASAL_FACTOR)
    return f


def explain_model(algo, n_explain, n_background, seeds, workers):
    model_path = config.MODELS_DIR / f"{algo}_best.zip"
    jobs = [(str(model_path), p, s) for p in config.TEST_PATIENTS for s in seeds]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        X_test = np.concatenate(list(ex.map(_collect, jobs)))
    rng = np.random.default_rng(0)
    X_test = X_test[rng.choice(len(X_test), min(n_explain, len(X_test)), replace=False)]
    X_train = pd.read_csv(FEATURES_PATH).to_numpy()
    background = X_train[rng.choice(len(X_train), n_background, replace=False)]
    ctrl = RL(model_path)
    f = policy_fn(ctrl.model, Scaler.load(SCALER_PATH))
    explainer = shap.Explainer(f, shap.maskers.Independent(background), feature_names=FEATURE_NAMES, algorithm="permutation")
    sv = explainer(X_test)
    out = config.RESULTS_DIR
    plt.figure()
    shap.plots.beeswarm(sv, max_display=12, show=False)
    plt.title(f"SHAP – {algo.upper()} (sortie : multiplicateur de basale)")
    plt.tight_layout()
    plt.savefig(out / f"shap_beeswarm_{algo}.png", dpi=150)
    plt.close()
    plt.figure()
    shap.plots.bar(sv, max_display=12, show=False)
    plt.title(f"Importance moyenne |SHAP| – {algo.upper()}")
    plt.tight_layout()
    plt.savefig(out / f"shap_bar_{algo}.png", dpi=150)
    plt.close()
    imp = pd.Series(np.abs(sv.values).mean(axis=0), index=FEATURE_NAMES).sort_values(ascending=False)
    imp.rename("mean_abs_shap").to_csv(out / f"shap_importance_{algo}.csv")
    return imp


def hypo_events(traces):
    rows = []
    for (ctrl, patient, seed), g in traces.groupby(["controller", "patient", "seed"]):
        g = g.reset_index(drop=True)
        low = (g["bg"] < config.BG_LOW).to_numpy()
        i = 0
        while i < len(g):
            if not low[i]:
                i += 1
                continue
            j = i
            while j < len(g) and low[j]:
                j += 1
            n2h = 120 // config.SAMPLE_TIME
            before = g.iloc[max(0, i - n2h): i]
            rows.append({
                "controller": ctrl, "patient": patient, "category": patient.split("#")[0], "seed": seed,
                "start": g.loc[i, "time"], "duration_min": (j - i) * config.SAMPLE_TIME,
                "nadir": float(g.loc[i:j - 1, "bg"].min()),
                "insulin_2h_before": float(before["insulin"].sum()),
                "bolus_2h_before": float(before["bolus"].sum()),
                "carbs_2h_before": float(before["cho"].sum()),
                "basal_during_mean": float(g.loc[i:j - 1, "basal"].mean() * 20),
                "after_meal_bolus": bool(before["bolus"].sum() > 0),
            })
            i = j
    return pd.DataFrame(rows)


def error_analysis(split):
    traces = pd.read_csv(config.RESULTS_DIR / f"traces_{split}.csv")
    ev = hypo_events(traces)
    out = config.RESULTS_DIR
    ev.to_csv(out / f"hypo_events_{split}.csv", index=False)
    if ev.empty:
        return ev
    summary = ev.groupby(["controller", "category"]).agg(
        events=("nadir", "size"), duration_mean=("duration_min", "mean"), nadir_mean=("nadir", "mean"),
        after_meal_bolus_pct=("after_meal_bolus", "mean"), basal_during_mean=("basal_during_mean", "mean"))
    summary["after_meal_bolus_pct"] *= 100
    summary.round(2).to_csv(out / f"hypo_summary_{split}.csv")
    fig, ax = plt.subplots(figsize=(8, 4))
    for ctrl, g in ev.groupby("controller"):
        ax.scatter(g["duration_min"], g["nadir"], label=ctrl, alpha=0.7)
    ax.axhline(config.BG_SEVERE_LOW, color="#d43f3a", ls="--", lw=1)
    ax.set_xlabel("durée de l'hypo (min)")
    ax.set_ylabel("glycémie minimale (mg/dL)")
    ax.set_title("Épisodes d'hypoglycémie (patients de test)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / f"hypo_events_{split}.png", dpi=150)
    plt.close(fig)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algos", nargs="+", default=["ppo", "sac"])
    parser.add_argument("--n-explain", type=int, default=500)
    parser.add_argument("--n-background", type=int, default=100)
    parser.add_argument("--days", type=int, default=2)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    seeds = list(config.TEST_SEEDS)[: args.days]
    for algo in args.algos:
        if not (config.MODELS_DIR / f"{algo}_best.zip").exists():
            print(f"{algo}: pas de modèle, ignoré")
            continue
        imp = explain_model(algo, args.n_explain, args.n_background, seeds, args.workers)
        print(f"\n{algo.upper()} – importance SHAP")
        print(imp.round(4).head(8))
    if (config.RESULTS_DIR / "traces_test.csv").exists():
        print("\nHypoglycémies par contrôleur")
        print(error_analysis("test"))


if __name__ == "__main__":
    main()
