import argparse
import warnings
from concurrent.futures import ProcessPoolExecutor

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config
from .controllers import RL, baselines
from .metrics import by_category, episode_metrics, summarize
from .rollout import run_episode

COLORS = {"basal_bolus": "#8a96a3", "pid": "#3c6e8f", "ppo": "#1f9d6b", "sac": "#d9891c"}


def controller_specs():
    specs = [("baseline", c.name) for c in baselines()]
    specs += [("rl", str(p)) for p in sorted(config.MODELS_DIR.glob("*_best.zip"))]
    return specs


def build(spec):
    kind, ref = spec
    if kind == "baseline":
        return next(c for c in baselines() if c.name == ref)
    from pathlib import Path
    return RL(Path(ref), name=Path(ref).stem.replace("_best", ""))


def _job(args):
    warnings.filterwarnings("ignore")
    import torch
    torch.set_num_threads(1)
    spec, patient, seed = args
    ctrl = build(spec)
    df = run_episode(ctrl, patient, seed)
    m = episode_metrics(df)
    m.update(controller=ctrl.name, patient=patient, seed=seed)
    return df, m


def evaluate(patients, seeds, workers=None):
    jobs = [(s, p, d) for s in controller_specs() for p in patients for d in seeds]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        out = list(ex.map(_job, jobs))
    traces = pd.concat([df for df, _ in out], ignore_index=True)
    results = pd.DataFrame([m for _, m in out])
    return traces, results


def plot_boxplot(results, path):
    order = list(summarize(results).index)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, col, title in zip(axes, ["tir", "tbr", "tar"], ["Temps dans la cible (%)", "Temps en hypo < 70 (%)", "Temps en hyper > 180 (%)"]):
        data = [results.loc[results.controller == c, col] for c in order]
        bp = ax.boxplot(data, patch_artist=True, tick_labels=order)
        for patch, c in zip(bp["boxes"], order):
            patch.set_facecolor(COLORS.get(c, "#999"))
            patch.set_alpha(0.7)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_traces(traces, patient, seed, path):
    sub = traces[(traces.patient == patient) & (traces.seed == seed)]
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax.axhspan(config.BG_LOW, config.BG_HIGH, color="#1f9d6b", alpha=0.1)
    for name, g in sub.groupby("controller"):
        t = pd.to_datetime(g["time"])
        ax.plot(t, g["bg"], label=name, color=COLORS.get(name), lw=1.6)
        ax2.plot(t, g["basal"] * 20, color=COLORS.get(name), lw=1)
    meals = sub[(sub.controller == sub.controller.iloc[0]) & (sub.cho > 0)]
    for _, r in meals.iterrows():
        ax.axvline(pd.to_datetime(r["time"]), color="#d9891c", ls=":", alpha=0.6)
    ax.set_ylabel("glycémie (mg/dL)")
    ax.set_title(f"{patient} (test), journée {seed}")
    ax.legend()
    ax.grid(alpha=0.3)
    ax2.set_ylabel("basale (U/h)")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_learning_curves(path):
    fig, ax = plt.subplots(figsize=(8, 4))
    found = False
    for algo in ["ppo", "sac"]:
        f = config.MODELS_DIR / f"{algo}_logs" / "evaluations.npz"
        if not f.exists():
            continue
        d = np.load(f)
        r = d["results"]
        ax.plot(d["timesteps"], r.mean(axis=1), label=algo.upper(), color=COLORS[algo])
        ax.fill_between(d["timesteps"], r.mean(axis=1) - r.std(axis=1), r.mean(axis=1) + r.std(axis=1), color=COLORS[algo], alpha=0.15)
        found = True
    if not found:
        plt.close(fig)
        return
    ax.set_xlabel("pas d'entraînement")
    ax.set_ylabel("récompense moyenne (validation)")
    ax.set_title("Courbes d'apprentissage")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--days", type=int, default=len(config.TEST_SEEDS))
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()
    seeds = list(config.TEST_SEEDS if args.split == "test" else config.VAL_SEEDS)[: args.days]
    patients = config.SPLITS[args.split]
    out = config.RESULTS_DIR
    out.mkdir(parents=True, exist_ok=True)
    traces, results = evaluate(patients, seeds, args.workers)
    traces.to_csv(out / f"traces_{args.split}.csv", index=False)
    results.to_csv(out / f"metrics_{args.split}.csv", index=False)
    summary = summarize(results)
    summary.round(2).to_csv(out / f"summary_{args.split}.csv")
    by_category(results).round(1).to_csv(out / f"tir_by_category_{args.split}.csv")
    plot_boxplot(results, out / f"boxplot_{args.split}.png")
    for p in [patients[0], patients[len(patients) // 2], patients[-1]]:
        plot_traces(traces, p, seeds[0], out / f"trace_{args.split}_{p.replace('#', '')}.png")
    plot_learning_curves(out / "learning_curves.png")
    pd.set_option("display.width", 160)
    print(summary[["tir_mean", "tbr_mean", "tbr_severe_mean", "tar_mean", "mean_bg_mean", "risk_mean", "failure_mean"]].round(2))
    print(by_category(results).round(1))


if __name__ == "__main__":
    main()
