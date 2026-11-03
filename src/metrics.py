import numpy as np
import pandas as pd

from . import config


def episode_metrics(df):
    bg = df["bg"].to_numpy()
    return {
        "tir": float(np.mean((bg >= config.BG_LOW) & (bg <= config.BG_HIGH)) * 100),
        "tbr": float(np.mean(bg < config.BG_LOW) * 100),
        "tbr_severe": float(np.mean(bg < config.BG_SEVERE_LOW) * 100),
        "tar": float(np.mean(bg > config.BG_HIGH) * 100),
        "mean_bg": float(bg.mean()),
        "cv": float(bg.std() / bg.mean() * 100),
        "lbgi": float(df["lbgi"].mean()),
        "hbgi": float(df["hbgi"].mean()),
        "risk": float((df["lbgi"] + df["hbgi"]).mean()),
        "insulin_total": float(df["insulin"].sum()),
        "hours": float(len(df) * config.SAMPLE_TIME / 60),
        "failure": bool(bg.min() < config.BG_TERMINATE_LOW or bg.max() > config.BG_TERMINATE_HIGH),
    }


def summarize(results):
    cols = ["tir", "tbr", "tbr_severe", "tar", "mean_bg", "cv", "lbgi", "hbgi", "risk", "failure"]
    agg = results.groupby("controller")[cols].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    return agg.sort_values("risk_mean")


def by_category(results):
    r = results.copy()
    r["category"] = r["patient"].str.split("#").str[0]
    return r.pivot_table(index="controller", columns="category", values="tir", aggfunc="mean")
