import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src import config
from src.controllers import baselines, load_rl_models
from src.metrics import episode_metrics
from src.rollout import run_episode

COLORS = {"basal_bolus": "#8a96a3", "pid": "#3c6e8f", "ppo": "#1f9d6b", "sac": "#d9891c"}

st.set_page_config(page_title="Contrôle glycémique par RL", layout="wide")
st.title("Contrôle glycémique du diabète de type 1 par apprentissage par renforcement")
st.caption("Simulateur UVA/Padova (simglucose). Les patients de test n'ont jamais été vus pendant l'entraînement.")


@st.cache_resource
def controllers():
    return {c.name: c for c in baselines() + load_rl_models()}


@st.cache_data(show_spinner=False)
def simulate(name, patient, seed, meals):
    df = run_episode(controllers()[name], patient, seed, meals=list(meals) if meals else None)
    return df, episode_metrics(df)


ctrls = controllers()
with st.sidebar:
    st.header("Paramètres")
    split = st.radio("Groupe de patients", ["test", "val", "train"], horizontal=True)
    patient = st.selectbox("Patient virtuel", config.SPLITS[split])
    seed = st.number_input("Graine de la journée", 0, 10_000, list(config.TEST_SEEDS)[0])
    chosen = st.multiselect("Contrôleurs", list(ctrls), default=list(ctrls))
    custom = st.checkbox("Choisir les repas")
    meals = None
    if custom:
        rows = []
        for label, h, g in [("Petit-déjeuner", 7.5, 45), ("Déjeuner", 12.5, 70), ("Dîner", 19.5, 80)]:
            c1, c2 = st.columns(2)
            hour = c1.number_input(f"{label} (heure)", 6.0, 29.0, h, 0.5, key=f"h{label}")
            grams = c2.number_input(f"{label} (g)", 0, 200, g, 5, key=f"g{label}")
            if grams > 0:
                rows.append((float(hour), float(grams)))
        meals = tuple(rows)
    if not any(n in ctrls for n in ["ppo", "sac"]):
        st.warning("Aucun modèle RL trouvé dans models/. Lancez python -m src.train.")

if not chosen:
    st.info("Choisissez au moins un contrôleur.")
    st.stop()

results = {}
with st.spinner("Simulation de 24 h en cours…"):
    for name in chosen:
        results[name] = simulate(name, patient, int(seed), meals)

fig, (ax, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
ax.axhspan(config.BG_LOW, config.BG_HIGH, color="#1f9d6b", alpha=0.1, label="cible 70–180")
for name, (df, _) in results.items():
    t = pd.to_datetime(df["time"])
    ax.plot(t, df["bg"], label=name, color=COLORS.get(name), lw=1.8)
    ax2.plot(t, df["basal"] * 20, color=COLORS.get(name), lw=1)
first = next(iter(results.values()))[0]
for _, r in first[first.cho > 0].iterrows():
    ax.axvline(pd.to_datetime(r["time"]), color="#d9891c", ls=":", alpha=0.6)
ax.set_ylabel("glycémie (mg/dL)")
ax.legend(loc="upper right")
ax.grid(alpha=0.3)
ax2.set_ylabel("basale (U/h)")
ax2.grid(alpha=0.3)
fig.tight_layout()
st.pyplot(fig)

table = pd.DataFrame({n: m for n, (_, m) in results.items()}).T
table = table[["tir", "tbr", "tbr_severe", "tar", "mean_bg", "cv", "risk", "insulin_total", "failure"]]
table.columns = ["Dans la cible %", "Hypo <70 %", "Hypo <54 %", "Hyper >180 %", "Glycémie moy.", "CV %", "Indice de risque", "Insuline totale (U)", "Échec"]
st.subheader("Métriques de la journée")
st.dataframe(table.style.format(precision=1), use_container_width=True)

summary = config.RESULTS_DIR / "summary_test.csv"
if summary.exists():
    st.subheader("Résultats globaux sur les patients de test")
    st.dataframe(pd.read_csv(summary, index_col=0).round(2), use_container_width=True)
    for img in ["boxplot_test.png", "learning_curves.png", "shap_bar_ppo_basal.png", "shap_bar_ppo_bolus.png", "shap_bar_sac_basal.png", "shap_bar_sac_bolus.png"]:
        p = config.RESULTS_DIR / img
        if p.exists():
            st.image(str(p))
