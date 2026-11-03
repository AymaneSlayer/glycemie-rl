# Contrôle glycémique du diabète de type 1 par apprentissage par renforcement

Projet du cours « Machine Learning et apprentissage automatique » (Prof. Y. Lamrani Alaoui), École Centrale Casablanca, 2026.

Un agent d'apprentissage par renforcement (PPO et SAC) apprend à piloter le débit d'insuline basale d'une pompe, toutes les 3 minutes, à partir des mesures d'un capteur de glucose en continu (CGM). Il est comparé à deux contrôleurs de référence sur des patients virtuels jamais vus pendant l'entraînement.

## Problématique

Dans le diabète de type 1, le pancréas ne produit plus d'insuline. Le patient doit doser lui-même l'insuline pour maintenir sa glycémie entre 70 et 180 mg/dL :

- trop d'insuline : hypoglycémie (< 70 mg/dL), danger immédiat ;
- pas assez : hyperglycémie (> 180 mg/dL), complications à long terme.

Le dosage est difficile car la glycémie dépend des repas, de l'heure et de chaque patient. La question du projet est : **un agent RL peut-il apprendre une stratégie de dosage plus sûre que les méthodes classiques ?**

## Données

Aucune donnée réelle de patient n'est utilisée. Les données sont générées par le simulateur **UVA/Padova** (implémentation open source [simglucose](https://github.com/jxx123/simglucose)), accepté par la FDA comme substitut aux essais précliniques. Il fournit 30 patients virtuels (10 enfants, 10 adolescents, 10 adultes), un modèle de capteur CGM bruité (Dexcom, une mesure toutes les 3 minutes) et une pompe (Insulet).

Les journées commencent à 6 h et durent 24 h. Les repas (petit-déjeuner, déjeuner, dîner et collations) sont tirés aléatoirement en heure et en quantité.

`python -m src.data` génère un jeu de référence (contrôleur basal-bolus, 3 journées par patient) dans `data/simulations.csv`, utilisé pour l'analyse exploratoire, et calcule la normalisation des entrées de l'agent dans `data/scaler.json`.

## Prévention des fuites de données

| Ensemble | Patients | Graines des journées |
|---|---|---|
| Entraînement | `#001` à `#006` de chaque catégorie (18) | 0 – 999 |
| Validation | `#007` et `#008` (6) | 1000 – 1002 |
| Test | `#009` et `#010` (6) | 2000 – 2004 |

- La séparation se fait **par patient**, avant tout prétraitement.
- La normalisation (moyenne, écart-type) est calculée uniquement sur les patients d'entraînement.
- Le meilleur modèle est choisi sur la validation. Le test ne sert qu'à l'évaluation finale.

## Formulation du problème RL

| Élément | Choix |
|---|---|
| Pas de décision | 3 minutes (480 par journée) |
| Approche | **Apprentissage résiduel** : l'agent corrige les décisions du PID au lieu de partir de zéro. Une action nulle reproduit exactement le PID |
| Observation (19 variables) | 12 dernières mesures CGM (36 min), pente CGM, insuline active (IOB), glucides actifs (COB) et repas en cours relatifs au patient, sortie du PID relative à la basale du patient, heure (sin, cos) |
| Action (2 dimensions) | `a₀ ∈ [-1, 1]` : basale = clip(basale PID + a₀ × basale du patient, 0, 3 × basale du patient). `a₁ ∈ [-1, 1]` : bolus de repas × 2^(0,6·a₁), soit entre ×0,66 et ×1,52 |
| Bolus de repas | formule classique (glucides / CR + correction), ajustée par l'agent : système **hybride**, comme les pompes commerciales |
| Récompense | 1 − risque de Kovatchev / 7,8, bornée à −5 : positive dans la cible 70–180, nulle à ses bords, négative au-delà. −100 et fin de l'épisode si la glycémie sort de [40, 500] mg/dL. Comme la récompense est positive dans la cible, arrêter tôt l'épisode n'est jamais avantageux |
| Algorithmes | PPO et SAC ([Stable-Baselines3](https://stable-baselines3.readthedocs.io)), réseaux 64×64 |

Une première version (v1), où l'agent pilotait seul la basale avec une récompense −risque/10, égalait le PID sans le dépasser. Ses modèles et résultats sont conservés dans `models/v1/` et `results/v1/`.

## Contrôleurs comparés

| Nom | Description |
|---|---|
| `basal_bolus` | Référence simple : basale fixe et bolus aux repas |
| `pid` | Référence classique : basale ajustée par un régulateur PID, coupure sous 80 mg/dL, bolus aux repas |
| `ppo` | Agent PPO (correction du PID et du bolus) |
| `sac` | Agent SAC (correction du PID et du bolus) |

## Métriques

Temps dans la cible 70–180 (TIR), temps en hypoglycémie < 70 et < 54, temps en hyperglycémie > 180, glycémie moyenne, coefficient de variation, indices de risque LBGI, HBGI et total, taux d'échec (glycémie hors de [40, 500]).

## Interprétabilité

`src/explain.py` calcule les valeurs **SHAP** (méthode par permutation) de la politique, pour ses deux sorties : la correction de basale et le multiplicateur de bolus. Il analyse aussi chaque épisode d'hypoglycémie des patients de test (durée, nadir, insuline et repas dans les 2 h précédentes).

## Structure

```
glycemie-rl/
├── src/
│   ├── config.py        paramètres, répartition des patients et des graines
│   ├── simulator.py     patient virtuel, capteur, pompe, repas, bolus, IOB/COB
│   ├── features.py      variables d'observation et normalisation
│   ├── controllers.py   basal-bolus, PID, agent RL
│   ├── metrics.py       métriques cliniques
│   ├── rollout.py       simulation d'une journée
│   ├── data.py          génération des données de référence
│   ├── env.py           environnement Gymnasium
│   ├── train.py         entraînement PPO / SAC
│   ├── evaluate.py      évaluation et figures
│   └── explain.py       SHAP et analyse des erreurs
├── notebooks/eda.ipynb  analyse exploratoire
├── app/streamlit_app.py démonstration interactive
├── tests/               tests automatiques
├── data/                données simulées et normalisation
├── models/              modèles entraînés et journaux d'entraînement
└── results/             tableaux et figures
```

## Installation

Python 3.11 ou plus récent.

```bash
git clone https://github.com/AymaneSlayer/glycemie-rl.git
cd glycemie-rl
pip install -r requirements.txt
```

Pour utiliser un GPU NVIDIA, installez la version CUDA de PyTorch en suivant [pytorch.org](https://pytorch.org/get-started/locally/). Sans GPU, ajoutez `--device cpu` à l'entraînement.

## Reproduction des résultats

```bash
python -m src.data
python -m src.train --algo ppo --timesteps 500000
python -m src.train --algo sac --timesteps 300000
python -m src.evaluate
python -m src.explain
```

| Étape | Sorties |
|---|---|
| `src.data` | `data/simulations.csv`, `data/features_train.csv`, `data/scaler.json` |
| `src.train` | `models/<algo>_best.zip` (meilleur sur validation), `models/<algo>_last.zip`, `models/<algo>_logs/` |
| `src.evaluate` | `results/summary_test.csv`, `results/metrics_test.csv`, `results/tir_by_category_test.csv`, `results/boxplot_test.png`, `results/trace_test_*.png`, `results/learning_curves.png` |
| `src.explain` | `results/shap_*.png`, `results/shap_importance_*.csv`, `results/hypo_events_test.*`, `results/hypo_summary_test.csv` |

Toutes les simulations sont déterministes : une même graine donne la même journée.

Options utiles : `--n-envs` (environnements en parallèle, 8 par défaut ; chacun utilise environ 600 Mo de RAM), `--eval-freq`, `--seed`, `--device`.

## Démonstration

```bash
streamlit run app/streamlit_app.py
```

On choisit un patient, une journée ou des repas personnalisés et les contrôleurs à comparer. L'application affiche les courbes de glycémie, la basale et les métriques de la journée.

## Tests

```bash
python -m pytest tests
```

## Résultats

Entraînement : PPO 500 000 pas (43 min), SAC 300 000 pas (61 min), sur une RTX 4060 Laptop avec 8 environnements en parallèle. Le meilleur modèle de chaque algorithme est celui qui obtient la meilleure récompense moyenne sur la validation.

### Validation (6 patients × 3 journées)

| Contrôleur | Dans la cible % | Hypo <70 % | Hypo <54 % | Hyper >180 % | Risque | Échecs % |
|---|---|---|---|---|---|---|
| PPO | **78,9** | 4,0 | 1,5 | 17,2 | **6,01** | 28 |
| SAC | 77,4 | 8,7 | 2,3 | **14,0** | 6,08 | 50 |
| PID | 78,5 | **3,3** | **1,2** | 18,2 | 6,20 | 28 |
| Basal-bolus | 77,0 | 6,5 | 2,9 | 16,6 | 6,64 | **22** |

Sur la validation, PPO est retenu : meilleur temps dans la cible et plus faible indice de risque.

### Test final (6 patients jamais vus × 5 journées)

| Contrôleur | Dans la cible % | Hypo <70 % | Hypo <54 % | Hyper >180 % | Risque | Échecs % |
|---|---|---|---|---|---|---|
| SAC | **89,6** | 6,1 | 1,2 | **4,4** | **3,10** | 17 |
| PID | 85,3 | **4,1** | **1,0** | 10,7 | 3,92 | 7 |
| Basal-bolus | 84,5 | 6,8 | 2,3 | 8,7 | 3,97 | **3** |
| PPO | 83,0 | 5,4 | 1,2 | 11,7 | 4,07 | **3** |

Un échec est une journée où la glycémie sort de [40, 500] mg/dL, ce qui arrête la simulation.

### Lecture

- Le modèle retenu sur la validation (PPO) ne fait pas mieux que le PID sur le test. SAC, moins bon sur la validation, est le meilleur sur le test : +4,3 points dans la cible et −21 % de risque par rapport au PID, mais avec plus d'hypoglycémies et d'échecs. Le classement change entre validation et test parce que chaque ensemble ne compte que 6 patients, très différents les uns des autres. Nous n'avons pas changé de modèle après avoir vu le test, pour ne pas créer de fuite.
- Quand un agent fait mieux que le PID, c'est surtout en réduisant les hyperglycémies, au prix d'un peu plus d'hypoglycémies.
- Les échecs se concentrent sur quelques patients (child#008 et adolescent#007 en validation, adult#009 en test) et touchent aussi les contrôleurs classiques : ils dépendent surtout du patient, pas seulement du contrôleur.
- La version v1 (agent seul, sans PID) égalait le PID sans le dépasser. La formulation résiduelle et la nouvelle récompense ont permis d'aller au-delà.

### Interprétabilité

- **SAC** corrige fortement le PID. Sa correction de basale dépend surtout des mesures CGM des 15 à 30 dernières minutes, de l'insuline active et de la pente du glucose, ce qui correspond au raisonnement d'un diabétologue. Son multiplicateur de bolus dépend surtout de l'heure de la journée et de la glycémie avant le repas.
- **PPO** reste très proche du PID : ses valeurs SHAP sont environ 7 fois plus faibles que celles de SAC. C'est ce qui explique qu'il fasse presque comme le PID.
- **Hypoglycémies du test** : chez tous les contrôleurs, elles touchent surtout les enfants et un adulte (adult#009). Moins de 16 % surviennent dans les 2 h suivant un bolus, et pendant l'hypo la basale est presque coupée : elles viennent d'insuline accumulée plus tôt, que le contrôleur ne peut plus retirer.

Figures : `results/shap_bar_*.png`, `results/shap_beeswarm_*.png`, `results/hypo_events_test.png`, `results/boxplot_test.png`, `results/trace_test_*.png`, `results/learning_curves.png`.

## Limites

- Les patients sont virtuels : les résultats ne se transfèrent pas directement à de vrais patients.
- Le bolus de repas suppose un comptage exact des glucides.
- Pas d'activité physique, de stress ni de variation de sensibilité à l'insuline au cours de la journée.
