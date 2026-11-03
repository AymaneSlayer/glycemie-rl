from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

CATEGORIES = ["adolescent", "adult", "child"]


def _names(ids):
    return [f"{c}#{i:03d}" for c in CATEGORIES for i in ids]


TRAIN_PATIENTS = _names(range(1, 7))
VAL_PATIENTS = _names(range(7, 9))
TEST_PATIENTS = _names(range(9, 11))
ALL_PATIENTS = TRAIN_PATIENTS + VAL_PATIENTS + TEST_PATIENTS

SPLITS = {"train": TRAIN_PATIENTS, "val": VAL_PATIENTS, "test": TEST_PATIENTS}

SAMPLE_TIME = 3
EPISODE_HOURS = 24
EPISODE_STEPS = EPISODE_HOURS * 60 // SAMPLE_TIME
START_HOUR = 6

HISTORY = 12
IOB_HOURS = 3
COB_HOURS = 2

BG_TARGET = 140
BG_LOW = 70
BG_SEVERE_LOW = 54
BG_HIGH = 180
BG_TERMINATE_LOW = 40
BG_TERMINATE_HIGH = 500

MAX_BASAL_FACTOR = 3.0
TERMINATION_PENALTY = 100.0

TRAIN_SEEDS = range(0, 1000)
VAL_SEEDS = range(1000, 1003)
TEST_SEEDS = range(2000, 2005)


def split_of(patient):
    for name, patients in SPLITS.items():
        if patient in patients:
            return name
    raise ValueError(patient)
