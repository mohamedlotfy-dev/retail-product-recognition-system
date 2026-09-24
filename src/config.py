"""
Central configuration for the Retail Product Recognition System.

Every other module (data_loader, model, train, evaluate, app) imports
its paths and constants from here instead of hard-coding them.
This is the single place you touch when the dataset, model file name,
or product catalog changes.
"""

import os

# --------------------------------------------------------------------------
# Project paths
# --------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

MODEL_PATH = os.path.join(MODELS_DIR, "product_recognition_model.keras")
CLASS_NAMES_PATH = os.path.join(MODELS_DIR, "class_names.json")
HISTORY_PATH = os.path.join(MODELS_DIR, "history.pkl")

CONFUSION_MATRIX_PATH = os.path.join(OUTPUTS_DIR, "confusion_matrix.png")
TRAINING_CURVES_PATH = os.path.join(OUTPUTS_DIR, "training_curves.png")
CLASSIFICATION_REPORT_PATH = os.path.join(OUTPUTS_DIR, "classification_report.txt")

# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------
# Downloaded once via kagglehub — see src/data_loader.py::download_dataset()
KAGGLE_DATASET_SLUG = "moltean/fruits"

IMG_SIZE = 100
BATCH_SIZE = 32
EPOCHS = 15
EARLY_STOPPING_PATIENCE = 3
VALIDATION_SPLIT = 0.20
RANDOM_SEED = 42

# The 5 product classes the current model recognizes.
# Add more classes here later to grow the catalog — the rest of the
# pipeline (data loading, training, app) reads from this list, so you
# never edit it in more than one place.
SELECTED_CLASSES = [
    "Apple Crimson Snow 1",
    "Avocado Black 2",
    "Dates 1",
    "Nectarine Flat 2",
    "Pear 10",
]

# --------------------------------------------------------------------------
# Retail simulation
# --------------------------------------------------------------------------
# Mock shelf prices (EGP) — powers the "checkout simulation" feature in
# the app, which is what turns this from a plain classifier demo into a
# stand-in for a real self-checkout / smart-shelf recognition system.
PRODUCT_PRICES = {
    "Apple Crimson Snow 1": 3.50,
    "Avocado Black 2": 9.00,
    "Dates 1": 12.00,
    "Nectarine Flat 2": 4.25,
    "Pear 10": 3.75,
}

# Below this confidence, the app flags the prediction as "not sure"
# instead of silently showing a possibly-wrong label. This matters a lot
# in a retail context — a wrong SKU means a wrong price on the receipt.
CONFIDENCE_THRESHOLD = 0.60
