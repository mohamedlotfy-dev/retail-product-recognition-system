"""
Evaluation entry point for the Retail Product Recognition System.

Run after src/train.py has produced a saved model:

    python -m src.evaluate

Loads the trained model and the test set, then saves (rather than
just plt.show()-ing, which is useless outside a notebook):
  - outputs/confusion_matrix.png
  - outputs/training_curves.png   (if history.pkl exists)
  - outputs/classification_report.txt
"""

import os
import json
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless-safe: works on a server with no display
import matplotlib.pyplot as plt

import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from src import config
from src.data_loader import download_dataset, load_dataset


def main():
    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)

    print("Loading model...")
    model = tf.keras.models.load_model(config.MODEL_PATH)

    with open(config.CLASS_NAMES_PATH, "r") as f:
        class_names = json.load(f)

    print("Loading test data...")
    dataset_path = download_dataset()
    data = load_dataset(dataset_path)
    X_test, y_test = data["X_test"], data["y_test"]

    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    print("Test Loss:", test_loss)
    print("Test Accuracy:", test_accuracy)

    y_prob = model.predict(X_test)
    y_pred = np.argmax(y_prob, axis=1)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=class_names)
    print(report)

    with open(config.CLASSIFICATION_REPORT_PATH, "w") as f:
        f.write(f"Test Accuracy: {accuracy * 100:.2f}%\n\n")
        f.write(report)

    # --- Confusion matrix ---
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(config.CONFUSION_MATRIX_PATH)
    plt.close(fig)
    print(f"Saved {config.CONFUSION_MATRIX_PATH}")

    # --- Training curves (accuracy / loss) ---
    if os.path.exists(config.HISTORY_PATH):
        with open(config.HISTORY_PATH, "rb") as f:
            history = pickle.load(f)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].plot(history["accuracy"], label="train")
        axes[0].plot(history["val_accuracy"], label="val")
        axes[0].set_title("Accuracy")
        axes[0].set_xlabel("Epoch")
        axes[0].legend()

        axes[1].plot(history["loss"], label="train")
        axes[1].plot(history["val_loss"], label="val")
        axes[1].set_title("Loss")
        axes[1].set_xlabel("Epoch")
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(config.TRAINING_CURVES_PATH)
        plt.close(fig)
        print(f"Saved {config.TRAINING_CURVES_PATH}")
    else:
        print("No history.pkl found — skipping training curves plot.")


if __name__ == "__main__":
    main()
