"""
Data loading utilities for the Retail Product Recognition System.

This module owns everything related to getting pixels into NumPy
arrays: downloading the dataset, listing image files, loading them
into memory, and producing train/validation/test splits.

Design note: this file does NOT hard-code a Kaggle-only path. It
always resolves the dataset through kagglehub, so the exact same code
runs on Colab, Kaggle, or a local machine — no environment-specific
branches.
"""

import os
import json

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

from src import config


def download_dataset() -> str:
    """Download (or reuse the cached copy of) the fruits-360 dataset.

    Returns the root path that contains the ``Training`` and ``Test``
    folders. Works the same way on Colab, Kaggle, or locally.
    """
    import kagglehub

    path = kagglehub.dataset_download(config.KAGGLE_DATASET_SLUG)
    dataset_path = os.path.join(path, "fruits-360_100x100", "fruits-360")

    train_path = os.path.join(dataset_path, "Training")
    test_path = os.path.join(dataset_path, "Test")

    if not os.path.isdir(train_path) or not os.path.isdir(test_path):
        raise FileNotFoundError(
            f"Expected Training/Test folders under {dataset_path}, "
            "but they were not found. Check the kagglehub download."
        )

    return dataset_path


def get_image_files(folder: str) -> list[str]:
    """Return the image file names inside `folder` (empty list if missing)."""
    if not os.path.isdir(folder):
        return []
    return [
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]


def _load_split(split_path: str, classes: list[str], img_size: int):
    """Load every image for the given classes under one split folder."""
    class_to_index = {name: i for i, name in enumerate(classes)}

    images, labels = [], []
    for class_name in classes:
        class_path = os.path.join(split_path, class_name)
        for image_name in get_image_files(class_path):
            image_path = os.path.join(class_path, image_name)
            image = Image.open(image_path).convert("RGB")
            image = image.resize((img_size, img_size))
            images.append(np.array(image))
            labels.append(class_to_index[class_name])

    return np.array(images), np.array(labels)


def load_dataset(dataset_path: str):
    """Load train/val/test arrays, already split and normalized to [0, 1].

    Returns a dict with keys: X_train, y_train, X_val, y_val, X_test, y_test.
    """
    train_path = os.path.join(dataset_path, "Training")
    test_path = os.path.join(dataset_path, "Test")

    X_train_full, y_train_full = _load_split(
        train_path, config.SELECTED_CLASSES, config.IMG_SIZE
    )
    X_test, y_test = _load_split(
        test_path, config.SELECTED_CLASSES, config.IMG_SIZE
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=config.VALIDATION_SPLIT,
        random_state=config.RANDOM_SEED,
        stratify=y_train_full,
    )

    # Normalize after splitting, exactly once, in one place.
    X_train = X_train.astype("float32") / 255.0
    X_val = X_val.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
    }


def save_class_names() -> None:
    """Persist the class list next to the trained model."""
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    with open(config.CLASS_NAMES_PATH, "w") as f:
        json.dump(config.SELECTED_CLASSES, f)
