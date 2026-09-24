"""
CNN architecture for the Retail Product Recognition System.

Kept in its own module so training, evaluation, and any future
experiment script all build the exact same architecture from one
place — no risk of the notebook and the app silently drifting apart.
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization,
)

from src import config


def build_model() -> Sequential:
    """Build the CNN used to recognize retail products from images.

    Compared to a bare Conv-Pool-Conv-Pool-Dense stack, this adds
    BatchNormalization (faster, more stable convergence) and Dropout
    (reduces overfitting) — both cheap additions that matter a lot on
    a dataset with only a handful of classes and a few thousand images.
    """
    num_classes = len(config.SELECTED_CLASSES)

    model = Sequential([
        Conv2D(32, (3, 3), activation="relu",
               input_shape=(config.IMG_SIZE, config.IMG_SIZE, 3)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(64, (3, 3), activation="relu"),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(128, (3, 3), activation="relu"),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(num_classes, activation="softmax"),
    ])

    return model
