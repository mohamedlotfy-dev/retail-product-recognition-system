"""
Training entry point for the Retail Product Recognition System.

Run this as a script (locally, on Colab, or on Kaggle — it makes no
assumption about the environment):

    python -m src.train

It downloads the dataset, builds the model, trains it with early
stopping, and saves the model + class names + training history under
models/. Evaluation lives separately in src/evaluate.py.
"""

import os
import pickle

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from src import config
from src.data_loader import download_dataset, load_dataset, save_class_names
from src.model import build_model


def main():
    tf.random.set_seed(config.RANDOM_SEED)

    print("Downloading / locating dataset...")
    dataset_path = download_dataset()

    print("Loading images into memory...")
    data = load_dataset(dataset_path)
    print("Shapes -> train:", data["X_train"].shape,
          "val:", data["X_val"].shape,
          "test:", data["X_test"].shape)

    print("Building model...")
    model = build_model()
    model.summary()

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=config.EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
    )

    print("Training...")
    history = model.fit(
        data["X_train"], data["y_train"],
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        validation_data=(data["X_val"], data["y_val"]),
        callbacks=[early_stop],
    )

    os.makedirs(config.MODELS_DIR, exist_ok=True)
    model.save(config.MODEL_PATH)
    save_class_names()

    with open(config.HISTORY_PATH, "wb") as f:
        pickle.dump(history.history, f)

    print(f"Saved model to {config.MODEL_PATH}")
    print(f"Saved class names to {config.CLASS_NAMES_PATH}")
    print(f"Saved training history to {config.HISTORY_PATH}")


if __name__ == "__main__":
    main()
