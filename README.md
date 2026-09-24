# Retail Product Recognition System

A CNN-based computer vision system that recognizes retail products from
images — the kind of model behind a self-checkout camera or a smart
shelf. It is demonstrated here on a 5-item fruit catalog, but the
pipeline (data loading, training, evaluation, app) is written so the
catalog can grow to any number of product classes without touching the
core code.

## Why "fruits" is a retail product recognition problem

A supermarket's produce section is a real product-recognition use
case: no barcode on a loose apple or avocado, so the checkout camera
has to recognize the item from its image alone, the same way this
model does. That is exactly what this project simulates end-to-end,
including a mock checkout flow that turns recognized items into an
itemized receipt.

## Features

- CNN image classifier trained from scratch (Conv2D + BatchNorm +
  MaxPooling + Dropout) on the fruits-360 dataset.
- Top-3 predictions with confidence scores, not just a single label.
- Confidence threshold that flags uncertain predictions instead of
  silently showing a possibly-wrong item — important when a wrong
  guess means a wrong price.
- **Checkout simulation**: upload several product photos at once and
  get an itemized mock receipt with a total, like a self-checkout
  camera scanning a basket.
- Model report tab in the app: confusion matrix, training curves, and
  the full classification report, generated once and reused.
- Clean separation between training code (`src/`), the app (`app/`),
  and experimentation (`notebooks/`) — see Project Structure below.

## Project Structure

```
retail-product-recognition-system/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── training_colab.ipynb   # thin notebook: clones repo, calls src/
├── src/
│   ├── config.py              # paths, classes, prices, all constants
│   ├── data_loader.py         # download / load / split the dataset
│   ├── model.py                # CNN architecture
│   ├── train.py                # training entry point
│   └── evaluate.py             # metrics, confusion matrix, plots
├── app/
│   └── app.py                  # Streamlit app (3 tabs)
├── models/                     # trained model + class_names.json (gitignored)
└── outputs/                    # confusion_matrix.png, training_curves.png (gitignored)
```

## Environment: Colab for training, anywhere for the rest

Training is done on **Google Colab** to use its free GPU — the
dataset (fruits-360, ~90k images across all classes) is large enough
that CPU training is painfully slow. `notebooks/training_colab.ipynb`
handles that: it clones this repo, installs `requirements.txt`, and
calls `src/train.py` and `src/evaluate.py`.

Everything else — the Streamlit app, and even re-running training
locally if you have a GPU — works the same way outside Colab, because
none of `src/` or `app/` depends on Colab-specific paths, background
processes, or tunnels. That separation is the main fix from the
original single-notebook version of this project, where the dataset
path, the training loop, and the Streamlit app were mixed together in
one Colab-only file.

For deploying the app itself, prefer **Streamlit Community Cloud** or
**Hugging Face Spaces** over exposing a Colab-hosted tunnel — both are
built for hosting a Streamlit app long-term instead of a session that
dies when the Colab runtime disconnects.

## Setup & Installation

```bash
git clone https://github.com/<your-username>/retail-product-recognition-system.git
cd retail-product-recognition-system
pip install -r requirements.txt
```

## Training the model

Either open `notebooks/training_colab.ipynb` in Colab, or run locally:

```bash
python -m src.train
python -m src.evaluate
```

This produces:
- `models/product_recognition_model.keras`
- `models/class_names.json`
- `models/history.pkl`
- `outputs/confusion_matrix.png`
- `outputs/training_curves.png`
- `outputs/classification_report.txt`

## Running the app

```bash
streamlit run app/app.py
```

Open the local URL Streamlit prints (usually `http://localhost:8501`).

## Dataset

[fruits-360](https://www.kaggle.com/datasets/moltean/fruits) on
Kaggle, downloaded automatically via `kagglehub` — no manual download
or Kaggle API key setup needed for the public dataset.

Current catalog (5 classes):
- Apple Crimson Snow 1
- Avocado Black 2
- Dates 1
- Nectarine Flat 2
- Pear 10

## Roadmap

- [ ] Grad-CAM visualization in the app: highlight which pixels drove
  each prediction — useful for explaining misclassifications.
- [ ] Grow the catalog beyond 5 classes and retrain.
- [ ] Data augmentation (rotation/flip/zoom) to improve robustness to
  real camera angles instead of the clean dataset images.
- [ ] Swap the price table for a real product database / API.
- [ ] Add a `tests/` suite for `src/data_loader.py` and `src/model.py`.

## License

MIT — see LICENSE (add one from GitHub's template when you create the
repo, if you don't already have one).
