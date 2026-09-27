# Amazon ML Challenge – Smart Product Pricing

A multimodal machine learning pipeline for predicting Amazon product prices from product catalog text and product images.

## Project Overview

This project treats product price prediction as a multimodal regression problem. It combines:

- **NLP:** TF-IDF features from `catalog_content`
- **Computer Vision:** 2048-dimensional ResNet50 image embeddings
- **Feature Fusion:** sparse text + image representation
- **Regression:** XGBoost
- **Engineering:** parallel image downloading and feature extraction
- **Evaluation:** RMSE and SMAPE

## Architecture

```text
                    Product Data
                         |
              +----------+----------+
              |                     |
              v                     v
       Catalog Content          Image URL
              |                     |
              v                     v
          TF-IDF                 ResNet50
              |                 2048-D Embedding
              |                     |
              +----------+----------+
                         |
                         v
                 Feature Fusion
                         |
                         v
                     XGBoost
                    Regression
                         |
                         v
                  Predicted Price
```

## Repository Structure

```text
amazon-ml-price-prediction/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/
│   └── final_model.py
│
├── data/
│   └── README.md
│
├── results/
│   └── README.md
│
└── notebooks/
    └── README.md
```

## Dataset

The project is based on the Amazon ML Challenge / Smart Product Pricing dataset.

The expected files are:

```text
data/
├── train.csv
└── test.csv
```

The dataset files are intentionally excluded from Git because they can be large. Download the permitted challenge dataset separately and place the files inside `data/`.

Expected training columns include:

- `sample_id`
- `catalog_content`
- `image_link`
- `price`

The test set contains the input fields and does not contain the target `price`.

## Feature Engineering

### 1. Text Features

TF-IDF is applied to `catalog_content`.

Configuration:

- Maximum features: 5,000
- N-gram range: `(1, 2)`
- English stop-word removal

### 2. Image Features

A pretrained ResNet50 is used as a feature extractor. The final classification layer is removed, producing a **2048-dimensional visual representation** for each product image.

If an image cannot be downloaded or processed, a zero vector is used so the pipeline can continue.

### 3. Multimodal Fusion

Text and image features are converted to sparse matrices and horizontally stacked:

```text
TF-IDF features + ResNet50 embeddings
```

The combined representation is used by XGBoost.

## Model

XGBoost regression:

- Objective: `reg:squarederror`
- Learning rate: `0.05`
- Maximum depth: `7`
- Subsample: `0.8`
- Column sampling: `0.8`
- Tree method: `hist`
- Early stopping: `50` rounds

## Validation

The training set is split into:

- 80% training
- 20% validation

The implementation reports:

- RMSE for local regression validation
- SMAPE to align with the challenge evaluation metric

## Parallel Image Processing

Image downloading and ResNet50 feature extraction use `ThreadPoolExecutor` to process multiple image URLs concurrently.

This is useful because image retrieval is I/O-heavy and can otherwise become a major bottleneck.

## Installation

Create a virtual environment:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

Place:

```text
data/train.csv
data/test.csv
```

Then run:

```bash
python src/final_model.py
```

By default, the script uses 1,000 rows for a quick test.

To run on the complete dataset, change:

```python
SAMPLE_SIZE = 1000
```

to:

```python
SAMPLE_SIZE = None
```

The generated prediction file is:

```text
results/test_out.csv
```

with columns:

```text
sample_id,price
```

## Important Notes

- Do not commit API keys, passwords, or `.env` files.
- Do not commit the large raw dataset unless its license and repository limits allow it.
- Do not claim a model performance number unless it was actually obtained from your run.
- The repository currently represents a TF-IDF + ResNet50 + XGBoost baseline.

## Tech Stack

Python • Pandas • NumPy • Scikit-learn • PyTorch • Torchvision • ResNet50 • XGBoost • SciPy • Pillow • Requests

## Possible Future Improvements

- Sentence-Transformer text embeddings
- CLIP-based multimodal embeddings
- Structured extraction of product quantity, pack size, brand and specifications
- Log-price transformation
- Hyperparameter optimization
- LightGBM / ensemble models
- Better missing-image handling
- Cross-validation
- Challenge-specific SMAPE optimization
