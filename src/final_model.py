# ==========================================================
# Amazon ML Challenge - Smart Product Pricing
# Text + Image + XGBoost Regression
# ==========================================================

import os
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO

import numpy as np
import pandas as pd
import requests
from PIL import Image
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import torch
import torchvision.models as models
import torchvision.transforms as transforms
import xgboost as xgb

warnings.filterwarnings("ignore")

# ==========================================================
# CONFIGURATION
# ==========================================================

# Use 1000 rows for a quick local test.
# Set to None to use the complete dataset.
SAMPLE_SIZE = 1000

RANDOM_STATE = 42
SKIP_IMAGE_EXTRACTION = False
USE_GPU = False
MAX_WORKERS = 8

NUM_BOOST_ROUNDS = 1000
EARLY_STOPPING_ROUNDS = 50


# ==========================================================
# PATHS
# ==========================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "results")

TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "test_out.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("AMAZON ML CHALLENGE - SMART PRODUCT PRICING")
print("=" * 70)
print(f"Project directory : {BASE_DIR}")
print(f"Data directory    : {DATA_DIR}")
print(f"Train path        : {TRAIN_PATH}")
print(f"Test path         : {TEST_PATH}")
print(f"Output path       : {OUTPUT_PATH}")
print("=" * 70)


# ==========================================================
# VALIDATE FILES
# ==========================================================

if not os.path.isfile(TRAIN_PATH):
    print("\nERROR: train.csv not found.")
    print("Place train.csv inside the project's data/ folder.")
    sys.exit(1)

if not os.path.isfile(TEST_PATH):
    print("\nERROR: test.csv not found.")
    print("Place test.csv inside the project's data/ folder.")
    sys.exit(1)


# ==========================================================
# LOAD DATA
# ==========================================================

print("\nLoading datasets...")

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

print("Original train shape:", train.shape)
print("Original test shape :", test.shape)

if "price" not in train.columns:
    print("\nERROR: 'price' column is missing from train.csv.")
    sys.exit(1)


# ==========================================================
# SAMPLE DATA
# ==========================================================

if SAMPLE_SIZE is not None:
    if len(train) > SAMPLE_SIZE:
        train = train.sample(
            n=SAMPLE_SIZE,
            random_state=RANDOM_STATE
        ).reset_index(drop=True)

    if len(test) > SAMPLE_SIZE:
        test = test.sample(
            n=SAMPLE_SIZE,
            random_state=RANDOM_STATE
        ).reset_index(drop=True)

print("Working train shape:", train.shape)
print("Working test shape :", test.shape)


# ==========================================================
# CLEAN DATA
# ==========================================================

for df in (train, test):
    if "catalog_content" not in df.columns:
        df["catalog_content"] = "missing"

    if "image_link" not in df.columns:
        df["image_link"] = "missing"

    df["catalog_content"] = (
        df["catalog_content"].fillna("missing").astype(str)
    )

    df["image_link"] = (
        df["image_link"].fillna("missing").astype(str)
    )

if "sample_id" in train.columns:
    train = train.drop_duplicates(
        subset=["sample_id"]
    ).reset_index(drop=True)

if "sample_id" in test.columns:
    test = test.drop_duplicates(
        subset=["sample_id"]
    ).reset_index(drop=True)

train["price"] = pd.to_numeric(
    train["price"], errors="coerce"
)

train = train.dropna(
    subset=["price"]
).reset_index(drop=True)

y = train["price"].values.astype(np.float32)

print("Final training rows:", len(train))


# ==========================================================
# SMAPE
# ==========================================================

def smape(y_true, y_pred):
    denominator = (
        np.abs(y_true) + np.abs(y_pred)
    ) / 2.0

    numerator = np.abs(
        y_true - y_pred
    )

    mask = denominator != 0

    if not np.any(mask):
        return 0.0

    return np.mean(
        numerator[mask] / denominator[mask]
    ) * 100.0


# ==========================================================
# TEXT FEATURES - TF-IDF
# ==========================================================

print("\n" + "=" * 70)
print("TEXT FEATURE EXTRACTION - TF-IDF")
print("=" * 70)

tfidf = TfidfVectorizer(
    max_features=5000,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=1
)

X_text_train = tfidf.fit_transform(
    train["catalog_content"]
)

X_text_test = tfidf.transform(
    test["catalog_content"]
)

print("TF-IDF train shape:", X_text_train.shape)
print("TF-IDF test shape :", X_text_test.shape)


# ==========================================================
# IMAGE FEATURES - RESNET50
# ==========================================================

if USE_GPU and torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print("\nPyTorch device:", DEVICE)


def build_resnet():
    print("\nLoading ResNet50...")

    try:
        weights = models.ResNet50_Weights.DEFAULT
        resnet = models.resnet50(weights=weights)
        image_transform = weights.transforms()
    except Exception:
        resnet = models.resnet50(pretrained=True)
        image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    # Remove the final classification layer.
    resnet = torch.nn.Sequential(
        *list(resnet.children())[:-1]
    )

    resnet.to(DEVICE)
    resnet.eval()

    print("ResNet50 loaded.")
    return resnet, image_transform


def extract_single_image_feature(
    url,
    resnet,
    image_transform,
    timeout=5
):
    feature = np.zeros(
        2048,
        dtype=np.float32
    )

    try:
        if not isinstance(url, str):
            return feature

        url = url.strip()

        if (
            not url
            or url.lower() == "missing"
            or not url.startswith("http")
        ):
            return feature

        response = requests.get(
            url,
            timeout=timeout
        )
        response.raise_for_status()

        image = Image.open(
            BytesIO(response.content)
        ).convert("RGB")

        image_tensor = (
            image_transform(image)
            .unsqueeze(0)
            .to(DEVICE)
        )

        with torch.no_grad():
            output = resnet(image_tensor)

        feature = (
            output.reshape(-1)
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    except Exception:
        pass

    return feature


def extract_image_features(
    df,
    resnet,
    image_transform,
    max_workers=8
):
    n = len(df)

    features = np.zeros(
        (n, 2048),
        dtype=np.float32
    )

    urls = (
        df["image_link"]
        .astype(str)
        .values
    )

    print(f"\nExtracting features for {n} images...")

    with ThreadPoolExecutor(
        max_workers=max_workers
    ) as executor:

        futures = {
            executor.submit(
                extract_single_image_feature,
                url,
                resnet,
                image_transform
            ): i
            for i, url in enumerate(urls)
        }

        for future in tqdm(
            as_completed(futures),
            total=n,
            desc="Images"
        ):
            i = futures[future]

            try:
                features[i] = future.result()
            except Exception:
                features[i] = np.zeros(
                    2048,
                    dtype=np.float32
                )

    return features


if SKIP_IMAGE_EXTRACTION:
    print("\nImage extraction disabled.")

    train_img_feat = np.zeros(
        (len(train), 2048),
        dtype=np.float32
    )

    test_img_feat = np.zeros(
        (len(test), 2048),
        dtype=np.float32
    )

else:
    resnet, image_transform = build_resnet()

    train_img_feat = extract_image_features(
        train,
        resnet,
        image_transform,
        MAX_WORKERS
    )

    test_img_feat = extract_image_features(
        test,
        resnet,
        image_transform,
        MAX_WORKERS
    )

print("Train image feature shape:", train_img_feat.shape)
print("Test image feature shape :", test_img_feat.shape)


# ==========================================================
# FEATURE FUSION
# ==========================================================

print("\n" + "=" * 70)
print("FEATURE FUSION")
print("=" * 70)

train_img_sparse = sparse.csr_matrix(
    train_img_feat
)

test_img_sparse = sparse.csr_matrix(
    test_img_feat
)

X_train = sparse.hstack(
    [X_text_train, train_img_sparse],
    format="csr"
)

X_test = sparse.hstack(
    [X_text_test, test_img_sparse],
    format="csr"
)

print("Combined X_train shape:", X_train.shape)
print("Combined X_test shape :", X_test.shape)


# ==========================================================
# TRAIN / VALIDATION SPLIT
# ==========================================================

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE
)

print("\nTraining rows  :", X_tr.shape[0])
print("Validation rows:", X_val.shape[0])


# ==========================================================
# XGBOOST
# ==========================================================

print("\n" + "=" * 70)
print("XGBOOST TRAINING")
print("=" * 70)

dtrain = xgb.DMatrix(
    X_tr,
    label=y_tr
)

dval = xgb.DMatrix(
    X_val,
    label=y_val
)

params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "learning_rate": 0.05,
    "max_depth": 7,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "nthread": -1,
    "seed": RANDOM_STATE,
    "tree_method": "hist"
}

if USE_GPU:
    params["device"] = "cuda"

evals = [
    (dtrain, "train"),
    (dval, "valid")
]

try:
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=NUM_BOOST_ROUNDS,
        evals=evals,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=100
    )
except TypeError:
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=NUM_BOOST_ROUNDS,
        evals=evals,
        verbose_eval=100
    )


# ==========================================================
# VALIDATION
# ==========================================================

print("\n" + "=" * 70)
print("VALIDATION")
print("=" * 70)

val_pred = model.predict(dval)

rmse = np.sqrt(
    mean_squared_error(
        y_val,
        val_pred
    )
)

val_smape = smape(
    y_val,
    val_pred
)

print(f"Validation RMSE : {rmse:.4f}")
print(f"Validation SMAPE: {val_smape:.4f}%")


# ==========================================================
# TEST PREDICTION
# ==========================================================

print("\n" + "=" * 70)
print("TEST PREDICTION")
print("=" * 70)

dtest = xgb.DMatrix(X_test)

test_pred = model.predict(dtest)

# Price cannot be negative.
test_pred = np.maximum(
    test_pred,
    0
)


# ==========================================================
# SUBMISSION
# ==========================================================

if "sample_id" in test.columns:
    sample_ids = test["sample_id"].values
else:
    sample_ids = np.arange(
        len(test_pred)
    )

submission = pd.DataFrame({
    "sample_id": sample_ids,
    "price": test_pred
})

submission.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\n" + "=" * 70)
print("SUCCESS")
print("=" * 70)
print(f"Output saved to: {OUTPUT_PATH}")
print("Output shape:", submission.shape)
print("\nFirst 10 predictions:")
print(submission.head(10))
print("=" * 70)
