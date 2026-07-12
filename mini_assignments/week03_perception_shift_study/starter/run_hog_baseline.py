"""HOG + LinearSVC baseline for the Week 3 perception study (Choice D).

This script is fully runnable on CPU. It:

1. Loads the synthesized training images from `data/train/` and `data/train_labels.json`.
2. Extracts HOG features for each training image.
3. Fits a `LinearSVC` classifier.
4. Loads the test images from `data/test/` and the test ID list.
5. Predicts a label and a probability-like confidence for each test image.
6. Writes a harness-compatible submission JSON.

Dependencies (all CPU, no GPU): numpy, scikit-learn, scikit-image, Pillow.

Usage:
    python starter/run_hog_baseline.py --out my_week03_submission.json

Then grade it:
    esp3201-eval grade --task week03_perception_shift --submission my_week03_submission.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.color import rgb2gray
from skimage.feature import hog
from sklearn.svm import LinearSVC


LAB_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = LAB_ROOT / "data"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"


def _hog_features(img_path: Path) -> np.ndarray:
    img = np.array(Image.open(img_path).convert("RGB"))
    gray = rgb2gray(img)
    return hog(
        gray,
        orientations=8,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        feature_vector=True,
    )


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="week03_submission_hog.json", help="output submission JSON path")
    args = parser.parse_args()

    train_labels_path = DATA_DIR / "train_labels.json"
    test_ids_path = DATA_DIR / "test_image_ids.json"
    if not train_labels_path.exists() or not test_ids_path.exists():
        raise SystemExit(
            "missing dataset files. run `python data/synthesize_dataset.py` first."
        )

    with train_labels_path.open() as f:
        train_layout = json.load(f)["items"]
    with test_ids_path.open() as f:
        test_ids = json.load(f)

    print(f"[hog_baseline] loading {len(train_layout)} training images...")
    X_train = np.stack([_hog_features(TRAIN_DIR / f"{item['image_id']}.png") for item in train_layout])
    y_train = np.array([item["label"] for item in train_layout])

    print("[hog_baseline] fitting LinearSVC...")
    clf = LinearSVC()
    clf.fit(X_train, y_train)

    print(f"[hog_baseline] predicting on {len(test_ids)} test images...")
    X_test = np.stack([_hog_features(TEST_DIR / f"{image_id}.png") for image_id in test_ids])
    decisions = clf.decision_function(X_test)
    confidences = _softmax(decisions)
    pred_indices = np.argmax(confidences, axis=1)
    classes = clf.classes_

    predictions = []
    for image_id, idx, conf_row in zip(test_ids, pred_indices, confidences):
        predictions.append(
            {
                "image_id": image_id,
                "predicted_label": str(classes[idx]),
                "confidence": float(round(conf_row[idx], 4)),
            }
        )

    submission = {"model_name": "hog_linear_baseline", "predictions": predictions}
    out_path = Path(args.out)
    with out_path.open("w") as f:
        json.dump(submission, f, indent=2)
    print(f"[hog_baseline] wrote {len(predictions)} predictions to {out_path}")
    print("[hog_baseline] next: esp3201-eval grade --task week03_perception_shift --submission", out_path)


if __name__ == "__main__":
    main()
