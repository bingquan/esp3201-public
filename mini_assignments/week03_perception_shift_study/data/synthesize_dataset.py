"""Synthesize the Week 3 perception dataset.

Generates a small RGB image dataset with four classes (`cat`, `dog`, `car`,
`person`) and four conditions (`nominal`, `low_light`, `blur`, `ood`). Class is
encoded by a deterministic geometric pattern; condition is encoded by a
post-processing transform applied to the class pattern.

The test split's image IDs match the harness ground truth in
`evaluation_harness/esp3201_eval/data/week03_groundtruth.json`. The training
split has its own IDs.

This dataset is intentionally synthetic. Students are not expected to learn
real visual recognition from it. The course is about evaluating perception
under shift, not training the model.

Usage:
    python data/synthesize_dataset.py
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter


LABELS = ["cat", "dog", "car", "person"]
IMG_SIZE = 64

OUT_DIR = Path(__file__).resolve().parent
TRAIN_DIR = OUT_DIR / "train"
TEST_DIR = OUT_DIR / "test"


def _class_canvas(label: str, seed: int) -> Image.Image:
    """Draw a class-distinctive pattern on a fresh canvas.

    Each class has a distinctive arrangement of primitive shapes. The exact
    positions and sizes vary with the seed so that within-class examples
    look different from each other.
    """
    rng = random.Random(seed)
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), color=(220, 220, 220))
    draw = ImageDraw.Draw(img)

    if label == "cat":
        cx = rng.randint(20, 44)
        cy = rng.randint(14, 30)
        r = rng.randint(8, 12)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(80, 60, 40))
    elif label == "dog":
        for dx in (-12, 12):
            cx = 32 + dx + rng.randint(-2, 2)
            cy = rng.randint(20, 32)
            r = rng.randint(6, 10)
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(120, 80, 50))
    elif label == "car":
        w = rng.randint(36, 50)
        h = rng.randint(12, 18)
        x0 = (IMG_SIZE - w) // 2 + rng.randint(-3, 3)
        y0 = rng.randint(40, 48)
        draw.rectangle([x0, y0, x0 + w, y0 + h], fill=(50, 80, 160))
    elif label == "person":
        w = rng.randint(10, 14)
        h = rng.randint(36, 46)
        x0 = (IMG_SIZE - w) // 2 + rng.randint(-3, 3)
        y0 = (IMG_SIZE - h) // 2 + rng.randint(-3, 3)
        draw.rectangle([x0, y0, x0 + w, y0 + h], fill=(180, 60, 60))
    else:
        raise ValueError(f"unknown label {label!r}")
    return img


def _apply_condition(img: Image.Image, condition: str, seed: int) -> Image.Image:
    rng = random.Random(seed)
    if condition == "nominal":
        return img
    if condition == "low_light":
        return Image.eval(img, lambda px: int(px * 0.3))
    if condition == "blur":
        return img.filter(ImageFilter.GaussianBlur(radius=2.0))
    if condition == "ood":
        # Out-of-distribution: rotate canvas plus tint the background.
        bg_color = (rng.randint(0, 80), rng.randint(0, 80), rng.randint(0, 80))
        rotated = img.rotate(rng.randint(20, 60), fillcolor=bg_color)
        return rotated
    raise ValueError(f"unknown condition {condition!r}")


def _emit(image_id: str, label: str, condition: str, out_dir: Path, seed: int) -> None:
    canvas = _class_canvas(label, seed=seed)
    transformed = _apply_condition(canvas, condition=condition, seed=seed + 1)
    out_dir.mkdir(parents=True, exist_ok=True)
    transformed.save(out_dir / f"{image_id}.png")


def _test_layout() -> list[dict]:
    """Test layout matches the harness ground truth's image_id + condition assignments.

    True labels are NOT stored in the public layout (they live only in the
    harness ground-truth file). The synthesizer needs the label to draw the
    image, so we re-derive it deterministically from the harness file when
    the script is run. Students can also see the labels by reading the
    harness data file, which is fine for the synthetic-development workflow;
    instructors should replace both the synthetic data and the harness
    ground truth with hidden data for real grading.
    """
    gt_path = Path(__file__).resolve().parents[3] / "evaluation_harness" / "esp3201_eval" / "data" / "week03_groundtruth.json"
    with gt_path.open() as f:
        gt = json.load(f)
    return [
        {"image_id": item["image_id"], "condition": item["condition"], "label": item["true_label"]}
        for item in gt["items"]
    ]


def _train_layout(items_per_class: int = 40) -> list[dict]:
    """Training split: nominal-condition only, IDs of the form img_train_<class>_<idx>."""
    out: list[dict] = []
    for label in LABELS:
        for i in range(items_per_class):
            out.append(
                {
                    "image_id": f"img_train_{label}_{i:03d}",
                    "condition": "nominal",
                    "label": label,
                }
            )
    return out


def main() -> None:
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)

    test_items = _test_layout()
    for idx, item in enumerate(test_items):
        _emit(item["image_id"], item["label"], item["condition"], TEST_DIR, seed=10_000 + idx)

    train_items = _train_layout()
    for idx, item in enumerate(train_items):
        _emit(item["image_id"], item["label"], item["condition"], TRAIN_DIR, seed=idx)

    with (OUT_DIR / "train_labels.json").open("w") as f:
        json.dump({"items": train_items}, f, indent=2)
    with (OUT_DIR / "test_image_ids.json").open("w") as f:
        json.dump([item["image_id"] for item in test_items], f, indent=2)

    print(f"wrote {len(train_items)} training images to {TRAIN_DIR}")
    print(f"wrote {len(test_items)} test images to {TEST_DIR}")
    print(f"wrote train_labels.json and test_image_ids.json to {OUT_DIR}")


if __name__ == "__main__":
    main()
