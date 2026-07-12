"""Template for running a pretrained perception model on the Week 3 test set.

This file is a template. Copy it to a new file (`run_my_model.py`) and fill in
the marked sections for the curated model you chose. The four supported
choices are documented at the bottom of this file.

The template handles:

- loading test image IDs
- iterating over test images
- writing a harness-compatible submission JSON

You fill in:

- model loading
- per-image prediction logic
- mapping the model output into (predicted_label, confidence)

Dependencies vary by choice. See the per-choice notes at the bottom.

Usage (after filling in):
    python starter/run_my_model.py --out my_week03_submission.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image


LAB_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = LAB_ROOT / "data"
TEST_DIR = DATA_DIR / "test"

LABELS = ["cat", "dog", "car", "person"]

# ============================================================
# FILL IN: set this to the exact model_name string from model_list.md
# Choices: "dinov2_linear_head", "siglip_zero_shot", "grounding_dino_class_presence"
# ============================================================
MODEL_NAME = "REPLACE_ME"


def load_model():
    """Load the pretrained model and any auxiliary state.

    FILL IN: replace the body of this function with your model-loading code.
    Return any object(s) you will need in `predict_one`.
    See per-choice notes at the bottom of this file.
    """
    raise NotImplementedError("fill in load_model() for your chosen model")


def predict_one(model, image: Image.Image) -> tuple[str, float]:
    """Predict one image. Return (predicted_label, confidence).

    FILL IN: replace the body of this function with your per-image prediction
    logic. The returned label must be one of LABELS. The returned confidence
    must be a float in [0, 1].

    See per-choice notes at the bottom of this file.
    """
    raise NotImplementedError("fill in predict_one() for your chosen model")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="week03_submission.json", help="output submission JSON path")
    args = parser.parse_args()

    test_ids_path = DATA_DIR / "test_image_ids.json"
    if not test_ids_path.exists():
        raise SystemExit("missing test_image_ids.json. run `python data/synthesize_dataset.py` first.")
    with test_ids_path.open() as f:
        test_ids = json.load(f)

    print(f"[template] loading model...")
    model = load_model()

    print(f"[template] predicting on {len(test_ids)} test images...")
    predictions = []
    for image_id in test_ids:
        img = Image.open(TEST_DIR / f"{image_id}.png").convert("RGB")
        label, conf = predict_one(model, img)
        if label not in LABELS:
            raise SystemExit(f"predict_one returned label {label!r} not in {LABELS}")
        if not 0.0 <= conf <= 1.0:
            raise SystemExit(f"predict_one returned confidence {conf} out of [0, 1] for {image_id}")
        predictions.append({"image_id": image_id, "predicted_label": label, "confidence": float(round(conf, 4))})

    submission = {"model_name": MODEL_NAME, "predictions": predictions}
    out_path = Path(args.out)
    with out_path.open("w") as f:
        json.dump(submission, f, indent=2)
    print(f"[template] wrote {len(predictions)} predictions to {out_path}")
    print("[template] next: esp3201-eval grade --task week03_perception_shift --submission", out_path)


# ============================================================
# Per-choice implementation notes
# ============================================================
#
# Choice A: dinov2_linear_head
#   Dependencies: pip install torch transformers
#   load_model(): load the DINOv2 backbone from facebook/dinov2-base, freeze
#     it, train a small linear head on the synthesized training set
#     (see data/train/, data/train_labels.json). The head should output
#     logits for the 4 classes; softmax for confidence.
#   predict_one(): feed the image through the backbone, run the linear head,
#     softmax over logits, return argmax label and max probability.
#   First run includes a one-time weight download (~350 MB).
#
# Choice B: siglip_zero_shot
#   Dependencies: pip install torch transformers
#   load_model(): load a SigLIP model (e.g., google/siglip-base-patch16-224).
#     Precompute text embeddings for the 4 class names. Return (model, text_embs).
#   predict_one(): compute image embedding, dot-product with each text
#     embedding, softmax over the 4 similarities, return argmax label and
#     max probability.
#   First run includes a one-time weight download (~750 MB).
#
# Choice C: grounding_dino_class_presence
#   Dependencies: pip install torch transformers
#   load_model(): load Grounding DINO (e.g., IDEA-Research/grounding-dino-tiny).
#     Build a prompt string of the form "cat. dog. car. person.". Return
#     (model, processor, prompt).
#   predict_one(): run the model on (image, prompt), collect detections,
#     pick the class with the highest-scoring detection, return that label
#     and the detection score (clipped to [0, 1]). If no detections fire,
#     fall back to a uniform-confidence guess and document this in your
#     analysis as a known failure mode.
#   First run includes a one-time weight download (~700 MB).
#
# For all three: keep the test loop CPU-runnable. Use small model variants
# where available. On a typical laptop, a full run over 40 images takes
# 1-5 minutes after weights are cached.
