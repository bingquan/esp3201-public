"""Week 3 task: perception under distribution shift.

Submission schema:
    {
        "model_name": "<string>",
        "predictions": [
            {"image_id": "<id>", "predicted_label": "<label>", "confidence": <0..1>},
            ...
        ]
    }

Ground-truth schema (`data/week03_groundtruth.json`):
    {
        "labels": ["<label>", ...],
        "items": [
            {"image_id": "<id>", "condition": "<cond>", "true_label": "<label>"},
            ...
        ]
    }

Metrics reported:
    - overall_accuracy
    - per_condition_accuracy: {condition: accuracy}
    - per_condition_mean_confidence: {condition: mean_conf}
    - per_condition_calibration_gap: {condition: mean_conf - accuracy}
    - confidence_variance
    - prediction_diversity: number of distinct labels in submission

Findings flag suspicious submissions:
    - missing_predictions
    - extra_predictions
    - unknown_label
    - confidence_out_of_range
    - constant_prediction
    - flat_confidence
    - large_calibration_gap_<cond>
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from ..report import Report
from .base import Task


_LARGE_CALIBRATION_GAP = 0.20
_FLAT_CONFIDENCE_STD = 0.02


class WeekThreePerceptionShiftTask(Task):
    task_id = "week03_perception_shift"
    description = "Classification under distribution shift across nominal, low-light, blur, and OOD conditions."

    def grade(self, submission: dict[str, Any]) -> Report:
        gt = self.load_groundtruth("week03_groundtruth.json")
        report = Report(task_id=self.task_id)
        report.meta["model_name"] = submission.get("model_name", "<unspecified>")

        gt_items = {item["image_id"]: item for item in gt["items"]}
        valid_labels = set(gt["labels"])
        preds_raw = submission.get("predictions", [])
        preds_by_id: dict[str, dict[str, Any]] = {}
        for p in preds_raw:
            if "image_id" not in p:
                report.add("schema_violation", "fail", "prediction is missing image_id")
                continue
            preds_by_id[p["image_id"]] = p

        missing = sorted(set(gt_items) - set(preds_by_id))
        extra = sorted(set(preds_by_id) - set(gt_items))
        if missing:
            report.add(
                "missing_predictions",
                "fail",
                "submission does not cover all test items",
                count=len(missing),
                examples=missing[:5],
            )
        if extra:
            report.add(
                "extra_predictions",
                "warn",
                "submission includes ids not in the test set",
                count=len(extra),
                examples=extra[:5],
            )

        condition_correct: dict[str, list[int]] = {}
        condition_conf: dict[str, list[float]] = {}
        all_confs: list[float] = []
        all_preds: list[str] = []
        for image_id, item in gt_items.items():
            pred = preds_by_id.get(image_id)
            cond = item["condition"]
            condition_correct.setdefault(cond, [])
            condition_conf.setdefault(cond, [])
            if pred is None:
                continue
            label = pred.get("predicted_label")
            confidence = pred.get("confidence")
            if label not in valid_labels:
                report.add(
                    "unknown_label",
                    "fail",
                    "predicted_label not in the allowed label set",
                    image_id=image_id,
                    predicted_label=label,
                )
                continue
            if not isinstance(confidence, (int, float)) or not 0.0 <= confidence <= 1.0:
                report.add(
                    "confidence_out_of_range",
                    "fail",
                    "confidence must be a float in [0, 1]",
                    image_id=image_id,
                    confidence=confidence,
                )
                continue
            condition_correct[cond].append(1 if label == item["true_label"] else 0)
            condition_conf[cond].append(float(confidence))
            all_confs.append(float(confidence))
            all_preds.append(label)

        per_condition_accuracy: dict[str, float] = {}
        per_condition_mean_conf: dict[str, float] = {}
        per_condition_gap: dict[str, float] = {}
        for cond in sorted(condition_correct):
            corrects = condition_correct[cond]
            confs = condition_conf[cond]
            if not corrects:
                per_condition_accuracy[cond] = float("nan")
                per_condition_mean_conf[cond] = float("nan")
                per_condition_gap[cond] = float("nan")
                continue
            acc = float(np.mean(corrects))
            mean_conf = float(np.mean(confs))
            gap = mean_conf - acc
            per_condition_accuracy[cond] = round(acc, 4)
            per_condition_mean_conf[cond] = round(mean_conf, 4)
            per_condition_gap[cond] = round(gap, 4)
            if abs(gap) > _LARGE_CALIBRATION_GAP:
                report.add(
                    f"large_calibration_gap_{cond}",
                    "warn",
                    "mean confidence diverges from accuracy on this condition",
                    condition=cond,
                    accuracy=per_condition_accuracy[cond],
                    mean_confidence=per_condition_mean_conf[cond],
                    gap=per_condition_gap[cond],
                )

        overall_accuracy = (
            round(float(np.mean([c for cs in condition_correct.values() for c in cs])), 4)
            if any(condition_correct.values())
            else 0.0
        )
        confidence_variance = round(float(np.var(all_confs)) if all_confs else 0.0, 6)
        prediction_diversity = len(set(all_preds))

        if all_preds:
            most_common_label, most_common_count = Counter(all_preds).most_common(1)[0]
            if most_common_count >= 0.9 * len(all_preds):
                report.add(
                    "constant_prediction",
                    "warn",
                    "submission predicts the same label for almost every input",
                    label=most_common_label,
                    fraction=round(most_common_count / len(all_preds), 4),
                )

        if all_confs and float(np.std(all_confs)) < _FLAT_CONFIDENCE_STD:
            report.add(
                "flat_confidence",
                "warn",
                "confidence values barely vary across inputs",
                std=round(float(np.std(all_confs)), 6),
            )

        report.metrics["overall_accuracy"] = overall_accuracy
        report.metrics["per_condition_accuracy"] = per_condition_accuracy
        report.metrics["per_condition_mean_confidence"] = per_condition_mean_conf
        report.metrics["per_condition_calibration_gap"] = per_condition_gap
        report.metrics["confidence_variance"] = confidence_variance
        report.metrics["prediction_diversity"] = prediction_diversity
        return report
