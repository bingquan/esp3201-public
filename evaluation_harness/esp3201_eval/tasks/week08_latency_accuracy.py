"""Week 8 task: latency / accuracy frontier under deployment configs.

Submission schema:
    {
        "model_family": "<string>",
        "configs": [
            {
                "config_name": "<string, e.g. fp32 | fp16 | int8_static>",
                "claimed_latency_ms": <float>,
                "claimed_accuracy": <float in [0,1]>,
                "predictions": [
                    {"input_id": "<id>", "predicted_label": "<label>"},
                    ...
                ]
            },
            ...
        ]
    }

Ground-truth schema (`data/week08_groundtruth.json`):
    {
        "labels": ["<label>", ...],
        "items": [{"input_id": "<id>", "true_label": "<label>"}, ...],
        "latency_bounds_ms": {"min": <float>, "max": <float>}
    }

Metrics reported per submitted config:
    - verified_accuracy
    - claimed_accuracy
    - accuracy_claim_error
    - claimed_latency_ms
    - latency_within_bounds

Aggregate metrics:
    - pareto_frontier: list of (config_name, latency, accuracy) on the frontier

Findings:
    - missing_config_predictions
    - claim_mismatch (verified vs claimed accuracy)
    - latency_out_of_bounds
    - unknown_label
    - duplicate_config_name
"""

from __future__ import annotations

from typing import Any

from ..report import Report
from .base import Task


_CLAIM_TOLERANCE = 0.02


class WeekEightLatencyAccuracyTask(Task):
    task_id = "week08_latency_accuracy"
    description = "Latency-accuracy frontier across quantization or runtime configurations."

    def grade(self, submission: dict[str, Any]) -> Report:
        gt = self.load_groundtruth("week08_groundtruth.json")
        report = Report(task_id=self.task_id)
        report.meta["model_family"] = submission.get("model_family", "<unspecified>")

        gt_by_id = {item["input_id"]: item for item in gt["items"]}
        valid_labels = set(gt["labels"])
        bounds = gt.get("latency_bounds_ms", {"min": 0.0, "max": float("inf")})

        configs = submission.get("configs", [])
        seen_names: set[str] = set()
        per_config_metrics: list[dict[str, Any]] = []

        for cfg in configs:
            name = cfg.get("config_name", "<unnamed>")
            if name in seen_names:
                report.add("duplicate_config_name", "fail", "config_name repeated", config_name=name)
                continue
            seen_names.add(name)

            claimed_accuracy = float(cfg.get("claimed_accuracy", float("nan")))
            claimed_latency = float(cfg.get("claimed_latency_ms", float("nan")))

            preds = {p["input_id"]: p for p in cfg.get("predictions", []) if "input_id" in p}
            missing = sorted(set(gt_by_id) - set(preds))
            if missing:
                report.add(
                    "missing_config_predictions",
                    "fail",
                    "config is missing predictions for some test items",
                    config_name=name,
                    count=len(missing),
                    examples=missing[:5],
                )

            correct = 0
            scored = 0
            for input_id, item in gt_by_id.items():
                pred = preds.get(input_id)
                if pred is None:
                    continue
                label = pred.get("predicted_label")
                if label not in valid_labels:
                    report.add(
                        "unknown_label",
                        "fail",
                        "predicted_label not in the allowed label set",
                        config_name=name,
                        input_id=input_id,
                        predicted_label=label,
                    )
                    continue
                scored += 1
                if label == item["true_label"]:
                    correct += 1

            verified_accuracy = round(correct / scored, 4) if scored else 0.0
            accuracy_claim_error = round(verified_accuracy - claimed_accuracy, 4)
            if scored and abs(accuracy_claim_error) > _CLAIM_TOLERANCE:
                report.add(
                    "claim_mismatch",
                    "fail",
                    "claimed accuracy disagrees with verified accuracy beyond tolerance",
                    config_name=name,
                    claimed_accuracy=claimed_accuracy,
                    verified_accuracy=verified_accuracy,
                    tolerance=_CLAIM_TOLERANCE,
                )

            latency_within_bounds = bounds["min"] <= claimed_latency <= bounds["max"]
            if not latency_within_bounds:
                report.add(
                    "latency_out_of_bounds",
                    "warn",
                    "claimed latency is outside the sanity bounds for this assignment",
                    config_name=name,
                    claimed_latency_ms=claimed_latency,
                    bounds_ms=bounds,
                )

            per_config_metrics.append(
                {
                    "config_name": name,
                    "verified_accuracy": verified_accuracy,
                    "claimed_accuracy": round(claimed_accuracy, 4),
                    "accuracy_claim_error": accuracy_claim_error,
                    "claimed_latency_ms": round(claimed_latency, 4),
                    "latency_within_bounds": latency_within_bounds,
                }
            )

        pareto = _pareto_frontier(per_config_metrics)
        report.metrics["per_config"] = per_config_metrics
        report.metrics["pareto_frontier"] = pareto
        report.metrics["num_configs"] = len(per_config_metrics)
        return report


def _pareto_frontier(per_config: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return configs on the Pareto frontier of (lower latency, higher verified_accuracy)."""
    pareto: list[dict[str, Any]] = []
    for c in per_config:
        dominated = False
        for other in per_config:
            if other is c:
                continue
            if (
                other["claimed_latency_ms"] <= c["claimed_latency_ms"]
                and other["verified_accuracy"] >= c["verified_accuracy"]
                and (
                    other["claimed_latency_ms"] < c["claimed_latency_ms"]
                    or other["verified_accuracy"] > c["verified_accuracy"]
                )
            ):
                dominated = True
                break
        if not dominated:
            pareto.append(
                {
                    "config_name": c["config_name"],
                    "claimed_latency_ms": c["claimed_latency_ms"],
                    "verified_accuracy": c["verified_accuracy"],
                }
            )
    pareto.sort(key=lambda x: x["claimed_latency_ms"])
    return pareto
