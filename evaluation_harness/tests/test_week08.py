"""Tests for the Week 8 latency-accuracy task."""

from __future__ import annotations

import json
from pathlib import Path

from esp3201_eval import tasks

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples"


def _load(name: str) -> dict:
    with (EXAMPLES / name).open() as f:
        return json.load(f)


def test_example_submission_runs_end_to_end() -> None:
    task = tasks.get("week08_latency_accuracy")
    submission = _load("week08_example_submission.json")
    report = task.grade(submission)

    assert report.task_id == "week08_latency_accuracy"
    assert report.metrics["num_configs"] == 3
    assert all("verified_accuracy" in c for c in report.metrics["per_config"])
    assert isinstance(report.metrics["pareto_frontier"], list)
    assert len(report.metrics["pareto_frontier"]) >= 1


def test_claim_mismatch_flagged() -> None:
    """The int8_dynamic config in the example claims 0.70 accuracy but
    verified accuracy on the example predictions is different. The harness
    should surface this mismatch."""
    task = tasks.get("week08_latency_accuracy")
    submission = _load("week08_example_submission.json")
    for cfg in submission["configs"]:
        if cfg["config_name"] == "int8_dynamic":
            cfg["claimed_accuracy"] = 0.99
    report = task.grade(submission)
    codes = {f.code for f in report.findings}
    assert "claim_mismatch" in codes


def test_missing_predictions_flagged() -> None:
    task = tasks.get("week08_latency_accuracy")
    submission = _load("week08_example_submission.json")
    submission["configs"][0]["predictions"] = submission["configs"][0]["predictions"][:3]
    report = task.grade(submission)
    codes = {f.code for f in report.findings}
    assert "missing_config_predictions" in codes


def test_duplicate_config_name_flagged() -> None:
    task = tasks.get("week08_latency_accuracy")
    submission = _load("week08_example_submission.json")
    duplicate = json.loads(json.dumps(submission["configs"][0]))
    submission["configs"].append(duplicate)
    report = task.grade(submission)
    codes = {f.code for f in report.findings}
    assert "duplicate_config_name" in codes


def test_pareto_frontier_excludes_dominated() -> None:
    task = tasks.get("week08_latency_accuracy")
    submission = {
        "model_family": "test",
        "configs": [
            {
                "config_name": "slow_and_inaccurate",
                "claimed_latency_ms": 100.0,
                "claimed_accuracy": 0.5,
                "predictions": [
                    {"input_id": f"in_{i:03d}", "predicted_label": "positive"} for i in range(1, 21)
                ],
            },
            {
                "config_name": "fast_and_accurate",
                "claimed_latency_ms": 10.0,
                "claimed_accuracy": 0.5,
                "predictions": [
                    {"input_id": f"in_{i:03d}", "predicted_label": "positive"} for i in range(1, 21)
                ],
            },
        ],
    }
    report = task.grade(submission)
    pareto_names = {c["config_name"] for c in report.metrics["pareto_frontier"]}
    assert "fast_and_accurate" in pareto_names
    assert "slow_and_inaccurate" not in pareto_names
