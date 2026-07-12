"""Tests for the Week 3 perception-shift task."""

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
    task = tasks.get("week03_perception_shift")
    submission = _load("week03_example_submission.json")
    report = task.grade(submission)

    assert report.task_id == "week03_perception_shift"
    assert "overall_accuracy" in report.metrics
    assert set(report.metrics["per_condition_accuracy"]) == {"nominal", "low_light", "blur", "ood"}
    nominal = report.metrics["per_condition_accuracy"]["nominal"]
    ood = report.metrics["per_condition_accuracy"]["ood"]
    assert nominal > ood, "example submission should be stronger on nominal than on ood"


def test_missing_predictions_flagged() -> None:
    task = tasks.get("week03_perception_shift")
    submission = _load("week03_example_submission.json")
    submission["predictions"] = submission["predictions"][:5]
    report = task.grade(submission)
    codes = {f.code for f in report.findings}
    assert "missing_predictions" in codes


def test_constant_prediction_flagged() -> None:
    task = tasks.get("week03_perception_shift")
    submission = _load("week03_example_submission.json")
    for p in submission["predictions"]:
        p["predicted_label"] = "cat"
        p["confidence"] = 0.95
    report = task.grade(submission)
    codes = {f.code for f in report.findings}
    assert "constant_prediction" in codes
    assert "flat_confidence" in codes


def test_unknown_label_flagged() -> None:
    task = tasks.get("week03_perception_shift")
    submission = _load("week03_example_submission.json")
    submission["predictions"][0]["predicted_label"] = "unicorn"
    report = task.grade(submission)
    codes = {f.code for f in report.findings}
    assert "unknown_label" in codes


def test_confidence_out_of_range_flagged() -> None:
    task = tasks.get("week03_perception_shift")
    submission = _load("week03_example_submission.json")
    submission["predictions"][0]["confidence"] = 1.5
    report = task.grade(submission)
    codes = {f.code for f in report.findings}
    assert "confidence_out_of_range" in codes
