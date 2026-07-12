"""Offline smoke tests for trustworthy_ai_lab.py -- no GPU, no API, no network.

Run from the lab root: python checks/smoke_test.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "starter"))

import numpy as np

import trustworthy_ai_lab as lab

SAMPLE_CSV = """condition,lighting,subgroup,n_samples,detections_correct
static_worker,bright,hi_vis_vest,900,891
static_worker,bright,dark_workwear,700,686
static_worker,dim,hi_vis_vest,600,588
static_worker,dim,dark_workwear,400,360
moving_worker,bright,hi_vis_vest,900,882
moving_worker,bright,dark_workwear,600,582
moving_worker,dim,hi_vis_vest,500,469
moving_worker,dim,dark_workwear,400,332
"""


def test_load_subgroup_csv():
    rows = lab.load_subgroup_csv(SAMPLE_CSV)
    assert len(rows) == 8
    assert rows[0]["n_samples"] == 900
    assert rows[0]["detections_correct"] == 891
    print("PASS test_load_subgroup_csv")


def test_weighted_accuracy_matches_headline():
    rows = lab.load_subgroup_csv(SAMPLE_CSV)
    acc = lab.weighted_accuracy(rows)
    assert abs(acc - 0.958) < 1e-6, acc
    print("PASS test_weighted_accuracy_matches_headline")


def test_worst_cell_is_moving_dim_dark():
    rows = lab.load_subgroup_csv(SAMPLE_CSV)
    worst = lab.worst_cell(rows)
    assert worst["condition"] == "moving_worker"
    assert worst["lighting"] == "dim"
    assert worst["subgroup"] == "dark_workwear"
    assert abs(worst["accuracy"] - 0.83) < 1e-6
    print("PASS test_worst_cell_is_moving_dim_dark")


def test_per_cell_accuracy_bounds():
    rows = lab.load_subgroup_csv(SAMPLE_CSV)
    cells = lab.per_cell_accuracy(rows)
    assert all(0.0 <= c["accuracy"] <= 1.0 for c in cells)
    print("PASS test_per_cell_accuracy_bounds")


def test_explainability_local_differs_from_global():
    X, y = lab.make_explainability_demo_data(seed=0)
    model = lab.train_explainability_demo_model(X, y)
    global_w = lab.global_feature_weight(model)
    assert global_w["feature_a"] > global_w["feature_b"], "feature_a should dominate globally"
    idx = int(np.argmin(np.abs(X[:, 0])))
    local = lab.local_explanation(model, X[idx], X)["local_importance"]
    assert local["feature_b"] > local["feature_a"], (
        "the spurious-correlation instance should show feature_b as locally more important, "
        "despite feature_a dominating globally"
    )
    print("PASS test_explainability_local_differs_from_global")


def test_membership_inference_gap_is_real():
    X_train, y_train, X_test, y_test = lab.make_membership_inference_demo_data(seed=0)
    model = lab.train_membership_inference_demo_model(X_train, y_train)
    result = lab.membership_confidence_gap(model, X_train, X_test)
    assert result["gap"] > 0.03, f"expected a real train/test confidence gap, got {result}"
    assert model.score(X_train, y_train) > 0.95, "training accuracy should show near-perfect memorization"
    print("PASS test_membership_inference_gap_is_real")


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\n{len(tests)}/{len(tests)} smoke tests passed.")
