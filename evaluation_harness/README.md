# Evaluation Harness

`esp3201_eval` is the auto-grading harness used by mini-assignments that ask students to submit predictions for hidden test cases. It is intentionally small, has minimal dependencies, and is readable end-to-end so students can inspect what they are being graded on.

The harness is currently used by:

- Week 3 perception shift mini-assignment (`week03_perception_shift`)
- Week 8 latency / accuracy frontier mini-assignment (`week08_latency_accuracy`)

## Install

Requires Python 3.10 or later. From the repository root:

```
pip install -e evaluation_harness
```

This installs the `esp3201-eval` command and the `esp3201_eval` Python package.

The only runtime dependency is `numpy`. Tests use `pytest`.

## Usage

List known tasks:

```
esp3201-eval list
```

Grade a submission:

```
esp3201-eval grade \
    --task week03_perception_shift \
    --submission path/to/submission.json
```

Emit the report as JSON (useful for instructor pipelines):

```
esp3201-eval grade \
    --task week08_latency_accuracy \
    --submission path/to/submission.json \
    --json
```

The exit code is `0` if there are no `fail`-severity findings, and `2` otherwise. `warn`-severity findings (such as a large calibration gap) do not fail the run; they are signals for the analysis-grading rubric.

## What the harness does and does not do

It does:

- check the submission against a schema (presence, types, value ranges)
- compute the numeric metrics the rubric uses (accuracy, per-condition accuracy, calibration gap, latency-accuracy frontier)
- emit structured `findings` describing suspicious patterns (constant predictions, flat confidence, fabricated accuracy claims, missing coverage)
- produce both a human-readable text report and a JSON report

It does not:

- decide a final grade
- run student models on real images
- enforce that the submitted model is the one the student claims to have used
- grade analysis quality; the TA does that, informed by the structured findings

## Repository layout

```
evaluation_harness/
├── pyproject.toml
├── README.md
├── esp3201_eval/
│   ├── __init__.py
│   ├── cli.py
│   ├── report.py
│   ├── tasks/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── week03_perception_shift.py
│   │   └── week08_latency_accuracy.py
│   └── data/
│       ├── week03_groundtruth.json
│       └── week08_groundtruth.json
├── examples/
│   ├── README.md
│   ├── week03_example_submission.json
│   └── week08_example_submission.json
└── tests/
    ├── test_week03.py
    └── test_week08.py
```

## Submission schemas

### Week 3 (`week03_perception_shift`)

```json
{
  "model_name": "<your-model-identifier>",
  "predictions": [
    {"image_id": "<id>", "predicted_label": "<label>", "confidence": 0.87},
    ...
  ]
}
```

- `predicted_label` must be one of the labels named in the ground-truth label set.
- `confidence` must be a float in [0, 1].
- The submission must cover every `image_id` in the test set.

### Week 8 (`week08_latency_accuracy`)

```json
{
  "model_family": "<your-model-identifier>",
  "configs": [
    {
      "config_name": "<name, e.g. fp32 | fp16 | int8_static>",
      "claimed_latency_ms": 18.4,
      "claimed_accuracy": 0.85,
      "predictions": [
        {"input_id": "<id>", "predicted_label": "<label>"},
        ...
      ]
    },
    ...
  ]
}
```

- Each config submission must include `predictions` so the harness can verify the claimed accuracy.
- A `claim_mismatch` finding is raised if `verified_accuracy` differs from `claimed_accuracy` by more than the per-task tolerance.

## For instructors

The repo ships with synthetic ground-truth files so the harness runs out of the box. To use real hidden test data:

1. Replace `esp3201_eval/data/week03_groundtruth.json` and `esp3201_eval/data/week08_groundtruth.json` with the real hidden test sets, keeping the same schema.
2. Distribute only the `image_id` / `input_id` list and condition labels to students, not the `true_label` field.
3. Run `esp3201-eval grade` per submission. The `--student-id` flag echoes a student identifier into `report.meta` for batch grading scripts.

The synthetic ground-truth files are intentionally small so the harness round-trip is fast and so test execution does not depend on real datasets.

## For students

You are graded on two things:

1. the metrics and findings the harness produces from your submission
2. your written analysis interpreting those metrics and findings

You can run the harness yourself against the example ground truth to sanity-check your submission format before submitting against the hidden test set. The findings the harness raises on your submission should appear in your written analysis. Submissions that ignore visible findings (a flagged calibration gap, a constant-prediction pattern, a claim mismatch) lose marks on analysis quality, not on numeric performance.

## Testing

```
pytest evaluation_harness/tests
```

This exercises both task implementations against the shipped example submissions and verifies that the major findings codes fire correctly.
