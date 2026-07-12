# Example Submissions

These are reference submissions that successfully run through the harness end-to-end. They are not answer keys; they exist so students and instructors can verify that their environment is set up correctly.

## Files

- `week03_example_submission.json`: full coverage of all 40 test items with plausible per-condition behavior. Calibration is intentionally imperfect so the report shows non-trivial findings.
- `week08_example_submission.json`: three deployment configs (`fp32`, `fp16`, `int8_dynamic`). The `int8_dynamic` entry includes one fabricated-claim case to show how the harness flags claim mismatches.

## Quick check

After installing the package (see `evaluation_harness/README.md`):

```
esp3201-eval grade --task week03_perception_shift --submission evaluation_harness/examples/week03_example_submission.json
esp3201-eval grade --task week08_latency_accuracy --submission evaluation_harness/examples/week08_example_submission.json
```

Both commands should print a metrics block and a findings block. If either fails to run, your environment setup is incomplete.
