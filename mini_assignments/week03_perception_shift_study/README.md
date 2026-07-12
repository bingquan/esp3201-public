# Week 3 Guided Lab

## Title

Perception Under Distribution Shift Using Pretrained Models

## Format

The Week 3 lab feeds directly into the Week 3 mini-assignment (Format B, auto-graded harness plus structured analysis). Students do not produce a written deliverable from the lab itself. The deliverable is the harness submission JSON plus the written analysis from the mini-assignment.

## Goal

Students take one pretrained perception model from the curated list, run it on a 40-image test set covering four conditions, and produce a submission JSON that the auto-grading harness scores. The lab is a structured workflow walkthrough so students can focus their effort on the analysis rather than on stitching pipelines together.

## Learning Outcomes

By the end of the lab, students should be able to:

- run a pretrained or classical perception model end-to-end on CPU
- produce a JSON submission matching the harness schema
- interpret per-condition accuracy and calibration findings
- identify which failure modes their submission's findings expose

## Lab Structure

### Part A. Setup (≈10 minutes)

1. Read `perception_packet.md` for the task description and condition definitions.
2. Generate the synthetic dataset: `python data/synthesize_dataset.py`.
3. Confirm that `data/test/` contains 40 PNG images named `img_n_001.png` through `img_o_010.png`.

### Part B. Model Choice (≈5 minutes)

4. Open `mini_assignments/week03_perception_shift_analysis/model_list.md`.
5. Pick one of the four curated models. Note the `model_name` string exactly.

### Part C. Run Your Chosen Model (≈30 minutes to a few hours)

6. For the HOG baseline (`hog_linear_baseline`), run `python starter/run_hog_baseline.py`. The script trains a linear classifier on the synthesized training split and writes a submission JSON.
7. For DINOv2, SigLIP, or Grounding DINO, copy `starter/run_pretrained_template.py` and follow the in-file comments for the model you picked. Expect a longer first run because of weight downloads.

### Part D. Harness Round-Trip (≈5 minutes)

8. Run the harness on your submission:

   ```
   esp3201-eval grade \
       --task week03_perception_shift \
       --submission <your_submission>.json
   ```

9. Inspect the report. If any `fail`-severity findings appear, fix them before submitting.

10. **Plot it.** The report's `per_condition_accuracy` and `per_condition_calibration_gap` are four
    numbers each — easy to skim past, easy to miss which condition is actually dragging the score
    down. Turn the report into two bar charts instead of reading a JSON blob:

    ```
    esp3201-eval grade --task week03_perception_shift --submission <your_submission>.json --json \
        | python starter/plot_report.py
    ```

    This also prints the worst condition and the largest calibration gap directly — the same
    "the headline accuracy hides a bad cell" finding as week10's fairness pillar, just for your
    own model's four distribution-shift conditions instead of a canned CSV.

### Part E. Move to the Mini-Assignment

11. Use the harness report — and the bar charts from step 10 — as the evidence base for your
    mini-assignment analysis. The mini-assignment template guides the rest of the deliverable.

## Required Deliverables From The Lab Itself

None. The mini-assignment is the deliverable. The lab exists so that students have a runnable workflow they can iterate on.

## Suggested Metrics To Watch In The Harness Report

- `overall_accuracy`
- `per_condition_accuracy` (nominal vs low_light vs blur vs ood)
- `per_condition_calibration_gap` (mean confidence minus accuracy on each condition)
- any `warn` or `fail` findings

## Instructor Guidance

- The synthetic dataset is for development. For real grading, replace the harness ground-truth file with hidden data using the same schema (see `evaluation_harness/README.md`).
- The HOG baseline is the only fully-self-contained Choice. The three deep-learning Choices require `transformers`, model-weight downloads, and a sane Python environment. If your cohort has uneven environment setups, point students at the HOG baseline first and let them earn a strong grade by writing strong analysis on a weaker numeric score.
- Encourage students to run the harness early and often. The harness produces all the evidence the analysis needs; treating it as the final check after the analysis is written misses the point.
