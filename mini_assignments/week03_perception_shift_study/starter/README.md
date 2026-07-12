# Week 3 Starter Scripts

## Quick start

1. From the repo root, install the harness and lab dependencies if you have not already:

   ```
   pip install -e evaluation_harness
   pip install numpy scikit-learn scikit-image pillow
   ```

2. Generate the synthetic dataset:

   ```
   python mini_assignments/week03_perception_shift_study/data/synthesize_dataset.py
   ```

   This creates `data/train/` (160 training images) and `data/test/` (40 test images matching the harness ground-truth IDs).

3. Choose one of the curated models from `mini_assignments/week03_perception_shift_analysis/model_list.md`.

4. Run the corresponding starter:

   - **`hog_linear_baseline`** (Choice D, fully runnable out of the box):

     ```
     python mini_assignments/week03_perception_shift_study/starter/run_hog_baseline.py
     ```

   - **`dinov2_linear_head` / `siglip_zero_shot` / `grounding_dino_class_presence`** (Choices A/B/C):

     Copy `run_pretrained_template.py` to a new filename. Follow the comments at the bottom of the file for the model you chose. You will need to `pip install torch transformers` first, and the first run will download model weights (350 MB to 750 MB depending on choice).

5. Grade your submission:

   ```
   esp3201-eval grade --task week03_perception_shift --submission <your_file>.json
   ```

6. Plot the report instead of reading the JSON by eye:

   ```
   esp3201-eval grade --task week03_perception_shift --submission <your_file>.json --json \
       | python starter/plot_report.py
   ```

## Files

- `run_hog_baseline.py`: fully runnable HOG + LinearSVC baseline (Choice D). Trains on the synthesized training set, predicts on the test set, writes submission JSON.
- `run_pretrained_template.py`: template for the three pretrained choices. Fill in `load_model()` and `predict_one()` for your chosen model. The bottom of the file documents what to put in each function for each curated choice.
- `plot_report.py`: turns a `--json` harness report into two bar charts (per-condition accuracy vs. the headline number, and per-condition calibration gap) and prints the worst condition and largest calibration gap. Reads a file path argument or stdin, so it composes directly with `esp3201-eval grade ... --json | python starter/plot_report.py`.

## Notes for students

- The HOG baseline will not score as high as a pretrained model nominally, but it is reproducible end-to-end with no weight downloads and no GPU. Choosing it is a defensible engineering choice that the rubric rewards if the analysis is strong.
- The pretrained choices have a one-time cost (weight downloads, dependency installs) but run on CPU after that. Plan for the first run to take longer than subsequent ones.
- If your environment cannot install `torch` or `transformers`, fall back to the HOG baseline rather than skipping the assignment.

## Notes for instructors

- All scripts read from `data/test/` using the image IDs in `data/test_image_ids.json`. The harness ground truth must use the same IDs for the auto-grading to make sense.
- For real grading, replace `evaluation_harness/esp3201_eval/data/week03_groundtruth.json` with hidden data using the same schema. The synthesized images will still be visible to students; that is intentional for the development workflow.
- The HOG baseline ships with no class-balancing or hyperparameter tuning. This is intentional. Students are not graded on baseline performance; they are graded on analysis.
