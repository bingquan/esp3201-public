# Week 3 Perception Packet

This packet is the student-facing description of the Week 3 perception task. Pair it with `README.md` for the workflow and with `mini_assignments/week03_perception_shift_analysis/README.md` for the deliverable.

## Scenario

A small mobile robot consumes a single forward-facing RGB camera. Its perception pipeline must classify each captured image into one of four categories: `cat`, `dog`, `car`, or `person`. The robot operates indoors. Deployment conditions can differ from the perception model's training distribution in lighting, sensor blur, and occasional out-of-distribution backgrounds.

The engineering question is not "what is the highest nominal accuracy?" but "which perception model is defensible for this deployment, given how it behaves under realistic shift?"

## Task Definition

- **Input:** 64x64 RGB PNG images, one per `image_id`.
- **Output:** for each `image_id`, a predicted label from `{cat, dog, car, person}` and a confidence in `[0, 1]`.
- **Submission shape:** see `evaluation_harness/README.md` (`week03_perception_shift` task schema).

## Dataset Conditions

The test set contains 40 images split into four conditions of 10 images each.

### Condition 1: `nominal`

Image IDs `img_n_001` through `img_n_010`. Clean synthesis: full brightness, no blur, the same class-distinctive shapes used at training time. This is the condition the model is expected to do best on.

### Condition 2: `low_light`

Image IDs `img_l_001` through `img_l_010`. Pixel intensities scaled down to simulate dim deployment lighting. Class-distinctive shapes are still present but reduced in contrast.

### Condition 3: `blur`

Image IDs `img_b_001` through `img_b_010`. Gaussian blur applied to simulate motion blur or out-of-focus capture.

### Condition 4: `ood`

Image IDs `img_o_001` through `img_o_010`. The class-distinctive shapes are placed against backgrounds and at angles the training distribution never showed. This is the most challenging condition and the closest analogue in the synthetic data to true distribution shift.

## Engineering Questions

When you read your harness report after the lab, you should be able to answer:

1. Which condition is your model strongest on? Why is that the expected outcome given the model you chose?
2. Which condition is your model weakest on? Is the drop graceful or catastrophic?
3. Does your model's confidence track its accuracy on each condition, or does it stay high even on the conditions it gets wrong?
4. Which failure mode in the Week 3 lecture taxonomy is most consistent with your numbers?

Your mini-assignment analysis is graded on how well you answer those questions against your real harness output.

## What This Packet Is Not

- Not a description of a real-world dataset. The data is synthetic by design.
- Not a leaderboard. The mini-assignment grade is roughly 20 percent harness numeric and 75 percent interpretive.
- Not a training assignment. You will not train a perception model from scratch. You will run a pretrained or classical model and reason about its behavior.

## Reference

For the model choices, see `mini_assignments/week03_perception_shift_analysis/model_list.md`. For the submission schema, see `evaluation_harness/README.md`. For the deliverable structure, see `mini_assignments/week03_perception_shift_analysis/submission_template.md`.
