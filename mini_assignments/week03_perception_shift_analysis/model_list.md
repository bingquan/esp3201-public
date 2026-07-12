# Week 3 Curated Pretrained Model List

Choose exactly one model from this list for your Week 3 submission. The list is curated so that all entries are runnable on a laptop CPU, have public weights, and expose a classification interface (either natively or via a documented zero-shot prompting recipe).

You must record your chosen model's exact `model_name` string in your submission JSON. Strings are case-sensitive.

## Choice A: DINOv2 backbone + linear classifier head

`model_name`: `dinov2_linear_head`

What it is:

- the DINOv2 self-supervised visual transformer backbone (frozen) with a linear classifier head trained on the released class names

Reference:

- https://arxiv.org/abs/2304.07193 (DINOv2)
- https://github.com/facebookresearch/dinov2

What it does well:

- strong general visual features
- reasonable transfer to unseen domains relative to a from-scratch CNN

What to watch for:

- the linear head is trained on whatever class names you fit it to; calibration depends on how that head was trained
- frozen backbone is a feature about which "modern perception is integration" was made in lecture

## Choice B: SigLIP zero-shot classifier

`model_name`: `siglip_zero_shot`

What it is:

- a SigLIP (or SigLIP 2) vision-language model used as a zero-shot classifier by computing image-text similarity against the released class names

Reference:

- https://arxiv.org/abs/2303.15343 (SigLIP)
- model cards on the Hugging Face model hub

What it does well:

- requires no training; class names are sufficient
- handles novel categories at the cost of variable accuracy

What to watch for:

- prompt sensitivity: paraphrasing class names ("cat" vs "kitten" vs "domestic cat") changes results
- calibration via softmax over a small label set tends to be optimistic on out-of-domain inputs

## Choice C: Grounding DINO + class-presence head

`model_name`: `grounding_dino_class_presence`

What it is:

- Grounding DINO run with the class names as prompts; predicted_label is the class with the highest-scoring detection in the image; confidence is the detection score for that class

Reference:

- https://arxiv.org/abs/2303.05499 (Grounding DINO)

What it does well:

- pulls open-vocabulary detection into a classification interface
- robust to multi-object scenes

What to watch for:

- hallucinated detections: the model will sometimes localize a class that is not present
- thresholding choices affect both accuracy and calibration

## Choice D: classical baseline (HOG + linear classifier)

`model_name`: `hog_linear_baseline`

What it is:

- HOG features plus a linear classifier trained on the released class names

Reference:

- Dalal and Triggs, "Histograms of Oriented Gradients for Human Detection" (2005)

What it does well:

- runs fast on CPU with no dependencies beyond OpenCV / scikit-learn
- provides a non-deep baseline the others are compared against

What to watch for:

- nominal accuracy will likely be lower than the deep choices
- the contrast between nominal and shifted accuracy will look different from the deep choices, which is the point

You are encouraged to choose Choice D at least once across the course. The Week 3 lecture explicitly argues that classical baselines are still valuable as comparison points; choosing the classical baseline and writing a strong analysis is not a weak choice.

## Adding a New Model (Instructor Action)

To extend this list, add a new entry with:

- a stable `model_name` string
- a short description of the model and its interface
- one canonical reference
- "what it does well" and "what to watch for"

Update `evaluation_harness` only if the submission shape changes. The four choices above all submit the same JSON schema.
