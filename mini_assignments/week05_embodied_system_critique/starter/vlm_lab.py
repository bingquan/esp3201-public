"""Week 5 hands-on lab: probing a VLM for object hallucination (POPE-style).

The grounding probe is fully reproducible: scenes are synthesized with EXACT
ground truth (which colored shapes are present), and the metric of interest is
the hallucination rate -- how often the model answers "yes" to an object that
is NOT in the image.

Three backends behind one `answer(image, question) -> "yes"|"no"` adapter:
  - MockVLM   : deterministic, offline; simulates a tunable yes-bias so the
                scoring pipeline can be verified with no model download.
  - HFVLM     : a small local vision-language model via transformers (PIN THIS).
  - GeminiVLM : a free-tier hosted VLM via google-generativeai (PIN THIS).

Attribution: the present/absent object-probing methodology follows POPE
(Li et al., "Evaluating Object Hallucination in Large Vision-Language Models",
2023). Scene synthesis and all code here are original to ESP3201.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

COLORS = {
    "red": (220, 40, 40),
    "blue": (40, 80, 220),
    "green": (40, 170, 70),
    "yellow": (235, 205, 40),
}
SHAPES = ["square", "circle", "triangle"]
VOCAB: List[str] = [f"{c} {s}" for c in COLORS for s in SHAPES]


# --------------------------------------------------------------------------- #
# Scene synthesis (exact ground truth)
# --------------------------------------------------------------------------- #
def synthesize_scene(seed: int, n_objects: int = 3, size: int = 256):
    """Return (PIL.Image, present_objects:set[str]) with exact ground truth."""
    from PIL import Image, ImageDraw  # local import so import-time stays light
    import random

    rng = random.Random(seed)
    chosen = rng.sample(VOCAB, n_objects)
    img = Image.new("RGB", (size, size), (245, 245, 245))
    draw = ImageDraw.Draw(img)
    slots = [(size * (i + 1) // (n_objects + 1), size // 2) for i in range(n_objects)]
    for (label, (cx, cy)) in zip(chosen, slots):
        color_name, shape = label.split(" ", 1)
        col = COLORS[color_name]
        r = size // 8
        if shape == "square":
            draw.rectangle([cx - r, cy - r, cx + r, cy + r], fill=col)
        elif shape == "circle":
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=col)
        else:  # triangle
            draw.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=col)
    return img, set(chosen)


def build_probe(present: Set[str], n_absent: int = 3, seed: int = 0
                ) -> List[Tuple[str, bool]]:
    """Return [(object, ground_truth_present)] mixing present and absent objects."""
    import random
    rng = random.Random(seed)
    absent_pool = [o for o in VOCAB if o not in present]
    absent = rng.sample(absent_pool, min(n_absent, len(absent_pool)))
    probe = [(o, True) for o in present] + [(o, False) for o in absent]
    rng.shuffle(probe)
    return probe


def build_binding_probe(present: Set[str], seed: int = 0
                        ) -> List[Tuple[str, bool]]:
    """A harder probe targeting binding errors. The absent objects are combos
    whose COLOR and SHAPE both appear in the scene but not together
    (e.g. scene has a 'red square' and a 'blue circle' -> ask about a
    'red circle'). VLMs hallucinate these far more than fully-absent objects,
    so the failure mode shows up even on a capable model."""
    import random
    rng = random.Random(seed)
    colors = {o.split(" ", 1)[0] for o in present}
    shapes = {o.split(" ", 1)[1] for o in present}
    traps = [f"{c} {s}" for c in colors for s in shapes if f"{c} {s}" not in present]
    rng.shuffle(traps)
    probe = [(o, True) for o in present] + [(o, False) for o in traps]
    rng.shuffle(probe)
    return probe


# Different phrasings of the SAME yes/no question. A grounded model answers all
# of them the same way; prompt sensitivity is when the answer flips with wording.
QUESTION_TEMPLATES = [
    "Is there a {obj} in the image? Answer yes or no.",
    "Can you see a {obj}? Answer yes or no.",
    "Does this image contain a {obj}? Answer yes or no.",
    "Is a {obj} present in this picture? Answer yes or no.",
]


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #
class MockVLM:
    """Offline backend. Answers correctly on present objects; on absent objects
    answers 'yes' with probability `yes_bias` (deterministically per question).
    Use this to validate the scoring pipeline before loading a real model."""

    def __init__(self, yes_bias: float = 0.4):
        self.yes_bias = yes_bias

    def answer(self, image, question: str, obj: str, present: bool) -> str:
        if present:
            return "yes"
        h = int(hashlib.sha256(f"{obj}|{question}".encode()).hexdigest(), 16)
        return "yes" if (h % 1000) / 1000.0 < self.yes_bias else "no"


class HFVLM:
    """Small local vision-language model via the standard transformers
    image-text-to-text interface (chat template + generate).

    Verified on `HuggingFaceTB/SmolVLM-Instruct` (see pinned_models.md). The
    same code path works for other instruct VLMs exposed through
    AutoModelForImageTextToText (SmolVLM2, Idefics, Qwen2-VL, ...). Pin and
    re-measure VRAM if you switch checkpoints.
    """

    def __init__(self, model_id: str = "HuggingFaceTB/SmolVLM-Instruct",
                 device: str = "cuda", dtype="float16"):
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, dtype=getattr(torch, dtype)).to(device)
        self.device = device

    def answer(self, image, question: str, obj: str = "", present: bool = False) -> str:
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": question}]}]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=[image], return_tensors="pt").to(self.device)
        n_in = inputs["input_ids"].shape[1]
        out = self.model.generate(**inputs, max_new_tokens=8, do_sample=False)
        text = self.processor.batch_decode(
            out[:, n_in:], skip_special_tokens=True)[0]
        return _parse_yes_no(text)


class GeminiVLM:
    """Free-tier hosted VLM via google-generativeai.

    PIN THIS: confirm the exact model id (e.g. a current Gemini Flash) and that
    your free-tier key works. Reads GOOGLE_API_KEY from the environment / Colab
    secret.
    """

    def __init__(self, model_id: str = "", api_key: Optional[str] = None):  # PIN THIS
        import os
        import google.generativeai as genai
        if not model_id:
            raise ValueError("Set model_id to a pinned Gemini model (see notebook).")
        genai.configure(api_key=api_key or os.environ["GOOGLE_API_KEY"])
        self.model = genai.GenerativeModel(model_id)

    def answer(self, image, question: str, obj: str = "", present: bool = False) -> str:
        resp = self.model.generate_content([question, image])
        return _parse_yes_no(resp.text)


def _parse_yes_no(text: str) -> str:
    t = (text or "").strip().lower()
    if t.startswith("yes") or " yes" in t[:12]:
        return "yes"
    if t.startswith("no") or " no" in t[:12]:
        return "no"
    return "yes" if "yes" in t else "no"


# --------------------------------------------------------------------------- #
# Run + score
# --------------------------------------------------------------------------- #
@dataclass
class ProbeRecord:
    scene_seed: int
    obj: str
    present: bool
    answered_yes: bool


def run_probe(backend, n_scenes: int = 8, n_objects: int = 3,
              n_absent: int = 3, base_seed: int = 0,
              hard: bool = False) -> List[ProbeRecord]:
    """Run the present/absent probe over synthesized scenes.

    hard=False: absent objects are random combos not in the scene.
    hard=True:  absent objects are binding traps (color and shape both present
                but not together) -- elicits hallucination even on good models.
    """
    records: List[ProbeRecord] = []
    for i in range(n_scenes):
        seed = base_seed + i
        image, present = synthesize_scene(seed, n_objects=n_objects)
        probe = (build_binding_probe(present, seed=seed) if hard
                 else build_probe(present, n_absent=n_absent, seed=seed))
        for obj, gt in probe:
            q = f"Is there a {obj} in the image? Answer yes or no."
            ans = backend.answer(image, q, obj=obj, present=gt)
            records.append(ProbeRecord(seed, obj, gt, ans == "yes"))
    return records


def score(records: List[ProbeRecord]) -> Dict[str, float]:
    present = [r for r in records if r.present]
    absent = [r for r in records if not r.present]
    tp = sum(r.answered_yes for r in present)
    fp = sum(r.answered_yes for r in absent)
    n_present = len(present)
    n_absent = len(absent)
    yes_total = sum(r.answered_yes for r in records)
    correct = tp + (n_absent - fp)
    return {
        "n_questions": len(records),
        "accuracy": round(correct / max(1, len(records)), 3),
        "hallucination_rate": round(fp / max(1, n_absent), 3),   # yes on ABSENT
        "recall_present": round(tp / max(1, n_present), 3),      # yes on PRESENT
        "yes_rate": round(yes_total / max(1, len(records)), 3),  # overall yes-bias
        "precision": round(tp / max(1, tp + fp), 3),
    }


# --------------------------------------------------------------------------- #
# Prompt-sensitivity probe (a signature VLM limitation, independent of whether
# the model hallucinates objects)
# --------------------------------------------------------------------------- #
@dataclass
class SensitivityRecord:
    scene_seed: int
    obj: str
    present: bool
    answers: List[bool]          # one per question template
    flipped: bool                # did the answer change with phrasing?


def run_prompt_sensitivity(backend, n_scenes: int = 4, n_objects: int = 3,
                           n_absent: int = 3, base_seed: int = 0
                           ) -> List[SensitivityRecord]:
    """Ask the SAME object question under several phrasings and record whether
    the answer flips. A well-grounded model is phrasing-invariant."""
    records: List[SensitivityRecord] = []
    for i in range(n_scenes):
        seed = base_seed + i
        image, present = synthesize_scene(seed, n_objects=n_objects)
        for obj, gt in build_probe(present, n_absent=n_absent, seed=seed):
            answers = []
            for tmpl in QUESTION_TEMPLATES:
                ans = backend.answer(image, tmpl.format(obj=obj), obj=obj, present=gt)
                answers.append(ans == "yes")
            records.append(SensitivityRecord(seed, obj, gt, answers,
                                             flipped=len(set(answers)) > 1))
    return records


def score_prompt_sensitivity(records: List[SensitivityRecord]) -> Dict[str, float]:
    present = [r for r in records if r.present]
    absent = [r for r in records if not r.present]

    def flip_rate(rs):
        return round(sum(r.flipped for r in rs) / max(1, len(rs)), 3)

    return {
        "n_objects": len(records),
        "n_templates": len(QUESTION_TEMPLATES),
        "flip_rate_overall": flip_rate(records),     # answer changed with wording
        "flip_rate_present": flip_rate(present),
        "flip_rate_absent": flip_rate(absent),
    }


# --------------------------------------------------------------------------- #
# Real-image probe (optional, for a stronger hallucination signal than the
# synthetic shapes give)
# --------------------------------------------------------------------------- #
def load_labeled_images(image_dir: str):
    """Load (PIL image, present_objects:set) pairs from a directory.

    Expects images plus a `labels.json` mapping filename -> list of present
    object strings (using the same `{color} {shape}` vocabulary, or your own).
    Lets an instructor swap in natural photos where VLM hallucination is more
    pronounced than on the synthetic shapes; the metric code is unchanged.
    """
    import json
    import os
    from PIL import Image
    labels = json.load(open(os.path.join(image_dir, "labels.json")))
    items = []
    for fname, present in labels.items():
        items.append((Image.open(os.path.join(image_dir, fname)).convert("RGB"),
                      set(present)))
    return items


def run_probe_on_images(backend, items, vocab=None, n_absent: int = 3,
                        seed: int = 0) -> List[ProbeRecord]:
    """Run the present/absent probe on caller-supplied (image, present) pairs."""
    import random
    vocab = vocab or VOCAB
    rng = random.Random(seed)
    records: List[ProbeRecord] = []
    for idx, (image, present) in enumerate(items):
        absent_pool = [o for o in vocab if o not in present]
        absent = rng.sample(absent_pool, min(n_absent, len(absent_pool)))
        probe = [(o, True) for o in present] + [(o, False) for o in absent]
        for obj, gt in probe:
            q = f"Is there a {obj} in the image? Answer yes or no."
            ans = backend.answer(image, q, obj=obj, present=gt)
            records.append(ProbeRecord(idx, obj, gt, ans == "yes"))
    return records


if __name__ == "__main__":
    for bias in (0.0, 0.4, 0.9):
        recs = run_probe(MockVLM(yes_bias=bias), n_scenes=8)
        print(f"MockVLM yes_bias={bias}: {score(recs)}")
    print("prompt sensitivity (mock yes_bias=0.4):",
          score_prompt_sensitivity(run_prompt_sensitivity(MockVLM(yes_bias=0.4))))
