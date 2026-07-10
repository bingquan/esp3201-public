"""Week 5b hands-on lab: evaluating a generated video / world-model rollout.

Generation needs a GPU (done on Colab). The EVALUATION here is fully offline
and reproducible: given a list of frames, it computes objective temporal
metrics that proxy the VBench-style dimensions students reason about:

  - temporal_consistency : 1 - mean per-pixel change between adjacent frames
                           (high = stable; too high can mean "nothing moves")
  - motion_magnitude     : mean per-pixel change (how much the scene moves)
  - flicker              : std over time of global brightness (sudden global
                           jumps -> incoherent generation)
  - brightness_drift     : how far global brightness wanders from frame 0

Prompt adherence and object-identity stability stay HUMAN-judged in the
worksheet -- the point is to combine objective metrics with grounded human
evaluation, not to pretend a single number captures video quality.

Attribution: the evaluation dimensions follow VBench (Huang et al., 2023) and
the diffusers AnimateDiff / CogVideoX tutorials. All code here is original to
ESP3201.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


Frame = np.ndarray  # HxWx3 uint8 or float


def _to_float(frames: List[Frame]) -> np.ndarray:
    arr = np.stack([np.asarray(f, dtype=np.float32) for f in frames])
    if arr.max() > 1.5:
        arr = arr / 255.0
    return arr  # (T, H, W, C) in [0, 1]


def mean_abs_diff(frames: List[Frame]) -> float:
    """Mean absolute per-pixel change between consecutive frames (motion)."""
    arr = _to_float(frames)
    if arr.shape[0] < 2:
        return 0.0
    diffs = np.abs(arr[1:] - arr[:-1])
    return float(diffs.mean())


def temporal_consistency(frames: List[Frame]) -> float:
    """1 - motion. High = visually stable across frames."""
    return float(1.0 - mean_abs_diff(frames))


def flicker(frames: List[Frame]) -> float:
    """Std over time of global mean brightness (global temporal instability)."""
    arr = _to_float(frames)
    per_frame = arr.reshape(arr.shape[0], -1).mean(axis=1)
    return float(per_frame.std())


def brightness_drift(frames: List[Frame]) -> float:
    """Max deviation of global brightness from the first frame."""
    arr = _to_float(frames)
    per_frame = arr.reshape(arr.shape[0], -1).mean(axis=1)
    return float(np.abs(per_frame - per_frame[0]).max())


def score_clip(frames: List[Frame], label: str = "") -> Dict[str, object]:
    """Objective metrics plus slots for the human-judged criteria."""
    return {
        "label": label,
        "n_frames": len(frames),
        "temporal_consistency": round(temporal_consistency(frames), 4),
        "motion_magnitude": round(mean_abs_diff(frames), 4),
        "flicker": round(flicker(frames), 4),
        "brightness_drift": round(brightness_drift(frames), 4),
        # human-judged in the worksheet (1-5):
        "prompt_adherence": None,
        "object_identity_stability": None,
        "physical_plausibility": None,
    }


# --------------------------------------------------------------------------- #
# Loaders / fallbacks
# --------------------------------------------------------------------------- #
def load_frames_from_dir(path: str) -> List[Frame]:
    """Load PNG/JPG frames (sorted by name) from a directory.

    Use this for the precomputed rollout-bank fallback when no GPU is available:
    point it at labs/week05_embodied_system_critique/rollout_bank/<clip>/.
    """
    import glob
    import os
    from PIL import Image
    files = sorted(glob.glob(os.path.join(path, "*.png")) +
                   glob.glob(os.path.join(path, "*.jpg")))
    return [np.asarray(Image.open(f).convert("RGB")) for f in files]


def synth_clip(kind: str, n: int = 16, size: int = 64, seed: int = 0) -> List[Frame]:
    """Synthetic clips to validate the metrics offline (no model needed):
      'static'  -> a still image (consistency ~1, motion ~0)
      'moving'  -> a square translating smoothly (some motion, low flicker)
      'noise'   -> random frames (high motion, high flicker)
    """
    rng = np.random.default_rng(seed)
    frames = []
    if kind == "static":
        base = rng.random((size, size, 3)).astype(np.float32)
        frames = [base.copy() for _ in range(n)]
    elif kind == "moving":
        for t in range(n):
            f = np.full((size, size, 3), 0.1, dtype=np.float32)
            x = int((size - 12) * t / max(1, n - 1))
            f[size // 2 - 6:size // 2 + 6, x:x + 12, :] = [0.9, 0.2, 0.2]
            frames.append(f)
    elif kind == "noise":
        frames = [rng.random((size, size, 3)).astype(np.float32) for _ in range(n)]
    else:
        raise ValueError(f"unknown kind '{kind}'")
    return frames


def frames_from_diffusers(output) -> List[Frame]:
    """Adapt a diffusers text-to-video output to a list of RGB frames.

    Handles both formats seen across pipelines: `output.frames[0]` as a list of
    PIL Images, or (e.g. zeroscope/ModelScope TextToVideoSDPipeline) as numpy
    arrays in [0, 1]. Verified against cerspense/zeroscope_v2_576w.
    """
    frames = output.frames[0]
    out = []
    for f in frames:
        if hasattr(f, "convert"):           # PIL image
            out.append(np.asarray(f.convert("RGB")))
        else:                                # numpy array (H, W, 3)
            out.append(np.asarray(f))
    return out


if __name__ == "__main__":
    for kind in ("static", "moving", "noise"):
        print(kind, "->", score_clip(synth_clip(kind), label=kind))
