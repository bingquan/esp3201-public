"""Week 6 video-eval lab smoke test (offline). CPU-only, no model.

Run from the lab root:  python checks/smoke_test_video.py
Validates the temporal metrics on synthetic clips. Real generation runs on Colab.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "starter"))

from video_eval import (  # noqa: E402
    synth_clip, score_clip, temporal_consistency, mean_abs_diff,
)


def test_static_is_perfectly_consistent():
    s = score_clip(synth_clip("static"))
    assert s["temporal_consistency"] == 1.0
    assert s["motion_magnitude"] == 0.0


def test_motion_ordering():
    static = mean_abs_diff(synth_clip("static"))
    moving = mean_abs_diff(synth_clip("moving"))
    noise = mean_abs_diff(synth_clip("noise"))
    assert static < moving < noise


def test_consistency_ordering():
    static = temporal_consistency(synth_clip("static"))
    moving = temporal_consistency(synth_clip("moving"))
    noise = temporal_consistency(synth_clip("noise"))
    assert noise < moving < static


def test_human_slots_present_but_unset():
    s = score_clip(synth_clip("moving"))
    for k in ("prompt_adherence", "object_identity_stability", "physical_plausibility"):
        assert k in s and s[k] is None


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} smoke tests passed.")
