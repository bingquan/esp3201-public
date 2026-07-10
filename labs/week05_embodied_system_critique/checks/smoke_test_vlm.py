"""Week 5a VLM lab smoke test (offline). CPU-only, no model download.

Run from the lab root:  python checks/smoke_test_vlm.py
Verifies the scene synthesis + probe + scoring pipeline with the mock backend.
The real VLM backends (HFVLM / GeminiVLM) are exercised only on Colab.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "starter"))

from vlm_lab import (  # noqa: E402
    MockVLM, VOCAB, synthesize_scene, build_probe, build_binding_probe,
    run_probe, score, run_prompt_sensitivity, score_prompt_sensitivity,
    run_probe_on_images,
)


def test_scene_has_exact_ground_truth():
    img, present = synthesize_scene(seed=3, n_objects=3)
    assert img.size == (256, 256)
    assert len(present) == 3 and present.issubset(set(VOCAB))


def test_probe_mixes_present_and_absent():
    _, present = synthesize_scene(seed=5, n_objects=3)
    probe = build_probe(present, n_absent=3, seed=5)
    gts = [gt for _, gt in probe]
    assert sum(gts) == 3 and gts.count(False) == 3


def test_no_hallucination_when_bias_zero():
    recs = run_probe(MockVLM(yes_bias=0.0), n_scenes=8)
    s = score(recs)
    assert s["hallucination_rate"] == 0.0
    assert s["accuracy"] == 1.0


def test_metric_tracks_hallucination():
    low = score(run_probe(MockVLM(yes_bias=0.1), n_scenes=8))
    high = score(run_probe(MockVLM(yes_bias=0.9), n_scenes=8))
    assert high["hallucination_rate"] > low["hallucination_rate"]
    assert high["yes_rate"] > low["yes_rate"]
    assert high["accuracy"] < low["accuracy"]


def test_binding_probe_uses_seen_colors_and_shapes():
    _, present = synthesize_scene(seed=7, n_objects=3)
    traps = [o for o, gt in build_binding_probe(present, seed=7) if not gt]
    colors = {o.split(" ", 1)[0] for o in present}
    shapes = {o.split(" ", 1)[1] for o in present}
    for t in traps:
        c, s = t.split(" ", 1)
        assert c in colors and s in shapes      # both seen, but not together
        assert t not in present


def test_prompt_sensitivity_metric():
    # The mock hashes (obj|question), so absent objects flip across phrasings.
    recs = run_prompt_sensitivity(MockVLM(yes_bias=0.4), n_scenes=3)
    s = score_prompt_sensitivity(recs)
    assert s["n_templates"] >= 3
    assert s["flip_rate_absent"] > 0.0          # wording changes the answer
    assert s["flip_rate_present"] == 0.0        # mock always confirms present


def test_run_probe_on_images():
    items = [(synthesize_scene(s, n_objects=3)[0], synthesize_scene(s, n_objects=3)[1])
             for s in range(3)]
    recs = run_probe_on_images(MockVLM(yes_bias=0.0), items)
    assert score(recs)["accuracy"] == 1.0


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} smoke tests passed.")
