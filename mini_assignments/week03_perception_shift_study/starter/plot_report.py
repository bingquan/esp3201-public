"""Visualize a week03 harness report (Part D-plus).

The harness prints per-condition accuracy and calibration gap as numbers in a
JSON blob. Reading four accuracy numbers and four gap numbers off a screen is
exactly the "headline hides the worst cell" problem this course keeps coming
back to (see week10's fairness pillar) -- so this script turns the report
into two bar charts instead of asking you to eyeball a printed dict.

Usage:
    esp3201-eval grade --task week03_perception_shift --submission sub.json --json > report.json
    python starter/plot_report.py report.json

    # or pipe directly, no intermediate file:
    esp3201-eval grade --task week03_perception_shift --submission sub.json --json \\
        | python starter/plot_report.py

Writes `<report>_report.png` (or `harness_report.png` when reading from
stdin) next to the input and also prints the worst condition and the largest
calibration gap, the same "what does the headline not tell you" framing as
week10's `worst_cell`.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def load_report(path: str | None) -> dict:
    if path is None:
        text = sys.stdin.read()
    else:
        text = Path(path).read_text()
    return json.loads(text)


def worst_condition(per_condition_accuracy: dict) -> tuple[str, float]:
    valid = {c: a for c, a in per_condition_accuracy.items() if not math.isnan(a)}
    if not valid:
        raise ValueError("no condition has any valid predictions")
    cond = min(valid, key=valid.get)
    return cond, valid[cond]


def largest_calibration_gap(per_condition_gap: dict) -> tuple[str, float]:
    valid = {c: g for c, g in per_condition_gap.items() if not math.isnan(g)}
    if not valid:
        raise ValueError("no condition has any valid predictions")
    cond = max(valid, key=lambda c: abs(valid[c]))
    return cond, valid[cond]


def plot_report(report: dict, out_path: Path) -> None:
    metrics = report["metrics"]
    overall = metrics["overall_accuracy"]
    per_acc = metrics["per_condition_accuracy"]
    per_gap = metrics["per_condition_calibration_gap"]
    conditions = sorted(per_acc)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    # Panel 1: per-condition accuracy, with the headline as a reference line.
    accs = [per_acc[c] for c in conditions]
    colors = ["#4c72b0" if not math.isnan(a) else "lightgray" for a in accs]
    axes[0].bar(conditions, [0 if math.isnan(a) else a for a in accs], color=colors)
    axes[0].axhline(overall, color="crimson", linestyle="--", linewidth=1.5,
                    label=f"overall (headline) = {overall:.3f}")
    axes[0].set_ylim(0, 1.15)
    axes[0].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axes[0].set_ylabel("accuracy")
    axes[0].set_title("Per-condition accuracy vs. the headline number")
    axes[0].legend(fontsize=8, loc="lower left")
    for i, a in enumerate(accs):
        label = "n/a" if math.isnan(a) else f"{a:.2f}"
        axes[0].text(i, (0 if math.isnan(a) else a) + 0.03, label, ha="center", fontsize=9)

    # Panel 2: per-condition calibration gap (mean confidence - accuracy).
    # Positive = overconfident on that condition; negative = underconfident.
    gaps = [per_gap[c] for c in conditions]
    bar_colors = ["#c44e52" if (not math.isnan(g) and g > 0) else
                  "#55a868" if not math.isnan(g) else "lightgray" for g in gaps]
    axes[1].bar(conditions, [0 if math.isnan(g) else g for g in gaps], color=bar_colors)
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_ylabel("mean confidence - accuracy")
    axes[1].set_title("Calibration gap per condition\n(red = overconfident, green = underconfident)")
    for i, g in enumerate(gaps):
        label = "n/a" if math.isnan(g) else f"{g:+.2f}"
        y = 0 if math.isnan(g) else g
        axes[1].text(i, y + (0.01 if y >= 0 else -0.03), label, ha="center", fontsize=9)

    model_name = report.get("meta", {}).get("model_name", "<unspecified>")
    fig.suptitle(f"week03_perception_shift -- {model_name}")
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


def main() -> None:
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    report = load_report(arg)

    cond, acc = worst_condition(report["metrics"]["per_condition_accuracy"])
    gap_cond, gap = largest_calibration_gap(report["metrics"]["per_condition_calibration_gap"])
    print(f"overall (headline) accuracy: {report['metrics']['overall_accuracy']:.3f}")
    print(f"worst condition: {cond} (accuracy={acc:.3f}) -- this is what the headline hides.")
    print(f"largest calibration gap: {gap_cond} ({gap:+.3f}) -- "
          f"{'overconfident' if gap > 0 else 'underconfident'} on this condition.")

    out_path = Path(arg).with_name(Path(arg).stem + "_report.png") if arg else Path("harness_report.png")
    plot_report(report, out_path)


if __name__ == "__main__":
    main()
