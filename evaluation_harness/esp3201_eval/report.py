"""Structured report types shared by all tasks."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Finding:
    """A single observation from grading.

    Severity levels:
        - "info": neutral observation, no penalty implied
        - "warn": something to investigate; appears in analysis-grading guidance
        - "fail": a structural problem (missing data, schema violation, fabricated claim)
    """

    code: str
    severity: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class Report:
    """Grading output for one submission.

    `metrics` are numeric quantities that students see and that the rubric uses.
    `findings` are structured notes that inform the analysis-grading rubric.
    `meta` carries task identity and student identifiers if provided.
    """

    task_id: str
    metrics: dict[str, Any] = field(default_factory=dict)
    findings: list[Finding] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def add(self, code: str, severity: str, message: str, **details: Any) -> None:
        self.findings.append(Finding(code=code, severity=severity, message=message, details=details))

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(asdict(self), indent=indent, sort_keys=True)

    def to_text(self) -> str:
        lines: list[str] = []
        lines.append(f"task: {self.task_id}")
        if self.meta:
            for k, v in sorted(self.meta.items()):
                lines.append(f"  {k}: {v}")
        lines.append("")
        lines.append("metrics:")
        for k, v in sorted(self.metrics.items()):
            lines.append(f"  {k}: {v}")
        lines.append("")
        if self.findings:
            lines.append("findings:")
            for f in self.findings:
                lines.append(f"  [{f.severity}] {f.code}: {f.message}")
                for k, v in sorted(f.details.items()):
                    lines.append(f"      {k}: {v}")
        else:
            lines.append("findings: none")
        return "\n".join(lines)
