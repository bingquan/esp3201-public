"""Task base class.

Each mini-assignment that uses the harness implements one Task subclass.
The contract is intentionally minimal so new tasks can be added in a single
file without changing the CLI.
"""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Any

from ..report import Report


class Task:
    task_id: str = ""
    description: str = ""

    def grade(self, submission: dict[str, Any]) -> Report:
        raise NotImplementedError

    @classmethod
    def load_groundtruth(cls, filename: str) -> dict[str, Any]:
        """Load a packaged ground-truth file from `esp3201_eval/data/`.

        Instructors replace the packaged file with a hidden test set for
        real grading. The version that ships with the repo is synthetic
        and exists so the harness runs end-to-end out of the box.
        """
        with resources.files("esp3201_eval").joinpath("data", filename).open("r") as f:
            return json.load(f)

    @staticmethod
    def load_submission(path: str | Path) -> dict[str, Any]:
        with Path(path).open("r") as f:
            return json.load(f)
