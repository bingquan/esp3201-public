"""Task registry."""

from __future__ import annotations

from .base import Task
from .week03_perception_shift import WeekThreePerceptionShiftTask
from .week08_latency_accuracy import WeekEightLatencyAccuracyTask

_REGISTRY: dict[str, type[Task]] = {
    WeekThreePerceptionShiftTask.task_id: WeekThreePerceptionShiftTask,
    WeekEightLatencyAccuracyTask.task_id: WeekEightLatencyAccuracyTask,
}


def get(task_id: str) -> Task:
    if task_id not in _REGISTRY:
        known = ", ".join(sorted(_REGISTRY))
        raise KeyError(f"unknown task {task_id!r}; known: {known}")
    return _REGISTRY[task_id]()


def list_tasks() -> list[str]:
    return sorted(_REGISTRY)
