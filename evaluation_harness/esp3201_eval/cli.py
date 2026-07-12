"""Command-line interface for the ESP3201 evaluation harness.

Examples:
    esp3201-eval list
    esp3201-eval grade --task week03_perception_shift --submission path/to/sub.json
    esp3201-eval grade --task week08_latency_accuracy --submission path/to/sub.json --json
"""

from __future__ import annotations

import argparse
import sys

from . import tasks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="esp3201-eval", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="list known tasks")

    grade = sub.add_parser("grade", help="grade a submission")
    grade.add_argument("--task", required=True, help="task id, e.g. week03_perception_shift")
    grade.add_argument("--submission", required=True, help="path to submission JSON file")
    grade.add_argument("--json", action="store_true", help="emit report as JSON")
    grade.add_argument("--student-id", default=None, help="optional student identifier echoed in report.meta")

    args = parser.parse_args(argv)

    if args.cmd == "list":
        for t in tasks.list_tasks():
            cls = tasks.get(t).__class__
            print(f"{t}\t{cls.description}")
        return 0

    if args.cmd == "grade":
        task = tasks.get(args.task)
        submission = task.load_submission(args.submission)
        report = task.grade(submission)
        if args.student_id:
            report.meta["student_id"] = args.student_id
        if args.json:
            print(report.to_json())
        else:
            print(report.to_text())
        return 0 if not any(f.severity == "fail" for f in report.findings) else 2

    return 1


if __name__ == "__main__":
    sys.exit(main())
