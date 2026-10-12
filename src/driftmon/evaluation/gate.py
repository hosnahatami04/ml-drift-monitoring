"""Regression gate for CI: compare a smoke evaluation against a stored
baseline and fail on a meaningful drop in detection or rise in false alarms.
"""

import json
from pathlib import Path
from typing import Any

from driftmon.evaluation.report import ResultsDocument

DETECTION_PREFIX = "detection_rate."
FALSE_ALARM_KEYS = ("false_alarm.stream", "false_alarm.per_window")
DEFAULT_TOLERANCE = 0.05


def gated_values(document: ResultsDocument) -> dict[str, float]:
    """The KPI values the gate watches, keyed like results.json."""
    values: dict[str, float] = {}
    for row in document.kpis:
        watched = row.key.startswith(DETECTION_PREFIX) or row.key in FALSE_ALARM_KEYS
        if watched and row.estimate.value is not None:
            values[row.key] = row.estimate.value
    return values


def baseline_from(
    document: ResultsDocument, tolerance: float = DEFAULT_TOLERANCE
) -> dict[str, Any]:
    if not document.smoke:
        raise ValueError("the KPI baseline must come from a smoke evaluation")
    return {
        "tolerance": tolerance,
        "seeds": document.provenance["seeds"],
        "kpis": gated_values(document),
    }


def check(document: ResultsDocument, baseline: dict[str, Any]) -> list[str]:
    """Violations of the baseline; empty means the gate passes. Tolerance is
    in absolute share, so 0.05 is five percentage points."""
    if document.provenance["seeds"] != baseline["seeds"]:
        return [
            f"smoke seeds {document.provenance['seeds']} differ from the baseline's "
            f"{baseline['seeds']}; regenerate the baseline"
        ]
    tolerance = float(baseline["tolerance"])
    current = gated_values(document)
    violations = []
    for key, reference in baseline["kpis"].items():
        if key not in current:
            violations.append(f"{key}: missing from the smoke results")
            continue
        value = current[key]
        if key.startswith(DETECTION_PREFIX) and value < reference - tolerance:
            violations.append(f"{key}: {value:.3f} dropped below baseline {reference:.3f}")
        if key in FALSE_ALARM_KEYS and value > reference + tolerance:
            violations.append(f"{key}: {value:.4f} rose above baseline {reference:.4f}")
    return violations


def load_baseline(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"KPI baseline not found: {path}")
    baseline: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return baseline


def write_baseline(path: Path, baseline: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(baseline, indent=2, sort_keys=True) + "\n", encoding="utf-8")
