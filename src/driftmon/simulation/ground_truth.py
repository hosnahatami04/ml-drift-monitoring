"""Ground truth describing what each simulated stream actually contains."""

import math
from dataclasses import dataclass
from typing import Any

from driftmon.simulation.scenarios import ScenarioConfig, StreamParams

# A seasonal window counts as active while the oscillation is at least half
# its amplitude; near the zero crossings there is nothing to detect.
ACTIVE_AMPLITUDE_SHARE = 0.5


@dataclass(frozen=True)
class GroundTruth:
    scenario: str
    kind: str
    severity: str | None
    t0: int | None
    affected_features: tuple[str, ...]
    affects_label: bool
    active_windows: tuple[int, ...]
    params: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "kind": self.kind,
            "severity": self.severity,
            "t0": self.t0,
            "affected_features": list(self.affected_features),
            "affects_label": self.affects_label,
            "active_windows": list(self.active_windows),
            "params": self.params,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GroundTruth":
        return cls(
            scenario=payload["scenario"],
            kind=payload["kind"],
            severity=payload["severity"],
            t0=payload["t0"],
            affected_features=tuple(payload["affected_features"]),
            affects_label=payload["affects_label"],
            active_windows=tuple(payload["active_windows"]),
            params=dict(payload["params"]),
        )


def build_ground_truth(
    scenario: ScenarioConfig, severity: str | None, params: StreamParams
) -> GroundTruth:
    if scenario.kind == "control":
        return GroundTruth(
            scenario=scenario.name,
            kind=scenario.kind,
            severity=None,
            t0=None,
            affected_features=(),
            affects_label=False,
            active_windows=(),
            params={},
        )

    assert severity is not None
    level = scenario.severity[severity]
    t0 = params.drift_start
    drifted = tuple(range(t0, params.n_windows))

    if scenario.kind == "concept":
        return GroundTruth(
            scenario=scenario.name,
            kind=scenario.kind,
            severity=severity,
            t0=t0,
            affected_features=(),
            affects_label=True,
            active_windows=drifted,
            params={"p": level.p or 0.0},
        )

    assert scenario.feature is not None
    extra: dict[str, float] = {"beta": level.beta or 0.0}
    active = drifted
    if scenario.kind == "gradual_covariate":
        assert scenario.ramp_windows is not None
        extra["ramp_windows"] = float(scenario.ramp_windows)
    if scenario.kind == "seasonal":
        assert scenario.period is not None
        extra["period"] = float(scenario.period)
        active = tuple(
            i
            for i in drifted
            if abs(math.sin(2.0 * math.pi * (i - t0) / scenario.period)) >= ACTIVE_AMPLITUDE_SHARE
        )

    return GroundTruth(
        scenario=scenario.name,
        kind=scenario.kind,
        severity=severity,
        t0=t0,
        affected_features=(scenario.feature,),
        affects_label=False,
        active_windows=active,
        params=extra,
    )
