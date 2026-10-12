"""Turn an evaluation run into results.json, results.md and figures.

results.json is the single source for every number quoted elsewhere. It must
be byte-identical across runs on the same machine, so it carries no
timestamps or absolute paths.
"""

import hashlib
import json
import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict

from driftmon.config import KpiTarget
from driftmon.evaluation import kpis
from driftmon.evaluation.harness import AblationArm, EvaluationRun
from driftmon.evaluation.kpis import Estimate, StreamOutcome

logger = logging.getLogger(__name__)

SCHEMA_VERSION: Literal[1] = 1

COVARIATE_KINDS = ("sudden_covariate", "gradual_covariate", "seasonal")

Comparator = Literal[">=", "<=", "report"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EstimateModel(_Strict):
    value: float | None
    ci_low: float | None
    ci_high: float | None
    n: int


class KpiRow(_Strict):
    key: str
    name: str
    scope: str
    unit: str
    estimate: EstimateModel
    target: float | None
    comparator: Comparator
    met: bool | None


class CellRow(_Strict):
    scenario: str
    severity: str
    n_streams: int
    detection_rate: EstimateModel
    median_ttd: EstimateModel
    alert_precision: EstimateModel
    harmful_share: EstimateModel
    median_lead_time: EstimateModel


class ControlSummary(_Strict):
    n_streams: int
    stream_false_alarm_rate: EstimateModel
    per_window_false_alarm_rate: EstimateModel
    streams_with_alarm_by_type: dict[str, int]
    alerts_by_type: dict[str, int]
    degradation_rate: EstimateModel


class AblationMetric(_Strict):
    metric: str
    scenario: str | None
    estimate: EstimateModel


class AblationRow(_Strict):
    group: str
    arm: str
    view: str
    window_size: int
    metrics: list[AblationMetric]
    control_streams_with_alarm_by_type: dict[str, int]
    timing: dict[str, int]


class StreamRow(_Strict):
    scenario: str
    kind: str
    severity: str | None
    seed: int
    alerts: list[tuple[int, str]]
    time_to_detect: int | None
    degradation_window: int | None
    lead_time: int | None


class ResultsDocument(_Strict):
    schema_version: Literal[1]
    smoke: bool
    provenance: dict[str, Any]
    kpis: list[KpiRow]
    cells: list[CellRow]
    control: ControlSummary
    detector_correctness: dict[str, float]
    ablations: list[AblationRow]
    streams: list[StreamRow]


def _est(estimate: Estimate) -> EstimateModel:
    return EstimateModel(**estimate.to_dict())


def _met(value: float | None, target: float | None, comparator: Comparator) -> bool | None:
    if value is None or target is None or comparator == "report":
        return None
    return value >= target if comparator == ">=" else value <= target


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def config_sha256(document: dict[str, Any]) -> str:
    payload = json.dumps(document, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _select(
    outcomes: Sequence[StreamOutcome],
    *,
    scenario: str | None = None,
    severity: str | None = None,
    kinds: Sequence[str] | None = None,
    drifted: bool | None = None,
) -> list[StreamOutcome]:
    selected = []
    for o in outcomes:
        if scenario is not None and o.scenario != scenario:
            continue
        if severity is not None and o.severity != severity:
            continue
        if kinds is not None and o.kind not in kinds:
            continue
        if drifted is not None and (o.t0 is not None) != drifted:
            continue
        selected.append(o)
    return selected


def build_results(
    run: EvaluationRun,
    *,
    smoke: bool,
    target_severity: str,
    targets: dict[str, KpiTarget],
    provenance: dict[str, Any],
    correctness: dict[str, float],
    samples: int,
    seed: int,
) -> ResultsDocument:
    outcomes = run.outcomes
    control = _select(outcomes, drifted=False)
    drifted = _select(outcomes, drifted=True)
    at_target = _select(drifted, severity=target_severity)

    def by_scenario(name: str) -> list[StreamOutcome]:
        return _select(at_target, scenario=name)

    def row(key: str, name: str, scope: str, unit: str, estimate: Estimate) -> KpiRow:
        goal = targets.get(key, KpiTarget())
        target, comparator = goal.target, goal.comparator
        return KpiRow(
            key=key,
            name=name,
            scope=scope,
            unit=unit,
            estimate=_est(estimate),
            target=target,
            comparator=comparator,
            met=_met(estimate.value, target, comparator),
        )

    scope = f"{target_severity} severity, evaluation seeds"
    covariate_all = _select(drifted, kinds=COVARIATE_KINDS)
    concept = _select(drifted, scenario="concept", severity=target_severity)
    correctness_worst = max(correctness.values()) if correctness else float("nan")
    rows = [
        row(
            "detection_rate.sudden_covariate",
            "Detection rate, sudden covariate",
            scope,
            "share",
            kpis.detection_rate(by_scenario("sudden_covariate"), samples, seed),
        ),
        row(
            "detection_rate.gradual_covariate",
            "Detection rate, gradual covariate",
            scope,
            "share",
            kpis.detection_rate(by_scenario("gradual_covariate"), samples, seed),
        ),
        row(
            "detection_rate.concept",
            "Detection rate, concept (performance_drift only)",
            scope,
            "share",
            kpis.detection_rate(by_scenario("concept"), samples, seed),
        ),
        row(
            "median_ttd.sudden_covariate",
            "Median time to detect, sudden covariate",
            scope,
            "windows",
            kpis.median_ttd(by_scenario("sudden_covariate"), samples, seed),
        ),
        row(
            "median_ttd.gradual_covariate",
            "Median time to detect, gradual covariate",
            scope,
            "windows",
            kpis.median_ttd(by_scenario("gradual_covariate"), samples, seed),
        ),
        row(
            "median_ttd.concept",
            "Median time to detect, concept",
            scope,
            "windows",
            kpis.median_ttd(by_scenario("concept"), samples, seed),
        ),
        row(
            "false_alarm.stream",
            "Stream false alarm rate (control)",
            "control streams, evaluation seeds",
            "share",
            kpis.stream_false_alarm_rate(control, samples, seed),
        ),
        row(
            "false_alarm.per_window",
            "Per-window false alarm rate (control)",
            "control streams, evaluation seeds",
            "alerts per window",
            kpis.per_window_false_alarm_rate(control, samples, seed),
        ),
        row(
            "alert_precision",
            "Alert precision, drifted streams",
            f"{scope}, all drift scenarios pooled",
            "share",
            kpis.alert_precision(at_target, samples, seed),
        ),
        row(
            "lead_time.harmful_covariate",
            "Median lead time, harmful covariate drift",
            "covariate scenarios, all severities, streams with degradation",
            "windows",
            kpis.median_lead_time(covariate_all, samples, seed),
        ),
        row(
            "harmful_share",
            "Harmful-drift share",
            f"{scope}, all drift scenarios pooled",
            "share",
            kpis.harmful_share(at_target, samples, seed),
        ),
        row(
            "input_silence.concept",
            "Input-detector alert rate on concept minus control (zero by construction)",
            f"{scope}, paired by seed",
            "share",
            kpis.paired_rate_difference(concept, control, kpis.INPUT_ALERT_TYPES, samples, seed),
        ),
        row(
            "detector_correctness",
            "Largest deviation from scipy (KS, chi-square)",
            "random inputs",
            "absolute",
            Estimate(value=correctness_worst, ci_low=None, ci_high=None, n=len(correctness)),
        ),
    ]

    cells = []
    for name in sorted({o.scenario for o in drifted}):
        for level in run.severities:
            subset = _select(drifted, scenario=name, severity=level)
            if not subset:
                continue
            cells.append(
                CellRow(
                    scenario=name,
                    severity=level,
                    n_streams=len(subset),
                    detection_rate=_est(kpis.detection_rate(subset, samples, seed)),
                    median_ttd=_est(kpis.median_ttd(subset, samples, seed)),
                    alert_precision=_est(kpis.alert_precision(subset, samples, seed)),
                    harmful_share=_est(kpis.harmful_share(subset, samples, seed)),
                    median_lead_time=_est(kpis.median_lead_time(subset, samples, seed)),
                )
            )

    types = sorted(kpis.ALL_ALERT_TYPES)
    control_summary = ControlSummary(
        n_streams=len(control),
        stream_false_alarm_rate=_est(kpis.stream_false_alarm_rate(control, samples, seed)),
        per_window_false_alarm_rate=_est(kpis.per_window_false_alarm_rate(control, samples, seed)),
        streams_with_alarm_by_type={
            t: sum(kpis.has_alert(o, frozenset({t})) for o in control) for t in types
        },
        alerts_by_type={t: sum(a.type == t for o in control for a in o.alerts) for t in types},
        degradation_rate=_est(kpis.harmful_share(control, samples, seed)),
    )

    streams = [
        StreamRow(
            scenario=o.scenario,
            kind=o.kind,
            severity=o.severity,
            seed=o.seed,
            alerts=[(a.window, a.type) for a in o.alerts],
            time_to_detect=kpis.time_to_detect(o),
            degradation_window=o.degradation_window,
            lead_time=kpis.lead_time(o),
        )
        for o in outcomes
    ]

    return ResultsDocument(
        schema_version=SCHEMA_VERSION,
        smoke=smoke,
        provenance=provenance,
        kpis=rows,
        cells=cells,
        control=control_summary,
        detector_correctness=correctness,
        ablations=[_ablation_row(arm, samples, seed) for arm in run.ablations],
        streams=streams,
    )


def _ablation_row(arm: AblationArm, samples: int, seed: int) -> AblationRow:
    control = _select(arm.outcomes, drifted=False)
    drifted = _select(arm.outcomes, drifted=True)
    metrics: list[AblationMetric] = []
    for name in sorted({o.scenario for o in drifted}):
        subset = _select(drifted, scenario=name)
        ttd = kpis.median_ttd(subset, samples, seed)
        metrics.append(
            AblationMetric(
                metric="detection_rate",
                scenario=name,
                estimate=_est(kpis.detection_rate(subset, samples, seed)),
            )
        )
        metrics.append(AblationMetric(metric="median_ttd", scenario=name, estimate=_est(ttd)))
        metrics.append(
            AblationMetric(
                metric="median_ttd_rows",
                scenario=name,
                estimate=_est(_scaled(ttd, arm.window_size)),
            )
        )
    metrics.append(
        AblationMetric(
            metric="stream_false_alarm_rate",
            scenario=None,
            estimate=_est(kpis.stream_false_alarm_rate(control, samples, seed)),
        )
    )
    metrics.append(
        AblationMetric(
            metric="per_window_false_alarm_rate",
            scenario=None,
            estimate=_est(kpis.per_window_false_alarm_rate(control, samples, seed)),
        )
    )
    return AblationRow(
        group=arm.group,
        arm=arm.arm,
        view=arm.view,
        window_size=arm.window_size,
        metrics=metrics,
        control_streams_with_alarm_by_type={
            t: sum(kpis.has_alert(o, frozenset({t})) for o in control)
            for t in sorted(kpis.ALL_ALERT_TYPES)
        },
        timing=arm.timing,
    )


def _scaled(estimate: Estimate, factor: float) -> Estimate:
    def scale(v: float | None) -> float | None:
        return None if v is None else v * factor

    return Estimate(
        value=scale(estimate.value),
        ci_low=scale(estimate.ci_low),
        ci_high=scale(estimate.ci_high),
        n=estimate.n,
    )


def validate_results(payload: dict[str, Any]) -> ResultsDocument:
    return ResultsDocument.model_validate(payload)


def write_json(path: Path, document: ResultsDocument) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = document.model_dump(mode="json")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}%"


def _num(value: float | None, digits: int = 1) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _fmt(estimate: EstimateModel, unit: str) -> str:
    fmt: Callable[[float | None], str]
    if unit == "share":
        fmt = _pct
    elif unit == "alerts per window":

        def fmt(v: float | None) -> str:
            return "n/a" if v is None else f"{100 * v:.2f}%"

    elif unit == "absolute":

        def fmt(v: float | None) -> str:
            return "n/a" if v is None else f"{v:.1e}"

    else:
        fmt = _num
    text = fmt(estimate.value)
    if estimate.ci_low is not None and estimate.ci_high is not None:
        text += f" [{fmt(estimate.ci_low)}, {fmt(estimate.ci_high)}]"
    return text


def _target(row: KpiRow) -> str:
    if row.comparator == "report" or row.target is None:
        return "report"
    if row.unit == "share":
        target = _pct(row.target)
    elif row.unit == "alerts per window":
        target = f"{100 * row.target:.1f}%"
    elif row.unit == "absolute":
        target = f"{row.target:.0e}"
    else:
        target = f"{row.target:g}"
    symbol = "≥" if row.comparator == ">=" else "≤"
    return f"{symbol} {target}"


def _status(met: bool | None) -> str:
    return {True: "met", False: "**missed**", None: "-"}[met]


def render_markdown(document: ResultsDocument) -> str:
    p = document.provenance
    lines = [
        "# Evaluation results" + (" (smoke subset)" if document.smoke else ""),
        "",
        f"Written by `driftmon evaluate{' --smoke' if document.smoke else ''}` from "
        "`results.json` in the same directory; every number below comes from that file.",
        "",
        f"- Evaluation seeds {p['seeds'][0]}-{p['seeds'][-1]} ({len(p['seeds'])} streams per "
        f"cell), severities: {', '.join(p['severities'])}",
        f"- Stream: {p['stream']['n_windows']} windows of {p['stream']['window_size']} rows, "
        f"drift from window {p['stream']['drift_start']}, label delay "
        f"{p['stream']['label_delay']}, detection horizon {p['horizon']}",
        f"- Thresholds frozen by calibration on seeds {p['calibration_seeds'][0]}-"
        f"{p['calibration_seeds'][-1]}: alpha {p['thresholds']['alpha']} "
        f"({p['thresholds']['correction']}), k={p['thresholds']['k_features']}, "
        f"KS D ≥ {p['thresholds']['min_effect']['ks_d']}, "
        f"PSI ≥ {p['thresholds']['min_effect']['psi']}, "
        f"Cramer's V ≥ {p['thresholds']['min_effect']['cramers_v']}, "
        f"Page-Hinkley delta {p['thresholds']['page_hinkley']['delta']} / "
        f"lambda {p['thresholds']['page_hinkley']['lambda']}",
        f"- Model artifact `{p['artifact']}` (model.pkl sha256 `{p['model_sha256'][:12]}`), "
        f"reference AUC {p['reference_auc']:.4f}; config sha256 `{p['config_sha256'][:12]}`",
        f"- 95% intervals: percentile bootstrap over streams, {p['bootstrap']['samples']} "
        f"resamples, seed {p['bootstrap']['seed']}",
        "",
        "## Detection KPIs",
        "",
        "| KPI | Scope | Value [95% CI] | Target | Status |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in document.kpis:
        unit = "" if row.unit in ("share", "absolute", "alerts per window") else f" {row.unit}"
        lines.append(
            f"| {row.name} | {row.scope} | {_fmt(row.estimate, row.unit)}{unit} "
            f"| {_target(row)} | {_status(row.met)} |"
        )

    lines += [
        "",
        "Time to detect counts windows from the drift start to the first correct alert",
        "(0 = caught in the first drifted window). A performance alert is raised when the",
        "window's labels arrive, so concept detection cannot be faster than the label delay.",
        "",
        "## By scenario and severity",
        "",
        "| Scenario | Severity | Detection rate | Median TTD | Alert precision "
        "| Harmful share | Median lead time |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for cell in document.cells:
        lead = _fmt(cell.median_lead_time, "windows")
        if cell.median_lead_time.n == 0:
            lead = "no harmful streams"
        lines.append(
            f"| {cell.scenario} | {cell.severity} | {_fmt(cell.detection_rate, 'share')} "
            f"| {_fmt(cell.median_ttd, 'windows')} | {_fmt(cell.alert_precision, 'share')} "
            f"| {_fmt(cell.harmful_share, 'share')} | {lead} |"
        )

    c = document.control
    lines += [
        "",
        "## Control streams",
        "",
        f"- {c.n_streams} streams with no injected drift",
        f"- Stream false alarm rate: {_fmt(c.stream_false_alarm_rate, 'share')}",
        f"- Per-window false alarm rate: "
        f"{_fmt(c.per_window_false_alarm_rate, 'alerts per window')}",
        "- Streams with at least one alarm, by type: "
        + ", ".join(f"{t} {n}" for t, n in c.streams_with_alarm_by_type.items()),
        "- Alerts by type: " + ", ".join(f"{t} {n}" for t, n in c.alerts_by_type.items()),
        f"- Streams whose rolling AUC dropped below the degradation line anyway: "
        f"{_fmt(c.degradation_rate, 'share')} (noise floor for harmful share)",
        "",
        "## Detector correctness",
        "",
        "Largest absolute deviation from scipy over random inputs:",
        "",
    ]
    lines += [f"- {k}: {v:.2e}" for k, v in sorted(document.detector_correctness.items())]

    if document.ablations:
        lines += _ablation_markdown(document.ablations)

    lines += ["", "## What didn't work", ""]
    missed = [r for r in document.kpis if r.met is False]
    if missed:
        for r in missed:
            lines.append(
                f"- {r.name}: {_fmt(r.estimate, r.unit)} against a target of {_target(r)}."
            )
    else:
        lines.append("- Every KPI with a target met it on the evaluation seeds.")
    lines += _diagnostics(document)

    if not document.smoke:
        lines += [
            "",
            "## Figures",
            "",
            "![Detection rate by severity](figures/detection_vs_severity.png)",
            "![Time to detect](figures/ttd_by_scenario.png)",
            "![Window size ablation](figures/window_size_ablation.png)",
            "![Lead time](figures/lead_time.png)",
        ]
    lines.append("")
    return "\n".join(lines)


def _ablation_markdown(ablations: Sequence[AblationRow]) -> list[str]:
    lines = [
        "",
        "## Ablations",
        "",
        "All arms reuse the frozen detector thresholds; none is re-tuned, so an arm's false",
        "alarm rate moves along with its sensitivity. Method, gate and correction arms change",
        "only the input-side decision and are scored on feature_drift alerts alone. The",
        "window-size arms keep the traffic volume fixed: drift start, horizon, label delay,",
        "alert cooldown and scenario timing are scaled in rows (the table below lists what",
        "each arm actually ran, including where a value had to be rounded), and they are",
        "scored on all alerts, so their time to detect is also given in rows.",
        "",
        "Detection counts any relevant alert inside the horizon, so an arm with a high",
        'control false alarm rate also "detects" by chance; read detection next to the',
        "false alarm columns.",
    ]
    for group in dict.fromkeys(a.group for a in ablations):
        arms = [a for a in ablations if a.group == group]
        scenarios = sorted({m.scenario for a in arms for m in a.metrics if m.scenario})
        rows_col = group == "window_size"
        header = "| Arm | " + " | ".join(
            f"{s} detection | {s} median TTD" + (" (rows)" if rows_col else "") for s in scenarios
        )
        header += " | Control stream FA | Control per-window FA |"
        lines += ["", f"### {group.replace('_', ' ')} ({arms[0].view})", "", header]
        lines.append("| --- " * (2 * len(scenarios) + 3) + "|")
        for arm in arms:
            cells = []
            for s in scenarios:
                metric = {(m.metric, m.scenario): m.estimate for m in arm.metrics}
                cells.append(_fmt(metric[("detection_rate", s)], "share"))
                ttd_key = "median_ttd_rows" if rows_col else "median_ttd"
                cells.append(_fmt(metric[(ttd_key, s)], "rows" if rows_col else "windows"))
            metric = {(m.metric, m.scenario): m.estimate for m in arm.metrics}
            cells.append(_fmt(metric[("stream_false_alarm_rate", None)], "share"))
            cells.append(_fmt(metric[("per_window_false_alarm_rate", None)], "alerts per window"))
            lines.append(f"| {arm.arm} | " + " | ".join(cells) + " |")
        lines.append("")
        lines.append("Control streams with an alarm, by type:")
        lines.append("")
        for arm in arms:
            by_type = ", ".join(
                f"{t} {n}" for t, n in arm.control_streams_with_alarm_by_type.items()
            )
            lines.append(f"- {arm.arm}: {by_type}")
        if rows_col:
            lines += _timing_markdown(arms)
    return lines


def _timing_markdown(arms: Sequence[AblationRow]) -> list[str]:
    keys = [
        ("n_windows", "Stream"),
        ("drift_start", "Drift start"),
        ("label_delay", "Label delay"),
        ("horizon", "Horizon"),
        ("cooldown_windows", "Alert cooldown"),
        ("ramp_windows", "Gradual ramp"),
        ("seasonal_period", "Seasonal period"),
    ]
    lines = [
        "",
        "Timing per arm, in rows (window size x windows):",
        "",
        "| Arm | " + " | ".join(label for _, label in keys) + " |",
        "| --- " * (len(keys) + 1) + "|",
    ]
    for arm in arms:
        size = arm.timing.get("window_size", arm.window_size)
        cells = [f"{arm.timing[k] * size:,}" if k in arm.timing else "-" for k, _ in keys]
        lines.append(f"| {arm.arm} | " + " | ".join(cells) + " |")
    return lines


def _diagnostics(document: ResultsDocument) -> list[str]:
    """Facts from the run that explain the misses; the reasoning around them
    lives in ADR 0008."""
    lines: list[str] = []
    by_key = {r.key: r for r in document.kpis}
    stream_fa = by_key.get("false_alarm.stream")
    if stream_fa is not None and stream_fa.met is False:
        types = document.control.streams_with_alarm_by_type
        lines.append(
            "- Control streams with an alarm, by alert type: "
            + ", ".join(f"{t} {n}" for t, n in types.items())
            + f" (of {document.control.n_streams})."
        )
    for row in document.kpis:
        if row.key.startswith("detection_rate.") and row.met is False:
            scenario = row.key.split(".", 1)[1]
            level = document.provenance["target_severity"]
            missed = [
                s.seed
                for s in document.streams
                if s.scenario == scenario and s.severity == level and s.time_to_detect is None
            ]
            lines.append(f"- Undetected {scenario} streams at {level} severity: seeds {missed}.")
    if stream_fa is not None and stream_fa.target is not None:
        noisy = []
        for arm in document.ablations:
            if arm.group != "window_size":
                continue
            fa = next(m.estimate for m in arm.metrics if m.metric == "stream_false_alarm_rate")
            if fa.value is not None and fa.value > stream_fa.target:
                noisy.append(f"{arm.arm} ({_pct(fa.value)})")
        if noisy:
            lines.append(
                "- With the frozen thresholds, smaller windows exceed the stream false alarm "
                f"target: {', '.join(noisy)}. Another window size needs its own calibration."
            )
    lead = by_key.get("lead_time.harmful_covariate")
    if lead is not None and lead.estimate.n:
        lines.append(
            f"- Lead time is measured on {lead.estimate.n} covariate streams whose rolling AUC "
            "actually fell below the degradation line."
        )
    return lines


def write_markdown(path: Path, document: ResultsDocument) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown(document), encoding="utf-8")


SERIES_COLORS = {
    "sudden_covariate": "#2a78d6",
    "gradual_covariate": "#eb6834",
    "concept": "#1baf7a",
    "seasonal": "#eda100",
}
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"


def _err(upper: float | None, lower: float | None) -> float:
    """Length of one side of an error bar; a missing bound draws none. A
    bound of exactly 0.0 is a real bound, not a missing one."""
    if upper is None or lower is None:
        return 0.0
    return upper - lower


def write_figures(out_dir: Path, document: ResultsDocument) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT_SECONDARY,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
            "text.color": TEXT_PRIMARY,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
        }
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    # No metadata: the PNG bytes then depend only on the data.
    save_kwargs: dict[str, Any] = {
        "dpi": 150,
        "bbox_inches": "tight",
        "metadata": {"Software": None},
    }

    # Detection rate against severity, one panel per scenario: several
    # scenarios sit at 1.0 and would hide each other on shared axes.
    severities = list(dict.fromkeys(c.severity for c in document.cells))
    present = [s for s in SERIES_COLORS if any(c.scenario == s for c in document.cells)]
    fig, axes = plt.subplots(1, len(present), figsize=(3 * len(present), 3.4), sharey=True)
    for ax, scenario in zip(np.atleast_1d(axes), present, strict=True):
        cells = [c for c in document.cells if c.scenario == scenario]
        x = [severities.index(c.severity) for c in cells]
        y = [c.detection_rate.value or 0.0 for c in cells]
        low = [_err(y_i, c.detection_rate.ci_low) for y_i, c in zip(y, cells, strict=True)]
        high = [_err(c.detection_rate.ci_high, y_i) for y_i, c in zip(y, cells, strict=True)]
        ax.errorbar(
            x,
            y,
            yerr=[low, high],
            color=SERIES_COLORS[scenario],
            linewidth=2,
            marker="o",
            markersize=6,
            capsize=3,
        )
        ax.set_xticks(range(len(severities)), severities)
        ax.set_xlim(-0.4, len(severities) - 0.6)
        ax.set_ylim(0, 1.05)
        ax.set_title(scenario, loc="left", fontsize=10)
    np.atleast_1d(axes)[0].set_ylabel("Detection rate (95% CI)")
    fig.suptitle("Detection rate by severity", x=0.01, ha="left")
    path = out_dir / "detection_vs_severity.png"
    fig.savefig(path, **save_kwargs)
    plt.close(fig)
    written.append(path)

    # Time to detect at the target severity, one row of dots per scenario.
    fig, ax = plt.subplots(figsize=(7, 4.2))
    target = document.provenance["target_severity"]
    scenarios = [s for s in SERIES_COLORS if any(c.scenario == s for c in document.cells)]
    for row_index, scenario in enumerate(scenarios):
        ttds = sorted(
            s.time_to_detect
            for s in document.streams
            if s.scenario == scenario and s.severity == target and s.time_to_detect is not None
        )
        totals = {t: ttds.count(t) for t in set(ttds)}
        step = 0.8 / max(max(totals.values(), default=1), 10)
        counts: dict[int, int] = {}
        xs, ys = [], []
        for t in ttds:
            offset = counts.get(t, 0)
            counts[t] = offset + 1
            xs.append(t)
            ys.append(row_index + step * (offset - (totals[t] - 1) / 2))
        ax.scatter(xs, ys, s=22, color=SERIES_COLORS[scenario], edgecolors=SURFACE, linewidths=0.8)
    ax.set_yticks(range(len(scenarios)), scenarios)
    ax.set_xlabel("Windows from drift start to first correct alert")
    ax.set_title(f"Time to detect, {target} severity (one dot per stream)", loc="left")
    path = out_dir / "ttd_by_scenario.png"
    fig.savefig(path, **save_kwargs)
    plt.close(fig)
    written.append(path)

    window_rows = [a for a in document.ablations if a.group == "window_size"]
    if window_rows:
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
        sizes = [a.window_size for a in window_rows]
        for scenario, color in SERIES_COLORS.items():
            det, ttd = [], []
            for arm in window_rows:
                metric = {(m.metric, m.scenario): m.estimate for m in arm.metrics}
                det.append(metric.get(("detection_rate", scenario)))
                ttd.append(metric.get(("median_ttd_rows", scenario)))
            if all(d is None for d in det):
                continue
            axes[0].plot(
                sizes,
                [d.value if d else None for d in det],
                color=color,
                marker="o",
                linewidth=2,
                label=scenario,
            )
            axes[1].plot(
                sizes, [t.value if t else None for t in ttd], color=color, marker="o", linewidth=2
            )
        fa = []
        for arm in window_rows:
            metric = {(m.metric, m.scenario): m.estimate for m in arm.metrics}
            fa.append(metric[("stream_false_alarm_rate", None)].value)
        axes[2].plot(sizes, fa, color=TEXT_SECONDARY, marker="o", linewidth=2)
        axes[2].axhline(0.10, color=TEXT_SECONDARY, linestyle="--", linewidth=1)
        axes[2].annotate(
            "target",
            (sizes[-1], 0.10),
            xytext=(0, 4),
            textcoords="offset points",
            ha="right",
            color=TEXT_SECONDARY,
            fontsize=9,
        )
        titles = ("Detection rate", "Median time to detect (rows)", "Control stream false alarms")
        for ax, title in zip(axes, titles, strict=True):
            ax.set_xscale("log", base=2)
            ax.set_xticks(sizes, [str(s) for s in sizes])
            ax.set_xlabel("Window size (rows)")
            ax.set_title(title, loc="left")
        axes[0].set_ylim(0, 1.05)
        axes[0].legend(frameon=False, fontsize=8, loc="lower right")
        path = out_dir / "window_size_ablation.png"
        fig.savefig(path, **save_kwargs)
        plt.close(fig)
        written.append(path)

    # Lead time over covariate streams that actually degraded.
    leads = [
        s.lead_time
        for s in document.streams
        if s.kind in COVARIATE_KINDS and s.lead_time is not None
    ]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    if leads:
        bins = range(min(leads) - 1, max(leads) + 2)
        ax.hist(leads, bins=bins, color=SERIES_COLORS["sudden_covariate"], edgecolor=SURFACE)
        ax.axvline(0, color=TEXT_SECONDARY, linewidth=1)
        ax.set_xlabel("Windows between first alert and measurable AUC drop")
        ax.set_ylabel("Streams")
    else:
        ax.text(0.5, 0.5, "No covariate stream degraded", ha="center", transform=ax.transAxes)
    ax.set_title("Lead time on harmful covariate drift (positive = warned early)", loc="left")
    path = out_dir / "lead_time.png"
    fig.savefig(path, **save_kwargs)
    plt.close(fig)
    written.append(path)
    return written
