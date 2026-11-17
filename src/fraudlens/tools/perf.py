"""Load-test reports: client latency from Locust, per-stage timings from ``/metrics``.

Server-side timings come from the service's own Prometheus histograms, scraped before
and after each run; the difference is that run alone. Quantiles are interpolated within
buckets the way Prometheus's ``histogram_quantile`` does, so they are estimates bounded by
the bucket edges. When a quantile falls past the last finite edge, the edge is reported as
a lower bound and the share of observations past it is given too.
"""

import csv
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SAMPLE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(?P<labels>[^}]*)\})?\s+(?P<value>\S+)$"
)
_LABEL = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')

SCORE_ROUTE = {"route": "/v1/transactions/score", "method": "POST", "status": "200"}


def _samples(text: str) -> list[tuple[str, dict[str, str], float]]:
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = _SAMPLE.match(line)
        if match:
            labels = dict(_LABEL.findall(match.group("labels") or ""))
            out.append((match.group("name"), labels, float(match.group("value"))))
    return out


@dataclass(frozen=True)
class Histogram:
    """A Prometheus histogram: cumulative ``(upper edge, count)`` buckets, sum and count.

    Edges and the sum are in seconds; the ``*_ms`` methods report milliseconds.
    """

    buckets: list[tuple[float, float]]
    sum: float
    count: float

    @classmethod
    def empty_like(cls, other: "Histogram") -> "Histogram":
        """A histogram with the same edges and nothing in it."""
        return cls([(edge, 0.0) for edge, _ in other.buckets], 0.0, 0.0)

    def __sub__(self, other: "Histogram") -> "Histogram":
        """The observations between two scrapes of the same histogram."""
        before = dict(other.buckets)
        return Histogram(
            [(edge, n - before.get(edge, 0.0)) for edge, n in self.buckets],
            self.sum - other.sum,
            self.count - other.count,
        )

    def mean_ms(self) -> float | None:
        """Mean in milliseconds, or ``None`` if empty."""
        return self.sum / self.count * 1000.0 if self.count else None

    def quantile_ms(self, q: float) -> float | None:
        """The ``q`` quantile in milliseconds, interpolated within its bucket."""
        if not self.count:
            return None
        rank = q * self.count
        lower, below = 0.0, 0.0
        for edge, cumulative in self.buckets:
            if cumulative >= rank:
                if math.isinf(edge):
                    return lower * 1000.0  # past the last finite edge: a lower bound
                if cumulative == below:
                    return edge * 1000.0
                return (lower + (edge - lower) * (rank - below) / (cumulative - below)) * 1000.0
            lower, below = edge, cumulative
        return lower * 1000.0

    def over_last_edge(self) -> float:
        """Share of observations above the last finite bucket edge."""
        finite = [n for edge, n in self.buckets if not math.isinf(edge)]
        if not self.count or not finite:
            return 0.0
        return (self.count - finite[-1]) / self.count


def parse_histograms(
    text: str, name: str, key: str, where: Mapping[str, str] | None = None
) -> dict[str, Histogram]:
    """Histograms called ``name`` in a scrape, keyed by the value of label ``key``.

    Args:
        text: The ``/metrics`` response.
        name: Metric name without the ``_bucket``, ``_sum`` and ``_count`` suffixes.
        key: The label that tells the histograms apart, for example ``stage``.
        where: Other labels a sample must have to be included.
    """
    buckets: dict[str, list[tuple[float, float]]] = {}
    sums: dict[str, float] = {}
    counts: dict[str, float] = {}
    for sample, labels, value in _samples(text):
        if where and any(labels.get(k) != v for k, v in where.items()):
            continue
        if key not in labels:
            continue
        group = labels[key]
        if sample == f"{name}_bucket":
            buckets.setdefault(group, []).append((float(labels["le"]), value))
        elif sample == f"{name}_sum":
            sums[group] = value
        elif sample == f"{name}_count":
            counts[group] = value
    return {
        group: Histogram(sorted(edges), sums.get(group, 0.0), counts.get(group, 0.0))
        for group, edges in buckets.items()
    }


def counter_deltas(before: str, after: str, name: str, key: str) -> dict[str, float]:
    """How much each labelled counter grew between two scrapes."""

    def values(text: str) -> dict[str, float]:
        return {labels[key]: v for s, labels, v in _samples(text) if s == name and key in labels}

    start = values(before)
    return {group: v - start.get(group, 0.0) for group, v in values(after).items()}


def locust_summary(stats_csv: Path) -> dict[str, Any]:
    """Client-side results from Locust's ``*_stats.csv``, aggregated row. Times in ms."""
    with stats_csv.open(newline="", encoding="utf-8") as handle:
        row = next(r for r in csv.DictReader(handle) if r["Name"] == "Aggregated")
    return {
        "requests": int(row["Request Count"]),
        "failures": int(row["Failure Count"]),
        "achieved_rps": round(float(row["Requests/s"]), 1),
        "avg_ms": float(row["Average Response Time"]),
        "p50_ms": float(row["50%"]),
        "p95_ms": float(row["95%"]),
        "p99_ms": float(row["99%"]),
        "max_ms": float(row["Max Response Time"]),
    }


def _diff(after: dict[str, Histogram], before: dict[str, Histogram]) -> dict[str, Histogram]:
    return {
        group: histogram - before.get(group, Histogram.empty_like(histogram))
        for group, histogram in after.items()
    }


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 2)


def run_summary(
    target_rps: int, duration_s: int, before: str, after: str, stats_csv: Path
) -> dict[str, Any]:
    """Everything measured in one run at one target rate."""
    stages = _diff(
        parse_histograms(after, "fraudlens_stage_seconds", "stage"),
        parse_histograms(before, "fraudlens_stage_seconds", "stage"),
    )
    score = _diff(
        parse_histograms(after, "fraudlens_request_seconds", "route", SCORE_ROUTE),
        parse_histograms(before, "fraudlens_request_seconds", "route", SCORE_ROUTE),
    ).get(SCORE_ROUTE["route"])
    return {
        "target_rps": target_rps,
        "duration_s": duration_s,
        **locust_summary(stats_csv),
        "server_score_ms": {
            "p50": _round(score.quantile_ms(0.5)) if score else None,
            "p95": _round(score.quantile_ms(0.95)) if score else None,
            "p99": _round(score.quantile_ms(0.99)) if score else None,
            "over_last_edge": round(score.over_last_edge(), 4) if score else None,
        },
        "stages": {
            name: {
                "count": int(h.count),
                "mean_ms": _round(h.mean_ms()),
                "p50_ms": _round(h.quantile_ms(0.5)),
                "p95_ms": _round(h.quantile_ms(0.95)),
            }
            for name, h in sorted(stages.items())
            if h.count
        },
        "decisions": counter_deltas(before, after, "fraudlens_decisions_total", "decision"),
    }


def _ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def render_markdown(report: Mapping[str, Any]) -> str:
    """The report as Markdown: machine, client latency per rate, then stages per rate."""
    machine = report["machine"]
    lines = [
        f"# Load test: {report['label']}",
        "",
        f"Generated by `python -m fraudlens.tools.loadtest` at commit `{report['commit']}`, "
        f"started {report['started_at']}. Nothing here is typed by hand.",
        "",
        f"- **Machine:** {machine['cpu']}, {machine['cpus']} CPUs, "
        f"{machine['memory_gb']} GB memory, {machine['os']}",
        f"- **Service:** {report['service']}",
        f"- **Stack:** {report['stack']}",
        "- **Traffic:** synthetic transactions on a fixed pool of cards, so each card's "
        "history grows during the run and the full hot path runs.",
        "",
        "## Client latency (Locust), milliseconds",
        "",
        "| Target rps | Achieved rps | Failures | p50 | p95 | p99 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for run in report["runs"]:
        lines.append(
            f"| {run['target_rps']} | {run['achieved_rps']} | {run['failures']} | "
            f"{_ms(run['p50_ms'])} | {_ms(run['p95_ms'])} | {_ms(run['p99_ms'])} |"
        )
    for run in report["runs"]:
        server = run["server_score_ms"]
        lines += [
            "",
            f"## Server stages at {run['target_rps']} rps, milliseconds",
            "",
            f"Scoring endpoint inside the service: p50 {_ms(server['p50'])}, "
            f"p95 {_ms(server['p95'])}, p99 {_ms(server['p99'])}.",
            "",
            "| Stage | Calls | Mean | p50 | p95 |",
            "| --- | --- | --- | --- | --- |",
        ]
        for name, stage in run["stages"].items():
            lines.append(
                f"| {name} | {stage['count']} | {_ms(stage['mean_ms'])} | "
                f"{_ms(stage['p50_ms'])} | {_ms(stage['p95_ms'])} |"
            )
    return "\n".join(lines) + "\n"
