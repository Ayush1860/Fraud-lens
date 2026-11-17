"""The drift job: compute the drift report from the audit trail.

The service runs it in the background every ``drift_interval_s`` seconds and keeps the
latest report for ``/v1/monitoring/summary`` and ``/metrics``. It can also be run once:

    python -m fraudlens.monitoring.drift_job --out docs/results/drift.json
"""

import argparse
import asyncio
import json
import logging
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from fraudlens.monitoring.drift import drift_report
from fraudlens.training.bundle import ModelBundle

logger = logging.getLogger(__name__)

FeatureRows = list[tuple[dict[str, float | int | str], str]]


class FeatureSource(Protocol):
    """Where recent decisions come from: the audit repository."""

    async def recent_features(self, limit: int) -> FeatureRows:
        """Feature vectors and decisions of the newest decisions, newest first."""


def validation_mix(bundle: ModelBundle) -> dict[str, float]:
    """The share of each decision on the validation slice, from the bundle's metrics."""
    shares: Mapping[str, float] = bundle.documents["metrics.json"]["valid_late"]["decision_share"]
    return {kind: float(share) for kind, share in shares.items()}


def report_from_rows(
    rows: Sequence[tuple[Mapping[str, Any], str]], bundle: ModelBundle, window: int
) -> dict[str, Any]:
    """Split newest-first rows into the live and previous windows and build the report."""
    latest, previous = rows[:window], rows[window : 2 * window]
    report = drift_report(
        [features for features, _ in latest],
        [features for features, _ in previous],
        [decision for _, decision in latest],
        reference=bundle.documents["reference.json"],
        validation_mix=validation_mix(bundle),
    )
    return {
        "computed_at": datetime.now(UTC).isoformat(),
        "model_version": bundle.version,
        **report,
    }


async def compute(source: FeatureSource, bundle: ModelBundle, *, window: int) -> dict[str, Any]:
    """Read the newest ``2 * window`` decisions and build the drift report."""
    rows = await source.recent_features(2 * window)
    # Building the report is CPU work; keep the event loop free for scoring.
    return await asyncio.to_thread(report_from_rows, rows, bundle, window)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for ``python -m fraudlens.monitoring.drift_job``."""
    from sqlalchemy.ext.asyncio import create_async_engine

    from fraudlens.common.logging import configure_logging
    from fraudlens.common.settings import get_settings
    from fraudlens.service.repository import Repository
    from fraudlens.training.bundle import latest_bundle

    parser = argparse.ArgumentParser(prog="python -m fraudlens.monitoring.drift_job")
    parser.add_argument("--out", type=Path, default=None, help="write the report as JSON")
    parser.add_argument("--window", type=int, default=None)
    args = parser.parse_args(argv)
    settings = get_settings()
    configure_logging(settings.log_level)
    bundle = ModelBundle.load(settings.bundle_dir or latest_bundle(settings.models_dir))

    async def run() -> dict[str, Any]:
        engine = create_async_engine(settings.database_url)
        try:
            repository = Repository(engine, timeout_s=60.0)
            return await compute(repository, bundle, window=args.window or settings.drift_window)
        finally:
            await engine.dispose()

    report = asyncio.run(run())
    text = json.dumps(report, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text if not args.out else args.out)
    names = [alarm["name"] for alarm in report["alarms"]]
    logger.info("drift computed", extra={"alarms": names, "window": report["window"]})
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
