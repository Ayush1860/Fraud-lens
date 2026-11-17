"""Run the load test at several rates and write the performance report.

For each target rate: scrape ``/metrics``, run Locust headless for a fixed time, scrape
again, and summarise the difference. The report records the machine it ran on.

    python -m fraudlens.tools.loadtest --label baseline --rates 50 200 500 --duration 60

Needs the ``load`` dependency group (``uv sync --group load``) and a running service.
"""

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from fraudlens.common.settings import get_settings
from fraudlens.tools.perf import render_markdown, run_summary

LOCUSTFILE = Path(__file__).with_name("locustfile.py")
"""Locust runs this file in its own process; the driver never imports it."""

PER_USER_RPS = 5
"""Requests per second each simulated client sends; users = target rate / this."""


def git_commit() -> str:
    """The current commit, or ``unknown``."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def machine_spec() -> dict[str, Any]:
    """CPU model, CPU count, memory and operating system of this machine."""
    cpu = platform.processor() or platform.machine()
    memory_gb: float | None = None
    cpuinfo, meminfo = Path("/proc/cpuinfo"), Path("/proc/meminfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    if meminfo.exists():
        for line in meminfo.read_text().splitlines():
            if line.startswith("MemTotal"):
                memory_gb = round(int(line.split()[1]) / 1024 / 1024, 1)
                break
    return {
        "cpu": cpu,
        "cpus": os.cpu_count(),
        "memory_gb": memory_gb,
        "os": f"{platform.system()} {platform.release()}",
    }


def run_locust(url: str, rate: int, duration_s: int, csv_prefix: Path, run_id: str) -> None:
    """Run Locust headless at ``rate`` requests per second for ``duration_s`` seconds."""
    users = max(1, math.ceil(rate / PER_USER_RPS))
    env = {
        **os.environ,
        "FRAUDLENS_LOAD_PER_USER_RPS": str(PER_USER_RPS),
        "FRAUDLENS_LOAD_RUN": run_id,
        "FRAUDLENS_API_KEY": get_settings().api_key.get_secret_value(),
    }
    subprocess.run(  # noqa: S603 - fixed arguments, no shell
        [
            sys.executable,
            "-m",
            "locust",
            "-f",
            str(LOCUSTFILE),
            "--headless",
            "--host",
            url,
            "--users",
            str(users),
            "--spawn-rate",
            str(max(1, users)),
            "--run-time",
            f"{duration_s}s",
            "--csv",
            str(csv_prefix),
            "--only-summary",
            "--loglevel",
            "WARNING",
        ],
        check=True,
        env=env,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for ``python -m fraudlens.tools.loadtest``."""
    parser = argparse.ArgumentParser(prog="python -m fraudlens.tools.loadtest")
    parser.add_argument("--url", default=None, help="service base URL")
    parser.add_argument("--label", required=True, help="report name, e.g. baseline")
    parser.add_argument("--rates", type=int, nargs="+", default=[50, 200, 500])
    parser.add_argument("--duration", type=int, default=60, help="seconds per rate")
    parser.add_argument("--warmup", type=int, default=15, help="unmeasured seconds first")
    parser.add_argument("--service", default="uvicorn, 1 worker", help="how the service ran")
    parser.add_argument(
        "--stack",
        default="Docker Compose: the service, Postgres and Redis containers on one machine",
        help="where the service and its stores ran",
    )
    parser.add_argument("--out", type=Path, default=Path("docs/perf"))
    args = parser.parse_args(argv)
    url = args.url or get_settings().service_url
    started = datetime.now(UTC)

    def scrape() -> str:
        return httpx.get(f"{url}/metrics", timeout=10).text

    runs = []
    with tempfile.TemporaryDirectory() as scratch:
        if args.warmup:
            run_locust(
                url, args.rates[0], args.warmup, Path(scratch) / "warmup", f"warm{started:%H%M%S}"
            )
        for rate in args.rates:
            prefix = Path(scratch) / f"rate{rate}"
            before = scrape()
            run_locust(url, rate, args.duration, prefix, f"r{rate}-{started:%H%M%S}")
            after = scrape()
            runs.append(
                run_summary(rate, args.duration, before, after, Path(f"{prefix}_stats.csv"))
            )

    report = {
        "label": args.label,
        "commit": git_commit(),
        "started_at": started.isoformat(timespec="seconds"),
        "machine": machine_spec(),
        "service": args.service,
        "stack": args.stack,
        "runs": runs,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{args.label}.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    (args.out / f"{args.label}.md").write_text(render_markdown(report), encoding="utf-8")
    print(render_markdown(report))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
