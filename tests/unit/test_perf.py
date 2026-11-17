"""The load-test report: Prometheus histogram maths, Locust's CSV and the rendering."""

from pathlib import Path

import pytest

from fraudlens.tools.perf import (
    Histogram,
    counter_deltas,
    locust_summary,
    parse_histograms,
    render_markdown,
    run_summary,
)

BEFORE = """
# HELP fraudlens_stage_seconds Latency of each stage.
# TYPE fraudlens_stage_seconds histogram
fraudlens_stage_seconds_bucket{stage="model",le="0.001"} 10.0
fraudlens_stage_seconds_bucket{stage="model",le="0.005"} 10.0
fraudlens_stage_seconds_bucket{stage="model",le="+Inf"} 10.0
fraudlens_stage_seconds_count{stage="model"} 10.0
fraudlens_stage_seconds_sum{stage="model"} 0.005
fraudlens_decisions_total{decision="approve"} 9.0
fraudlens_decisions_total{decision="review"} 1.0
fraudlens_degraded_responses_total 1.0
"""

SCORE = 'route="/v1/transactions/score",method="POST",status="200"'

AFTER = f"""
fraudlens_stage_seconds_bucket{{stage="model",le="0.001"}} 60.0
fraudlens_stage_seconds_bucket{{stage="model",le="0.005"}} 110.0
fraudlens_stage_seconds_bucket{{stage="model",le="+Inf"}} 110.0
fraudlens_stage_seconds_count{{stage="model"}} 110.0
fraudlens_stage_seconds_sum{{stage="model"}} 0.205
fraudlens_stage_seconds_bucket{{stage="database",le="0.001"}} 0.0
fraudlens_stage_seconds_bucket{{stage="database",le="0.005"}} 40.0
fraudlens_stage_seconds_bucket{{stage="database",le="+Inf"}} 100.0
fraudlens_stage_seconds_count{{stage="database"}} 100.0
fraudlens_stage_seconds_sum{{stage="database"}} 1.0
fraudlens_request_seconds_bucket{{{SCORE},le="0.005"}} 50.0
fraudlens_request_seconds_bucket{{{SCORE},le="0.01"}} 100.0
fraudlens_request_seconds_bucket{{{SCORE},le="+Inf"}} 100.0
fraudlens_request_seconds_count{{{SCORE}}} 100.0
fraudlens_request_seconds_sum{{{SCORE}}} 0.6
fraudlens_decisions_total{{decision="approve"}} 99.0
fraudlens_decisions_total{{decision="review"}} 11.0
fraudlens_degraded_responses_total 5.0
"""

LOCUST_CSV = (
    "Type,Name,Request Count,Failure Count,Median Response Time,Average Response Time,"
    "Min Response Time,Max Response Time,Average Content Size,Requests/s,Failures/s,"
    "50%,66%,75%,80%,90%,95%,98%,99%,99.9%,99.99%,100%\n"
    "POST,/v1/transactions/score,5990,10,8,9.5,3,120,180,99.8,0.17,8,9,10,11,13,16,22,31,80,110,120\n"
    ",Aggregated,5990,10,8,9.5,3,120,180,99.8,0.17,8,9,10,11,13,16,22,31,80,110,120\n"
)


def test_histograms_are_parsed_by_label() -> None:
    stages = parse_histograms(AFTER, "fraudlens_stage_seconds", "stage")

    assert set(stages) == {"model", "database"}
    assert stages["model"].count == 110
    assert stages["model"].buckets == [(0.001, 60.0), (0.005, 110.0), (float("inf"), 110.0)]


def test_a_run_is_the_difference_between_two_scrapes() -> None:
    before = parse_histograms(BEFORE, "fraudlens_stage_seconds", "stage")
    after = parse_histograms(AFTER, "fraudlens_stage_seconds", "stage")

    model = after["model"] - before["model"]
    database = after["database"] - before.get("database", Histogram.empty_like(after["database"]))

    assert model.count == 100
    assert model.mean_ms() == pytest.approx(2.0)
    assert model.buckets[0] == (0.001, 50.0)
    # Half the database calls are over the last finite edge: the p95 is only a lower bound.
    assert database.quantile_ms(0.30) == pytest.approx(4.0)
    assert database.quantile_ms(0.95) == pytest.approx(5.0)
    assert database.over_last_edge() == 0.6


def test_an_empty_histogram_has_no_quantile() -> None:
    empty = Histogram([(0.001, 0.0), (float("inf"), 0.0)], 0.0, 0.0)

    assert empty.quantile_ms(0.5) is None
    assert empty.mean_ms() is None


def test_counters_are_differenced() -> None:
    assert counter_deltas(BEFORE, AFTER, "fraudlens_decisions_total", "decision") == {
        "approve": 90.0,
        "review": 10.0,
    }


def test_locust_csv_is_read_from_the_aggregated_row(tmp_path: Path) -> None:
    path = tmp_path / "run_stats.csv"
    path.write_text(LOCUST_CSV, encoding="utf-8")

    summary = locust_summary(path)

    assert summary == {
        "requests": 5990,
        "failures": 10,
        "achieved_rps": 99.8,
        "avg_ms": 9.5,
        "p50_ms": 8.0,
        "p95_ms": 16.0,
        "p99_ms": 31.0,
        "max_ms": 120.0,
    }


def test_a_run_summary_renders_as_a_table(tmp_path: Path) -> None:
    path = tmp_path / "run_stats.csv"
    path.write_text(LOCUST_CSV, encoding="utf-8")
    run = run_summary(100, 60, BEFORE, AFTER, path)
    report = {
        "label": "baseline",
        "commit": "abc1234",
        "started_at": "2026-11-18T10:00:00+00:00",
        "machine": {"cpu": "Example CPU", "cpus": 4, "memory_gb": 16.0, "os": "Linux"},
        "service": "uvicorn, 1 worker",
        "stack": "Compose",
        "runs": [run],
    }

    text = render_markdown(report)

    assert run["server_score_ms"]["p50"] == pytest.approx(5.0)
    assert run["stages"]["model"]["mean_ms"] == pytest.approx(2.0)
    assert run["decisions"] == {"approve": 90.0, "review": 10.0}
    assert "| 100 | 99.8 | 10 | 4 | 8.00 | 16.00 | 31.00 |" in text
    assert run["degraded"] == 4
    assert "Example CPU" in text
    assert "| model |" in text


def test_the_driver_runs_each_rate_and_writes_both_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fraudlens.tools import loadtest

    scrapes = iter([BEFORE, AFTER] * 3)
    monkeypatch.setattr(
        loadtest.httpx, "get", lambda url, timeout: type("R", (), {"text": next(scrapes)})()
    )
    rates: list[int] = []

    def fake_locust(url: str, rate: int, duration_s: int, prefix: Path, run_id: str) -> None:
        rates.append(rate)
        Path(f"{prefix}_stats.csv").write_text(LOCUST_CSV, encoding="utf-8")

    monkeypatch.setattr(loadtest, "run_locust", fake_locust)

    code = loadtest.main(
        ["--url", "http://svc", "--label", "trial", "--rates", "50", "200", "--out", str(tmp_path)]
    )

    assert code == 0
    assert rates == [50, 50, 200]  # warm-up at the first rate, then each measured rate
    report = (tmp_path / "trial.json").read_text(encoding="utf-8")
    assert '"target_rps": 200' in report
    assert (tmp_path / "trial.md").read_text(encoding="utf-8").startswith("# Load test: trial")
    assert loadtest.machine_spec()["cpus"]


def test_metrics_render_from_the_multiprocess_directory_when_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fraudlens.service.metrics import MULTIPROC_ENV, ServiceMetrics

    metrics = ServiceMetrics()
    metrics.decided("approve", degraded=False, replayed=False, score=0.01)
    assert b"fraudlens_decisions_total" in metrics.render()

    monkeypatch.setenv(MULTIPROC_ENV, str(tmp_path))
    # Nothing has been written to the directory by a worker, so nothing is reported:
    # the scrape reads the workers' files, not this process's registry.
    assert b"fraudlens_decisions_total" not in metrics.render()


def test_failed_requests_are_reported_not_fatal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fraudlens.tools import loadtest

    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> object:
        calls.append(command)
        prefix = command[command.index("--csv") + 1]
        Path(f"{prefix}_stats.csv").write_text(LOCUST_CSV, encoding="utf-8")
        return type("Done", (), {"returncode": 1})()  # Locust exits 1 when any request failed

    monkeypatch.setattr(loadtest.subprocess, "run", fake_run)
    monkeypatch.setattr(
        loadtest,
        "get_settings",
        lambda: type(
            "S", (), {"api_key": type("K", (), {"get_secret_value": lambda self: "k"})()}
        )(),
    )

    loadtest.run_locust("http://svc", 500, 5, tmp_path / "rate500", "run")

    assert len(calls) == 1
    assert "--users" in calls[0]


def test_a_locust_run_that_wrote_nothing_is_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fraudlens.tools import loadtest

    monkeypatch.setattr(
        loadtest.subprocess, "run", lambda command, **kwargs: type("Done", (), {"returncode": 2})()
    )
    monkeypatch.setattr(
        loadtest,
        "get_settings",
        lambda: type(
            "S", (), {"api_key": type("K", (), {"get_secret_value": lambda self: "k"})()}
        )(),
    )

    with pytest.raises(RuntimeError, match="no statistics"):
        loadtest.run_locust("http://svc", 50, 5, tmp_path / "rate50", "run")
