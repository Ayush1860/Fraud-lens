"""``make report``: every number in the documents, generated from committed artefacts.

1. If a model bundle is present (``models/`` is gitignored), copy its small JSON documents
   into ``docs/results/release/``, so the numbers can be regenerated from a fresh clone.
   Copying reads the stored results; it never scores anything again.
2. Write ``docs/results/report.md``, ``docs/model_card.md`` and ``docs/validation_report.md``.
3. Replace the block between the report markers in ``README.md``.

A test regenerates all of it and fails if a committed document differs, so no number in
them can be edited by hand.
"""

import argparse
import json
import re
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fraudlens.training.bundle import latest_bundle, manifest_sha256
from fraudlens.training.report import write_report

RESULTS_DIR = Path("docs/results")
PERF_DIR = Path("docs/perf")
RELEASE_DOCS = (
    "metrics.json",
    "training.json",
    "test_metrics.json",
    "thresholds.json",
    "costs.json",
    "threshold_curve.json",
)
START = "<!-- report:start -->"
END = "<!-- report:end -->"


def export_release(bundle_dir: Path, results_dir: Path = RESULTS_DIR) -> Path:
    """Copy a bundle's result documents into ``results_dir/release/``."""
    target = results_dir / "release"
    target.mkdir(parents=True, exist_ok=True)
    for name in RELEASE_DOCS:
        if (bundle_dir / name).exists():
            shutil.copyfile(bundle_dir / name, target / name)
    (target / "bundle.json").write_text(
        json.dumps(
            {"version": bundle_dir.name, "manifest_sha256": manifest_sha256(bundle_dir)}, indent=2
        )
        + "\n",
        encoding="utf-8",
    )
    return target


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


@dataclass(frozen=True)
class Artefacts:
    """Every committed result the documents quote."""

    release: dict[str, Any]
    experiments: dict[str, Any] | None
    explainability: dict[str, Any] | None
    perf: list[dict[str, Any]]

    @classmethod
    def load(cls, results_dir: Path = RESULTS_DIR, perf_dir: Path = PERF_DIR) -> "Artefacts":
        """Read the artefacts; missing ones are ``None`` or empty."""
        release_dir = results_dir / "release"
        release = {
            name.removesuffix(".json"): _load(release_dir / name)
            for name in (*RELEASE_DOCS, "bundle.json")
        }
        perf = sorted(
            (_load(path) for path in perf_dir.glob("*.json")),
            key=lambda report: report["started_at"],
        )
        return cls(
            release={k: v for k, v in release.items() if v is not None},
            experiments=_load(results_dir / "experiments.json"),
            explainability=_load(results_dir / "explainability.json"),
            perf=perf,
        )


def _interval(summary: Mapping[str, Any], digits: int = 3) -> str:
    low, high = summary["ci95"]
    return f"{summary['mean']:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]"


def _pct(value: float, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f}%"


def readme_block(art: Artefacts) -> str:
    """The headline numbers for the README."""
    test = art.release.get("test_metrics")
    if not test:
        return "No release has been evaluated yet. Run `make train evaluate report`."
    ranking = test["ranking"]
    version = art.release["bundle"]["version"]
    lines = [
        f"Release `{version}`, scored once on the public test file: {test['rows']:,} "
        f"transactions over {test['days']} days, {test['frauds']:,} frauds. Intervals are 95% "
        "day-block bootstrap intervals. Simulated data: these numbers show method, not "
        "real-world accuracy.",
        "",
        "| Measure | Test result |",
        "| --- | --- |",
        f"| Average precision | {_interval(ranking['average_precision'])} |",
        "| Frauds caught when 1% of traffic is reviewed | "
        f"{_interval(ranking['recall_at_1pct'])} |",
        f"| Fraud dollars caught at the same budget | "
        f"{_interval(ranking['dollar_recall_at_1pct'])} |",
        f"| Precision of those alerts | {_interval(ranking['precision_at_1pct'])} |",
        f"| Expected cost per 1,000 transactions | ${test['expected_cost_per_1000']:,.2f}, against "
        f"${test['cost_of_approving_everything_per_1000']:,.2f} for approving everything |",
    ]
    if art.explainability:
        additivity, stability = art.explainability["additivity"], art.explainability["stability"]
        lines.append(
            f"| Explanations | contributions add up to the margin within "
            f"{additivity['max_abs_error']:.1e} log-odds on {additivity['rows']:,} rows; top three "
            f"reasons unchanged by a 1% amount change for {_pct(stability['share_unchanged'])} "
            f"of {stability['rows']:,} flagged rows |"
        )
    best = _best_perf(art.perf)
    if best:
        for run in best["runs"]:
            lines.append(
                f"| Latency, {run['target_rps']} requests/s offered | "
                f"{run['achieved_rps']:.0f} served; p50 {run['p50_ms']:.0f} ms, "
                f"p95 {run['p95_ms']:.0f} ms, p99 {run['p99_ms']:.0f} ms |"
            )
        lines += [
            "",
            f"Latency: {best['service']}, `{best['label']}`, on a GitHub Actions runner "
            f"({best['machine']['cpu']}, {best['machine']['cpus']} CPUs) with Postgres, Redis "
            "and the load generator on the same machine. Every configuration measured, before "
            "and after the fix, is compared in [docs/perf](docs/perf/README.md).",
        ]
    return "\n".join(lines)


def _best_perf(reports: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    """The report that served the most requests at its highest offered rate."""
    if not reports:
        return None
    return max(reports, key=lambda r: (max(run["achieved_rps"] for run in r["runs"]), r["label"]))


def _median_cpu(stats: Path, container: str) -> float | None:
    if not stats.exists():
        return None
    values = sorted(
        float(m) for m in re.findall(rf"{re.escape(container)} ([\d.]+)%", stats.read_text())
    )
    return values[len(values) // 2] if values else None


def perf_index(reports: Sequence[Mapping[str, Any]], perf_dir: Path = PERF_DIR) -> str:
    """``docs/perf/README.md``: every load-test report side by side."""
    lines = [
        "# Performance",
        "",
        "Generated by `make report` from the reports in this directory; each was written by "
        "the *Load test* workflow on a GitHub Actions runner, with Locust, the service, "
        "Postgres and Redis on the same machine. Labels end with the commit measured.",
        "",
        "| Report | Service | Machine | Offered | Served | p50 ms | p95 ms | p99 ms | "
        "Degraded | Service CPU (median) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for report in reports:
        cpu = _median_cpu(perf_dir / f"{report['label']}-containers.txt", "fraudlens-api-1")
        for run in report["runs"]:
            lines.append(
                f"| [{report['label']}]({report['label']}.md) | {report['service']} | "
                f"{report['machine']['cpu']} | {run['target_rps']} | {run['achieved_rps']:.0f} | "
                f"{run['p50_ms']:.0f} | {run['p95_ms']:.0f} | {run['p99_ms']:.0f} | "
                f"{run.get('degraded', 'n/a')} | "
                f"{'n/a' if cpu is None else f'{cpu:.0f}%'} |"
            )
    lines += [
        "",
        "## What the profile showed",
        "",
        "- **Before the fix, one worker used far more CPU than one thread can** (the service "
        "CPU column). A uvicorn worker runs Python on one thread, so the extra CPU could only "
        "be XGBoost's OpenMP threads, which "
        "it starts on every core even to score a single row and which keep spinning between "
        "requests. Postgres stayed at a small fraction of a CPU.",
        "- **Adding workers made it worse.** Three workers, each with an OpenMP thread per "
        "core, oversubscribed the four CPUs shared with Postgres, Redis and Locust: the model "
        "and explanation stages slowed sharply (compare the stage tables in each report) and "
        "throughput fell below one worker's.",
        "- **The fix:** score on one thread per process (`FRAUDLENS_MODEL_THREADS`, default 1). "
        "One row per request gains nothing from more. A single worker then stays near one "
        "CPU, and three workers scale instead of competing.",
        "- **Where time goes now.** The database stage, an insert and commit per decision, is "
        "the largest stage at every rate; it grows under load mostly because requests queue "
        "for the event loop and the connection pool, not because Postgres is busy.",
        "- **Degraded responses** at the highest rate are requests whose card lock was not "
        "free within 150 ms: the service scores them from the transaction alone rather than "
        "letting latency grow without bound.",
        "",
        "## Caveats",
        "",
        "- Each run landed on whatever runner GitHub assigned, and the CPU models differ "
        "between runs. Compare CPU shares and throughput ceilings across runs, not "
        "millisecond-level differences.",
        "- The model is a synthetic one with the released model's size (398 trees, depth 6); "
        "timing depends on tree count and depth, not on what the trees learned.",
        "- The load generator shares the machine with the service, so the highest rates "
        "measure the whole box, not the service alone.",
        "",
    ]
    return "\n".join(lines)


def update_readme(text: str, block: str) -> str:
    """Replace the generated block in the README.

    Raises:
        ValueError: if the README has no report markers.
    """
    if START not in text or END not in text:
        raise ValueError("README.md has no report markers")
    head, rest = text.split(START, 1)
    _, tail = rest.split(END, 1)
    return f"{head}{START}\n{block}\n{END}{tail}"


def _fairness_rows(test: Mapping[str, Any]) -> list[str]:
    rows = []
    for kind, title in (("gender", "Gender"), ("age_band", "Age band")):
        for row in test["fairness"][kind]:
            fpr = row["false_positive_rate"]
            rows.append(
                f"| {title} {row['group']} | {row['genuine_rows']:,} | "
                f"{'n/a' if fpr is None else f'{fpr:.3%}'} | "
                f"{'n/a' if row['recall'] is None else f'{row["recall"]:.3f}'} |"
            )
    return rows


def model_card(art: Artefacts) -> str:
    """``docs/model_card.md``."""
    r = art.release
    training, metrics, test = r.get("training"), r.get("metrics"), r.get("test_metrics")
    thresholds, costs = r.get("thresholds"), r.get("costs")
    if not (training and metrics and test and thresholds and costs):
        return "# Model card\n\nNo evaluated release yet. Run `make train evaluate report`.\n"
    early, late, ranking = metrics["valid_early"], metrics["valid_late"], test["ranking"]
    bundle = r["bundle"]
    lines = [
        f"# Model card: FraudLens {bundle['version']}",
        "",
        "Generated by `make report` from `docs/results/`. Every number below comes from a "
        "committed JSON artefact.",
        "",
        "## Intended use",
        "",
        "Scores card transactions as they are authorised and decides approve, review or block. "
        "Flagged transactions carry reason codes for an analyst. It is a portfolio project on "
        "simulated data: it is not validated for real card traffic and must not decide real "
        "payments.",
        "",
        "## Model",
        "",
        f"- Arm {training['arm']}: {training['arm_description']}.",
        f"- {training['best_iteration'] + 1} trees, chosen by early stopping on average "
        "precision; hyperparameters tuned with Optuna over prequential folds.",
        f"- Manifest SHA-256 `{bundle['manifest_sha256']}`; trained at commit "
        f"`{training['git_commit'][:12]}`.",
        "- Calibrated with isotonic regression on the late validation slice.",
        "",
        "## Data",
        "",
        "Sparkov simulated card transactions. Training uses the public training file only, "
        "split by time: "
        f"{training['rows']['train']:,} training rows ({training['frauds']['train']:,} frauds), "
        f"a {training['rows']['gap']:,}-row gap, then "
        f"{training['rows']['valid_early']:,} early and {training['rows']['valid_late']:,} late "
        "validation rows. Names, addresses and card numbers are removed or tokenised at "
        "ingestion; gender is kept only for the fairness check below.",
        "",
        "## Features",
        "",
        "27 features from one definition shared by training and serving: the amount and its "
        "pattern against the card's last 30 days, velocity over 1 hour, 24 hours and 7 days, "
        "distance from home and from the previous purchase, novelty of merchant and category, "
        "time of day, and cardholder context. Every window reads strictly earlier transactions.",
        "",
        "## Performance",
        "",
        "| Measure | Early validation | Test (scored once) |",
        "| --- | --- | --- |",
        f"| Average precision | {early['average_precision']:.4f} | "
        f"{_interval(ranking['average_precision'], 4)} |",
        f"| Recall at a 1% review budget | {early['recall_at_1pct']:.3f} | "
        f"{_interval(ranking['recall_at_1pct'])} |",
        f"| Precision at a 1% review budget | {early['precision_at_1pct']:.3f} | "
        f"{_interval(ranking['precision_at_1pct'])} |",
        f"| Dollar recall at a 1% review budget | {early['dollar_recall_at_1pct']:.3f} | "
        f"{_interval(ranking['dollar_recall_at_1pct'])} |",
        f"| ROC-AUC (reported, not optimised) | {early['roc_auc']:.4f} | "
        f"{ranking['roc_auc']['mean']:.4f} |",
        "",
        f"Test: {test['rows']:,} transactions over {test['days']} days, {test['frauds']:,} frauds.",
        "",
        "## Calibration and thresholds",
        "",
        f"- Brier score on the test file: {test['brier_raw']:.6f} raw, "
        f"{test['brier_calibrated']:.6f} calibrated.",
        f"- Review at a calibrated score of {thresholds['t_review']:.4f} or more, block at "
        f"{thresholds['t_block']:.4f} or more, chosen to minimise expected cost with no more "
        f"than {_pct(costs['review_capacity_share'])} of transactions sent to review.",
        f"- Cost assumptions (`configs/costs.yaml`): ${costs['review_cost']:.2f} per review, "
        f"{_pct(costs['margin_rate'], 0)} of a wrongly blocked amount, plus "
        f"${costs['friction_cost']:.2f} friction per wrong block. Missed fraud costs its amount.",
        f"- Expected cost on the test file: ${test['expected_cost_per_1000']:,.2f} per 1,000 "
        f"transactions, against ${test['cost_of_approving_everything_per_1000']:,.2f} for "
        "approving everything.",
        f"- Decision share on the late validation slice: approve "
        f"{_pct(late['decision_share']['approve'], 2)}, review "
        f"{_pct(late['decision_share']['review'], 2)}, block "
        f"{_pct(late['decision_share']['block'], 2)}.",
        "",
        "## Fairness check",
        "",
        "False-positive rate of any flag among genuine test transactions, and recall among "
        "frauds. The model never sees gender; age enters only as a feature. A gap is a finding "
        "to discuss, not to hide: it reaches the model through correlated features such as "
        "category and amount.",
        "",
        "| Group | Genuine transactions | False-positive rate | Recall |",
        "| --- | --- | --- | --- |",
        *_fairness_rows(test),
        "",
        "## Limits",
        "",
        "- Simulated data. Sparkov's fraud is generated by rules, which is part of why "
        "average precision is so high; real fraud adapts.",
        "- Fraud labels are complete and immediate here. In production they arrive weeks late "
        "and some are never reported.",
        "- Merchant coordinates in Sparkov are noise around the cardholder's home, so the "
        "travel features carry less signal than they would on real data.",
        "- The thresholds depend on cost assumptions; the threshold lab shows how they move.",
        "- Reason codes say why the model scored high, not that fraud happened.",
        "",
    ]
    return "\n".join(lines)


def validation_report(art: Artefacts) -> str:
    """``docs/validation_report.md``, under the three headings validators use."""
    r = art.release
    test, training = r.get("test_metrics"), r.get("training")
    if not (test and training and art.experiments and art.explainability):
        return "# Validation report\n\nArtefacts are missing. Run `make report` after training.\n"
    arms = {arm["arm"]: arm["summary"] for arm in art.experiments["arms"]}
    chosen = arms[training["arm"]]
    baseline = arms.get("A0")
    expl = art.explainability
    ranking = [row["feature"] for row in expl["global_ranking"][:5]]
    lines = [
        f"# Validation report: FraudLens {r['bundle']['version']}",
        "",
        "Generated by `make report`. Organised under the three components of model validation "
        "in US supervisory guidance: conceptual soundness, outcomes analysis and ongoing "
        "monitoring.",
        "",
        "## Conceptual soundness",
        "",
        "- **Time-ordered data.** Splits are by time with a gap, features read strictly earlier "
        "transactions, and any resampling happens inside the training fold. A property test "
        "checks that a later transaction never changes an earlier one's features.",
        "- **One feature definition.** Training and serving call the same function; a parity "
        "test compares the batch builder with it on real transactions, and an integration test "
        "compares features computed online by the running service with the batch builder.",
        f"- **Imbalance handled by evidence.** Seven arms were compared over "
        f"{len(art.experiments['seeds'])} seeds. The released arm {training['arm']} reached "
        f"average precision {_interval(chosen['average_precision'])} on early validation"
        + (
            f", against {_interval(baseline['average_precision'])} for logistic regression."
            if baseline
            else "."
        ),
        f"- **Explanations are exact and checked.** Contributions add up to the margin within "
        f"{expl['additivity']['max_abs_error']:.1e} log-odds on {expl['additivity']['rows']:,} "
        "rows, and agree with a brute-force Shapley calculation written independently. The "
        f"top three reason groups are unchanged by a 1% change in amount for "
        f"{_pct(expl['stability']['share_unchanged'])} of {expl['stability']['rows']:,} "
        f"flagged rows. The largest global contributions: {', '.join(ranking)}.",
        "",
        "## Outcomes analysis",
        "",
        f"The public test file was scored once, by the only function allowed to read it: "
        f"{test['rows']:,} transactions over {test['days']} days.",
        "",
        "| Measure | Test |",
        "| --- | --- |",
        f"| Average precision | {_interval(test['ranking']['average_precision'], 4)} |",
        f"| Recall at a 1% review budget | {_interval(test['ranking']['recall_at_1pct'])} |",
        f"| Precision at a 1% review budget | {_interval(test['ranking']['precision_at_1pct'])} |",
        f"| Expected cost per 1,000 | ${test['expected_cost_per_1000']:,.2f} |",
        "",
        "The full tables, the reliability diagram and the review-queue ordering are in "
        "[results/report.md](results/report.md). A test re-scores 1,000 stored decisions from "
        "their audit rows and reproduces every score exactly.",
        "",
        "## Ongoing monitoring",
        "",
        "- **Feature drift:** population stability index per feature over the newest 10,000 "
        "decisions against the training histograms, refreshed every minute; calendar "
        "features are reported but never alarm.",
        "- **Segment drift:** amount within each category, newest window against the previous "
        "one; this is what catches a shift confined to one category.",
        "- **Decision mix:** live approve, review and block shares against validation.",
        "- **Explanation drift:** monthly mean absolute contribution per feature, for all "
        f"rows and for frauds, over {len({row['month'] for row in expl['monthly_attribution']})} "
        "months (`results/explainability.json`).",
        "- **Outcomes:** realised precision of labelled alerts, on the console's monitoring page.",
        "- Alarms and what to do about them: [runbook.md](runbook.md).",
        "",
        "## Regulatory context",
        "",
        "Stated as the project plan's sources summarise it; this is not legal advice.",
        "",
        "- United States: Federal Reserve SR 26-2 superseded SR 11-7 on 17 April 2026 "
        "([Federal Reserve](https://www.federalreserve.gov/supervisionreg/srletters/sr2602.htm)) "
        "and keeps conceptual soundness, outcomes analysis and ongoing monitoring as the "
        "components of validation.",
        "- India: the RBI's FREE-AI committee report of August 2025 sets seven principles, one "
        'of them "Understandable by Design", and names fraud detection as a major use of AI '
        "([RBI](https://www.rbi.org.in/scriptS/PublicationReportDetails.aspx?ID=1306)).",
        "- European Union: the AI Act lists credit scoring as high-risk but excludes systems "
        "used to detect financial fraud ([Annex III](https://artificialintelligenceact.eu/annex/3/)).",
        "",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for ``python -m fraudlens.tools.report``."""
    parser = argparse.ArgumentParser(prog="python -m fraudlens.tools.report")
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--perf-dir", type=Path, default=PERF_DIR)
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    parser.add_argument("--readme", type=Path, default=Path("README.md"))
    args = parser.parse_args(argv)

    try:
        bundle = latest_bundle(args.models_dir)
    except FileNotFoundError:
        print("no local bundle: using the release documents already in docs/results/release")
    else:
        export_release(bundle, args.results_dir)

    release_dir = args.results_dir / "release"
    write_report(args.results_dir, release_dir if release_dir.exists() else None)
    art = Artefacts.load(args.results_dir, args.perf_dir)
    (args.docs_dir / "model_card.md").write_text(model_card(art), encoding="utf-8")
    (args.docs_dir / "validation_report.md").write_text(validation_report(art), encoding="utf-8")
    if art.perf:
        (args.perf_dir / "README.md").write_text(
            perf_index(art.perf, args.perf_dir), encoding="utf-8"
        )
    args.readme.write_text(
        update_readme(args.readme.read_text(encoding="utf-8"), readme_block(art)), encoding="utf-8"
    )
    print("wrote report.md, model_card.md, validation_report.md and the README block")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
