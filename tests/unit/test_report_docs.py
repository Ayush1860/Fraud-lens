"""Every number in the README, the model card and the validation report is generated.

The documents are regenerated from the committed artefacts and compared with what is
committed, so a number edited by hand, or a stale document, fails the build.
"""

import json
import shutil
from pathlib import Path

import pytest

from fraudlens.tools import report
from fraudlens.tools.report import (
    END,
    START,
    Artefacts,
    model_card,
    readme_block,
    update_readme,
    validation_report,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def artefacts() -> Artefacts:
    return Artefacts.load(ROOT / "docs/results", ROOT / "docs/perf")


def test_the_readme_block_matches_the_artefacts(artefacts: Artefacts) -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert update_readme(readme, readme_block(artefacts)) == readme, (
        "README numbers differ from the artefacts: run `make report`"
    )


@pytest.mark.parametrize(
    ("document", "render"),
    [("model_card.md", model_card), ("validation_report.md", validation_report)],
)
def test_generated_documents_match_the_artefacts(
    artefacts: Artefacts, document: str, render: object
) -> None:
    committed = (ROOT / "docs" / document).read_text(encoding="utf-8")

    assert render(artefacts) == committed, f"docs/{document} is stale: run `make report`"  # type: ignore[operator]


def test_the_block_replaces_only_what_is_between_the_markers() -> None:
    text = f"before\n{START}\nold\n{END}\nafter\n"

    assert update_readme(text, "new") == f"before\n{START}\nnew\n{END}\nafter\n"
    with pytest.raises(ValueError, match="markers"):
        update_readme("no markers here", "new")


def test_missing_artefacts_give_placeholders_not_numbers() -> None:
    empty = Artefacts(release={}, experiments=None, explainability=None, perf=[])

    assert "No release" in readme_block(empty)
    assert "No evaluated release" in model_card(empty)
    assert "missing" in validation_report(empty)


def test_the_command_exports_a_bundle_and_writes_everything(
    tmp_path: Path, toy_bundle_dir: Path
) -> None:
    models = tmp_path / "models"
    shutil.copytree(toy_bundle_dir, models / toy_bundle_dir.name)
    results = tmp_path / "results"
    shutil.copytree(ROOT / "docs/results", results, ignore=shutil.ignore_patterns("release"))
    docs, perf = tmp_path / "docs", tmp_path / "perf"
    docs.mkdir()
    perf.mkdir()
    readme = tmp_path / "README.md"
    readme.write_text(f"# x\n{START}\n{END}\n", encoding="utf-8")

    code = report.main(
        [
            "--models-dir",
            str(models),
            "--results-dir",
            str(results),
            "--perf-dir",
            str(perf),
            "--docs-dir",
            str(docs),
            "--readme",
            str(readme),
        ]
    )

    assert code == 0
    exported = json.loads((results / "release/bundle.json").read_text(encoding="utf-8"))
    assert exported["version"] == toy_bundle_dir.name
    assert len(exported["manifest_sha256"]) == 64
    assert (docs / "model_card.md").exists()
    assert (docs / "validation_report.md").exists()
    assert f"## Release {toy_bundle_dir.name}" in (results / "report.md").read_text(
        encoding="utf-8"
    )
