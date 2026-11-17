"""The training command line must import under a notebook's inline plotting backend.

Jupyter and Colab export ``MPLBACKEND=module://matplotlib_inline.backend_inline`` to every
shell command they run. The project's own environment has no such module, so a plain
``import matplotlib`` there fails before any training starts.
"""

import os
import subprocess
import sys
from pathlib import Path


def test_cli_imports_and_plots_under_an_inline_backend(tmp_path: Path) -> None:
    script = (
        "import json, pathlib, sys\n"
        "from fraudlens.training.cli import main\n"
        "from fraudlens.training import report\n"
        "results = pathlib.Path(sys.argv[1])\n"
        "results.mkdir()\n"
        "curve = {'recall': [0.0, 1.0], 'precision': [1.0, 0.5]}\n"
        "report._pr_figure({'arms': [{'arm': 'A1', 'pr_curve': curve}]},"
        " results / 'pr.png', 't')\n"
        "print('ok')\n"
    )
    # Colab's exact value. Matplotlib treats it as the "inline" backend, which is valid only
    # when matplotlib-inline is installed, and the project environment does not install it.
    env = {**os.environ, "MPLBACKEND": "module://matplotlib_inline.backend_inline"}

    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", script, str(tmp_path / "out")],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "out" / "pr.png").exists()
