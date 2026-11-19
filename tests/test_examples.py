"""Keep the README and the examples honest: they must run against the current API."""

import re
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = sorted((ROOT / "examples").glob("0*.py"))


@pytest.mark.slow
@pytest.mark.parametrize("script", EXAMPLES, ids=lambda p: p.name)
def test_example_scripts_run(script, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runpy.run_path(str(script), run_name="__main__")


def test_readme_python_blocks_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    blocks = re.findall(r"```python\n(.*?)```", text, flags=re.S)
    assert blocks
    env: dict = {}
    for block in blocks:
        exec(compile(block, "README.md", "exec"), env)  # noqa: S102
    assert (tmp_path / "ghz.gif").exists()


@pytest.mark.slow
def test_tour_notebook_executes(tmp_path, monkeypatch):
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    pytest.importorskip("ipykernel")
    nb = nbformat.read(ROOT / "examples" / "tour.ipynb", as_version=4)
    monkeypatch.setenv("MPLBACKEND", "Agg")
    client = nbclient.NotebookClient(
        nb, timeout=300, kernel_name="python3", resources={"metadata": {"path": str(tmp_path)}}
    )
    client.execute()
    assert all(c.get("outputs") is not None for c in nb.cells if c.cell_type == "code")
