"""The serving dependency group must not drift from the project's pins.

`[dependency-groups] serving` lists the subset the scoring API imports, so the
container image can skip torch, mlflow, feast and evidently - over a gigabyte of
packages that never run on the serving path, headed for a 1 GB t3.micro.

That subset repeats version pins from `[project.dependencies]`. Duplicated pins
rot: someone bumps one and the image silently serves a different version of a
library than every test ran against. This makes that a failing test instead.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def _requirements(entries: list[str]) -> dict[str, str]:
    """{package: full spec}, keyed by name with any extras stripped."""
    parsed = {}
    for entry in entries:
        # "psycopg[binary]==3.3.5" -> name "psycopg"
        name = re.split(r"[\[=<>!;]", entry, maxsplit=1)[0].strip().lower()
        parsed[name] = entry.strip()
    return parsed


@pytest.fixture(scope="module")
def config() -> dict:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


def test_serving_group_exists(config):
    assert "serving" in config["dependency-groups"], (
        "The serving group is what keeps torch out of the container image."
    )


def test_every_serving_pin_matches_the_project_pin(config):
    project = _requirements(config["project"]["dependencies"])
    serving = _requirements(config["dependency-groups"]["serving"])

    mismatched = {
        name: (spec, project.get(name))
        for name, spec in serving.items()
        if project.get(name) != spec
    }
    assert not mismatched, (
        "serving pins disagree with [project.dependencies] - the container would "
        f"run versions nothing was tested against: {mismatched}"
    )


def test_serving_excludes_the_heavy_training_packages(config):
    """The whole point. If one of these creeps in, the image bloats past what a
    t3.micro can comfortably build and nobody notices until the deploy."""
    serving = _requirements(config["dependency-groups"]["serving"])
    for package in ("torch", "torch-geometric", "mlflow", "feast", "evidently"):
        assert package not in serving, (
            f"{package} is not imported on the serving path and must not be in the image"
        )


def test_serving_covers_what_the_api_imports(config):
    """A missing runtime dependency fails at container start, not at build -
    i.e. in the deploy, not in CI."""
    serving = _requirements(config["dependency-groups"]["serving"])
    required = {
        "fastapi",
        "uvicorn",
        "pydantic",
        "pydantic-settings",
        "xgboost",
        "shap",
        "pandas",
        "numpy",
        "pyarrow",
        "sqlalchemy",
        "psycopg",
        "neo4j",
        "pyjwt",
        "bcrypt",
        "boto3",
    }
    missing = required - set(serving)
    assert not missing, f"serving group is missing runtime dependencies: {sorted(missing)}"
