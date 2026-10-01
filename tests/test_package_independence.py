"""Packaging checks catch editable-checkout and former-project dependencies."""

import ast
import importlib
import os
import subprocess
import sys
from pathlib import Path

import mooneural


def test_import_is_inert_and_former_dependencies_are_not_loaded():
    code = "import sys, mooneural; assert 'tensorflow' not in sys.modules; assert 'dsge_hmc' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], env=os.environ.copy(), check=True)
    assert not any(n == "dsge_hmc" or n.startswith("dsge_hmc.") for n in sys.modules)


def test_every_production_module_has_no_former_dependency():
    root = Path(mooneural.__file__).parent
    for path in root.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else (
                [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
            assert not any(n.split(".")[0] in {"dsge_hmc", "common_utils", "tests"} for n in names), str(path)
        name = "mooneural." + ".".join(path.relative_to(root).with_suffix("").parts)
        if name.endswith(".__init__"):
            name = name[:-9]
        importlib.import_module(name)


def test_runtime_source_bindings_work_from_installed_package():
    from mooneural.training import generic_coverage_runner as coverage
    from mooneural.training import generic_finite_objective_runner as finite

    for module in (coverage, finite):
        sources = module._sources()
        assert sources
        coverage._check_sources(sources)


def test_required_wheel_location_when_declared():
    expected = os.environ.get("MOONEURAL_EXPECT_INSTALLED_ROOT")
    if expected:
        assert Path(mooneural.__file__).resolve().is_relative_to(Path(expected).resolve())
