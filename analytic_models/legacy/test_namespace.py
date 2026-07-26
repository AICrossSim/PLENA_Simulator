from __future__ import annotations

import ast
from pathlib import Path


def test_formal_dse_does_not_import_legacy_namespace() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    formal_sources = (
        repo_root / "analytic_models/dse",
        repo_root / "Workspace/qwen3_32b_dense_analytic/run_optuna_dse.py",
    )
    for source in formal_sources:
        paths = source.rglob("*.py") if source.is_dir() else (source,)
        for path in paths:
            tree = ast.parse(path.read_text(), filename=str(path))
            imported = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported.append(node.module)
            assert not any(
                name == "analytic_models.legacy"
                or name.startswith("analytic_models.legacy.")
                for name in imported
            )
