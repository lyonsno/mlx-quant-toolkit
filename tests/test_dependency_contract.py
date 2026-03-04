import re
import tomllib
import unittest
from pathlib import Path


def _normalize_dep_name(spec: str) -> str:
    # Handle version pins, environment markers, and optional extras notation.
    core = spec.split(";", 1)[0].strip()
    core = core.split("[", 1)[0].strip()
    return re.split(r"[<>=!~ ]", core, maxsplit=1)[0].strip()


class DependencyContractTests(unittest.TestCase):
    def setUp(self):
        self.repo_root = Path(__file__).resolve().parents[1]
        self.pyproject_path = self.repo_root / "pyproject.toml"

    def _load_pyproject_project_table(self):
        data = tomllib.loads(self.pyproject_path.read_text())
        project = data.get("project", {})
        self.assertIsInstance(project, dict)
        return project

    def test_runtime_optional_components_are_not_required_dependencies(self):
        project = self._load_pyproject_project_table()
        required = project.get("dependencies", [])
        dep_names = {_normalize_dep_name(spec) for spec in required}

        # Contract: pipeline behavior degrades gracefully without these packages.
        # They should remain install-optional rather than hard required.
        self.assertNotIn("mlx", dep_names)
        self.assertNotIn("mlx-lm", dep_names)
        self.assertNotIn("pyarrow", dep_names)

    def test_optional_dependency_groups_cover_runtime_optional_components(self):
        project = self._load_pyproject_project_table()
        optional = project.get("optional-dependencies", {})
        self.assertIsInstance(optional, dict)

        self.assertIn("mlx", optional)
        self.assertIn("parquet", optional)

        mlx_deps = {_normalize_dep_name(spec) for spec in optional["mlx"]}
        parquet_deps = {_normalize_dep_name(spec) for spec in optional["parquet"]}

        self.assertIn("mlx", mlx_deps)
        self.assertIn("mlx-lm", mlx_deps)
        self.assertIn("pyarrow", parquet_deps)

    def test_all_extra_is_exact_union_of_capability_extras(self):
        project = self._load_pyproject_project_table()
        optional = project.get("optional-dependencies", {})
        self.assertIsInstance(optional, dict)

        # Guardrail: docs recommend installing with --extra all.
        # Keep "all" synchronized with capability extras users rely on.
        self.assertIn("all", optional)
        self.assertIn("mlx", optional)
        self.assertIn("parquet", optional)

        all_deps = {_normalize_dep_name(spec) for spec in optional["all"]}
        mlx_deps = {_normalize_dep_name(spec) for spec in optional["mlx"]}
        parquet_deps = {_normalize_dep_name(spec) for spec in optional["parquet"]}

        self.assertEqual(all_deps, mlx_deps | parquet_deps)


if __name__ == "__main__":
    unittest.main()
