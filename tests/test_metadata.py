import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


def _load_metadata():
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts" / "metadata.py"
    spec = importlib.util.spec_from_file_location("metadata", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load metadata module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MetadataModuleTests(unittest.TestCase):
    def setUp(self):
        self.repo_root = Path(__file__).resolve().parents[1]
        self.metadata = _load_metadata()

    def test_find_config_json_prefers_root_and_file_parent(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            model_dir = tmp_path / "model"
            model_dir.mkdir(parents=True, exist_ok=True)

            root_cfg = model_dir / "config.json"
            root_cfg.write_text(json.dumps({"hidden_size": 4}))

            nested_dir = model_dir / "model"
            nested_dir.mkdir(parents=True, exist_ok=True)
            nested_cfg = nested_dir / "config.json"
            nested_cfg.write_text(json.dumps({"hidden_size": 8}))

            found = self.metadata.find_config_json(model_dir)
            self.assertEqual(found, root_cfg)

            weight_file = model_dir / "weights.safetensors"
            weight_file.write_text("stub")
            found_from_file = self.metadata.find_config_json(weight_file)
            self.assertEqual(found_from_file, root_cfg)

    def test_find_config_json_falls_back_to_common_subpath(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            model_dir = tmp_path / "model"
            model_dir.mkdir(parents=True, exist_ok=True)

            nested_dir = model_dir / "model"
            nested_dir.mkdir(parents=True, exist_ok=True)
            nested_cfg = nested_dir / "config.json"
            nested_cfg.write_text(json.dumps({"hidden_size": 8}))

            found = self.metadata.find_config_json(model_dir)
            self.assertEqual(found, nested_cfg)

    def test_parse_config_json_invalid_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            bad_path = Path(tmp_dir) / "config.json"
            bad_path.write_text("{not-json")
            parsed = self.metadata.parse_config_json(bad_path)
            self.assertEqual(parsed, {})

    def test_shape_budget_aliases_and_coercion(self):
        cfg = {
            "n_layer": "24",
            "d_model": "4096",
            "n_routed_experts": "128",
            "expert_intermediate_size": "1408",
            "shared_expert_intermediate_size": 512,
            "experts_per_token": "4",
            "n_shared_experts": "1",
            "decoder_sparse_step": 1,
            "full_attention_interval": 4,
            "first_k_dense_replace": 2,
            "mlp_only_layers": [0, 1],
            "num_experts": "oops",
        }

        budget = self.metadata.ModelShapeBudget.from_config_dict(cfg)

        self.assertEqual(budget.num_hidden_layers, 24)
        self.assertEqual(budget.hidden_size, 4096)
        self.assertEqual(budget.num_experts, 128)
        self.assertEqual(budget.moe_intermediate_size, 1408)
        self.assertEqual(budget.shared_expert_intermediate_size, 512)
        self.assertEqual(budget.num_experts_per_tok, 4)
        self.assertEqual(budget.n_shared_experts, 1)
        self.assertEqual(budget.decoder_sparse_step, 1)
        self.assertEqual(budget.full_attention_interval, 4)
        self.assertEqual(budget.first_k_dense_replace, 2)
        self.assertEqual(budget.mlp_only_layers, [0, 1])

    def test_shape_budget_parses_example_configs(self):
        examples = {
            "kikekewl/qwen3-next-80b-a3b-thinking/config.json": {
                "hidden_size": 2048,
                "num_hidden_layers": 48,
                "num_experts": 512,
                "moe_intermediate_size": 512,
            },
            "lmstudio-community/GLM-4.5-Air-mxfp4/config.json": {
                "hidden_size": 4096,
                "num_hidden_layers": 46,
                "num_experts": 128,
                "num_experts_per_tok": 8,
            },
            "lmstudio-community/gpt-oss-20b-MXFP4-Q8/config.json": {
                "hidden_size": 2880,
                "num_hidden_layers": 24,
                "num_experts": 32,
                "num_experts_per_tok": 4,
            },
        }

        base = self.repo_root / "example_safetensors_folder_metadata_convention_variance"
        for rel_path, expected in examples.items():
            cfg_path = base / rel_path
            parsed = self.metadata.parse_config_json(cfg_path)
            budget = self.metadata.ModelShapeBudget.from_config_dict(parsed)
            for key, value in expected.items():
                self.assertEqual(getattr(budget, key), value)

    def test_parse_config_json_empty_file_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            p = Path(tmp_dir) / "empty.json"
            p.write_text("")
            parsed = self.metadata.parse_config_json(p)
            self.assertEqual(parsed, {})

    def test_parse_config_json_non_dict_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            p = Path(tmp_dir) / "config.json"
            p.write_text(json.dumps([1, 2, 3]))
            parsed = self.metadata.parse_config_json(p)
            self.assertEqual(parsed, {})

    def test_find_safetensors_index_json_prefers_root(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            model_dir = tmp_path / "model"
            model_dir.mkdir(parents=True, exist_ok=True)

            root_index = model_dir / "model.safetensors.index.json"
            root_index.write_text(json.dumps({"weight_map": {}}))

            fallback = model_dir / "weights.safetensors.index.json"
            fallback.write_text(json.dumps({"weight_map": {}}))

            found = self.metadata.find_safetensors_index_json(model_dir)
            self.assertEqual(found, root_index)

            weight_file = model_dir / "weights.safetensors"
            weight_file.write_text("stub")
            found_from_file = self.metadata.find_safetensors_index_json(weight_file)
            self.assertEqual(found_from_file, root_index)

    def test_find_safetensors_index_json_falls_back_without_deep_recursion(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            model_dir = tmp_path / "model"
            model_dir.mkdir(parents=True, exist_ok=True)

            fallback = model_dir / "weights.safetensors.index.json"
            fallback.write_text(json.dumps({"weight_map": {}}))

            nested_dir = model_dir / "nested"
            nested_dir.mkdir(parents=True, exist_ok=True)
            nested = nested_dir / "nested.safetensors.index.json"
            nested.write_text(json.dumps({"weight_map": {}}))

            found = self.metadata.find_safetensors_index_json(model_dir)
            self.assertEqual(found, fallback)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            model_dir = tmp_path / "model"
            model_dir.mkdir(parents=True, exist_ok=True)

            nested_dir = model_dir / "nested"
            nested_dir.mkdir(parents=True, exist_ok=True)
            nested = nested_dir / "nested.safetensors.index.json"
            nested.write_text(json.dumps({"weight_map": {}}))

            found = self.metadata.find_safetensors_index_json(model_dir)
            self.assertIsNone(found)

    def test_parse_safetensors_index_returns_weight_map_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            index_path = tmp_path / "model.safetensors.index.json"
            payload = {
                "metadata": {"format": "pt", "total_size": 123},
                "weight_map": {
                    "layers.0.experts.0.down_proj.weight": "shard1.safetensors",
                    "layers.0.experts.1.down_proj.weight": "shard2.safetensors",
                },
            }
            index_path.write_text(json.dumps(payload, indent=2))

            weight_map, metadata = self.metadata.parse_safetensors_index(index_path)
            self.assertEqual(weight_map, payload["weight_map"])
            self.assertEqual(metadata, payload["metadata"])
