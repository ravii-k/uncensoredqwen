"""Hugging Face GGUF selection behavior using fake API metadata only."""

import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import pytest

module_path = Path(__file__).resolve().parents[1] / "server" / "find_model.py"
module_spec = spec_from_file_location("qwenfree_find_model", module_path)
assert module_spec is not None and module_spec.loader is not None
find_model = module_from_spec(module_spec)
sys.modules[module_spec.name] = find_model
module_spec.loader.exec_module(find_model)

GIB = find_model.GIB
ModelSelectionError = find_model.ModelSelectionError
quant_for_filename = find_model.quant_for_filename
resolve_model = find_model.resolve_model
select_quant = find_model.select_quant


def sibling(name: str, size_gib: float) -> SimpleNamespace:
    return SimpleNamespace(rfilename=name, size=int(size_gib * GIB))


def test_selects_preferred_q4_and_complete_shards() -> None:
    first = "Qwen-Q4_K_M-00001-of-00002.gguf"
    second = "Qwen-Q4_K_M-00002-of-00002.gguf"
    siblings = [
        sibling("Qwen-IQ4_XS.gguf", 12),
        sibling(first, 9),
        sibling(second, 9),
        sibling("Qwen-Q3_K_M.gguf", 8),
    ]

    selection = select_quant("sample/model", siblings)

    assert selection.quant == "Q4_K_M"
    assert selection.files == [first, second]
    assert selection.primary_file == first
    assert selection.total_size_bytes == 18 * GIB


def test_falls_back_to_iq4_then_q3_when_weights_exceed_budget() -> None:
    siblings = [
        sibling("large-Q4_K_M.gguf", 25),
        sibling("small-IQ4_XS.gguf", 17),
        sibling("smaller-Q3_K_M.gguf", 8),
    ]
    assert select_quant("sample/model", siblings).quant == "IQ4_XS"
    assert (
        select_quant(
            "sample/model",
            [siblings[0], siblings[2]],
        ).quant
        == "Q3_K_M"
    )


def test_incomplete_shard_set_is_rejected() -> None:
    with pytest.raises(ModelSelectionError, match="No complete"):
        select_quant(
            "sample/model",
            [sibling("Qwen-Q4_K_M-00001-of-00002.gguf", 9)],
        )


def test_repository_path_traversal_is_rejected() -> None:
    with pytest.raises(ModelSelectionError, match="No complete"):
        select_quant(
            "sample/model",
            [sibling("../escape-Q4_K_M.gguf", 8)],
        )


def test_quant_detection_requires_gguf_extension() -> None:
    assert quant_for_filename("folder/Model-IQ4_XS.gguf") == "IQ4_XS"
    assert quant_for_filename("Model-Q4_K_M.safetensors") is None


def test_missing_repo_lists_candidates_for_explicit_choice() -> None:
    class MissingRepo(Exception):
        pass

    class FakeApi:
        def model_info(self, repo_id, **kwargs):
            if repo_id == "community/Qwen3.8-27B-Uncensored-GGUF":
                return SimpleNamespace(
                    siblings=[SimpleNamespace(rfilename="model-Q4_K_M.gguf")]
                )
            raise MissingRepo(repo_id)

        def list_models(self, **kwargs):
            return [
                SimpleNamespace(
                    id="community/Qwen3.8-27B-Uncensored-GGUF"
                )
            ]

    with pytest.raises(ModelSelectionError, match="MODEL_REPO") as caught:
        resolve_model(
            FakeApi(),
            "missing/default",
            missing_repo_errors=(MissingRepo,),
        )

    assert "community/Qwen3.8-27B-Uncensored-GGUF" in str(caught.value)


def test_resolve_model_requests_live_file_metadata_and_honors_token() -> None:
    class FakeApi:
        def __init__(self) -> None:
            self.call = None

        def model_info(self, repo_id, **kwargs):
            self.call = (repo_id, kwargs)
            return SimpleNamespace(
                siblings=[sibling("verified-Q4_K_M.gguf", 16)]
            )

    api = FakeApi()
    selection = resolve_model(api, "verified/repo", token="hf-placeholder")

    assert selection.repo_id == "verified/repo"
    assert api.call == (
        "verified/repo",
        {"files_metadata": True, "token": "hf-placeholder"},
    )
