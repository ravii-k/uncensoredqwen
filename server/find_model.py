#!/usr/bin/env python3
"""Select a verified Qwen3.8-27B uncensored GGUF from Hugging Face."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import PurePosixPath
from typing import Any

DEFAULT_REPO = "JonathanColetti/Qwen3.8-27B-Uncensored-GGUF"
DEFAULT_MAX_MODEL_GIB = 24.0
QUANT_ORDER = ("Q4_K_M", "IQ4_XS", "Q3_K_M")
GIB = 1024**3


class ModelSelectionError(RuntimeError):
    """Raised when no verified GGUF fits the configured memory budget."""


@dataclass(frozen=True, slots=True)
class ModelSelection:
    """The exact files and model metadata needed by the bootstrap script."""

    repo_id: str
    quant: str
    files: list[str]
    primary_file: str
    total_size_bytes: int
    max_model_gib: float


def _get(item: Any, key: str, default: Any = None) -> Any:
    """Read a field from either HF metadata objects or test dictionaries."""
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def quant_for_filename(filename: str) -> str | None:
    """Return a supported quant name when present in a GGUF filename."""
    if not filename.lower().endswith(".gguf"):
        return None
    lowered = filename.lower()
    patterns = {
        "Q4_K_M": r"(?<![a-z0-9])q4[_-]?k[_-]?m(?![a-z0-9])",
        "IQ4_XS": r"(?<![a-z0-9])iq4[_-]?xs(?![a-z0-9])",
        "Q3_K_M": r"(?<![a-z0-9])q3[_-]?k[_-]?m(?![a-z0-9])",
    }
    for quant, pattern in patterns.items():
        if re.search(pattern, lowered):
            return quant
    return None


def _safe_repo_filename(filename: str) -> bool:
    """Reject absolute paths and traversal in untrusted Hub file metadata."""
    path = PurePosixPath(filename)
    return bool(filename) and not path.is_absolute() and "\\" not in filename and all(
        part not in {"", ".", ".."} for part in path.parts
    )


def _groups_for_quant(siblings: list[Any], quant: str) -> list[tuple[list[str], int]]:
    """Group a quant's files into single GGUFs or complete shard sets."""
    groups: dict[str, dict[str, Any]] = {}
    for sibling in siblings:
        filename = str(_get(sibling, "rfilename", ""))
        size = _get(sibling, "size")
        if (
            not _safe_repo_filename(filename)
            or quant_for_filename(filename) != quant
            or not isinstance(size, int)
            or size <= 0
        ):
            continue
        split = re.match(
            r"^(?P<stem>.*)-(?P<index>\d{5})-of-(?P<count>\d{5})\.gguf$",
            filename,
            flags=re.IGNORECASE,
        )
        if split:
            stem = split.group("stem").lower()
            group = groups.setdefault(
                stem,
                {"files": [], "count": int(split.group("count")), "indices": []},
            )
            if group["count"] != int(split.group("count")):
                continue
            group["files"].append((int(split.group("index")), filename, size))
            group["indices"].append(int(split.group("index")))
        else:
            groups[filename.lower()] = {
                "files": [(1, filename, size)],
                "count": 1,
                "indices": [1],
            }

    valid: list[tuple[list[str], int]] = []
    for group in groups.values():
        expected = group["count"]
        files_with_size = group["files"]
        indices = group["indices"]
        if len(files_with_size) != expected or sorted(indices) != list(
            range(1, expected + 1)
        ):
            continue
        files_with_size.sort(key=lambda item: item[0])
        filenames = [item[1] for item in files_with_size]
        size = sum(item[2] for item in files_with_size)
        valid.append((filenames, size))
    return valid


def select_quant(
    repo_id: str,
    siblings: list[Any],
    *,
    max_model_gib: float = DEFAULT_MAX_MODEL_GIB,
) -> ModelSelection:
    """Pick the preferred complete quant that fits under the weight budget."""
    if max_model_gib <= 0:
        raise ValueError("max_model_gib must be positive.")
    max_bytes = int(max_model_gib * GIB)
    for quant in QUANT_ORDER:
        candidates = [
            group
            for group in _groups_for_quant(siblings, quant)
            if group[1] <= max_bytes
        ]
        if candidates:
            candidates.sort(
                key=lambda group: (
                    any("nomtp" in name.lower() for name in group[0]),
                    group[1],
                    group[0][0].lower(),
                )
            )
            filenames, total_size = candidates[0]
            return ModelSelection(
                repo_id=repo_id,
                quant=quant,
                files=filenames,
                primary_file=filenames[0],
                total_size_bytes=total_size,
                max_model_gib=max_model_gib,
            )
    raise ModelSelectionError(
        "No complete Q4_K_M, IQ4_XS, or Q3_K_M GGUF fits the "
        f"{max_model_gib:g} GiB weight budget in {repo_id}."
    )


def _has_gguf_files(api: Any, repo_id: str, token: str | None) -> bool:
    """Check a search result for GGUF files without downloading model data."""
    try:
        info = api.model_info(repo_id, files_metadata=False, token=token)
    except Exception:
        return False
    return any(
        str(_get(item, "rfilename", "")).lower().endswith(".gguf")
        for item in (_get(info, "siblings", []) or [])
    )


def discover_candidate_repos(api: Any, token: str | None = None) -> list[str]:
    """Find other similarly named uncensored GGUF repositories for user choice."""
    try:
        models = api.list_models(search="Qwen3.8-27B", limit=100, token=token)
    except TypeError:
        models = api.list_models(search="Qwen3.8-27B")
    except Exception:
        return []

    candidates: list[str] = []
    cues = ("uncensored", "unleashed", "abliterat", "heretic")
    for model in models:
        repo_id = str(_get(model, "id", "") or _get(model, "modelId", ""))
        lowered = repo_id.lower()
        if not repo_id or not any(cue in lowered for cue in cues):
            continue
        if _has_gguf_files(api, repo_id, token):
            candidates.append(repo_id)
        if len(candidates) >= 12:
            break
    return sorted(set(candidates), key=str.lower)


def resolve_model(
    api: Any,
    repo_id: str,
    *,
    token: str | None = None,
    max_model_gib: float = DEFAULT_MAX_MODEL_GIB,
    missing_repo_errors: tuple[type[BaseException], ...] = (),
) -> ModelSelection:
    """Verify a repository, or show search results if it is missing."""
    try:
        info = api.model_info(repo_id, files_metadata=True, token=token)
    except Exception as exc:
        if not missing_repo_errors or not isinstance(exc, missing_repo_errors):
            raise
        candidates = discover_candidate_repos(api, token)
        suffix = "\n".join(f"  {candidate}" for candidate in candidates)
        guidance = (
            "Set MODEL_REPO in the Kaggle notebook to one of these repositories, "
            "then rerun the server cell."
        )
        if suffix:
            guidance = f"{guidance}\nCandidates:\n{suffix}"
        raise ModelSelectionError(
            f"Requested Hugging Face repository was not found: {repo_id}. {guidance}"
        ) from exc

    siblings = list(_get(info, "siblings", []) or [])
    if not siblings:
        raise ModelSelectionError(f"Repository {repo_id} contains no files.")
    return select_quant(repo_id, siblings, max_model_gib=max_model_gib)


def main(argv: list[str] | None = None) -> int:
    """Resolve a repository through the Hugging Face API and print its choice."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=os.environ.get("MODEL_REPO", DEFAULT_REPO))
    parser.add_argument("--max-model-gib", type=float, default=DEFAULT_MAX_MODEL_GIB)
    parser.add_argument(
        "--json", action="store_true", help="Print machine-readable JSON."
    )
    args = parser.parse_args(argv)

    try:
        from huggingface_hub import HfApi
        from huggingface_hub.utils import RepositoryNotFoundError
    except ImportError:
        print(
            "huggingface_hub is required in the Kaggle server environment.",
            file=sys.stderr,
        )
        return 2

    token = os.environ.get("HF_TOKEN") or None
    try:
        selection = resolve_model(
            HfApi(),
            args.repo,
            token=token,
            max_model_gib=args.max_model_gib,
            missing_repo_errors=(RepositoryNotFoundError,),
        )
    except ModelSelectionError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except Exception as exc:
        print(
            f"Hugging Face lookup failed ({type(exc).__name__}); check Kaggle Internet "
            "access and repository permissions.",
            file=sys.stderr,
        )
        return 1

    result = asdict(selection)
    if args.json:
        print(json.dumps(result))
    else:
        print(f"Repository: {selection.repo_id}")
        print(f"Quant: {selection.quant}")
        print(f"Total model weight size: {selection.total_size_bytes / GIB:.2f} GiB")
        print(f"Primary GGUF: {selection.primary_file}")
        print("Files to download:")
        for filename in selection.files:
            print(f"  {filename}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
