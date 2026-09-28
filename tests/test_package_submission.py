"""Unit tests for scripts/package_submission.py file discovery.

find_submission_files() has to accept both naming conventions the harness
produces: `<model>_<dataset>_submission.jsonl` (written by evaluate(), i.e.
`--mode all` / `--mode eval`) and `<model>_<dataset>.submission.jsonl`
(written by inference.py / inference_video.py under `--mode infer`). It
also has to prefer the canonical split of a task over debug/variant splits
(`VANTAGE_2DGrounding_val`, `VANTAGE_VQA_8frame_200`, ...) and, among
equally ranked candidates, the most recently modified file.

scripts/ has no __init__.py, so the module is loaded by path, exactly as
scripts/validate_submission.py does.
"""
import importlib.util
import os
import time
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "package_submission.py"


def _load():
    spec = importlib.util.spec_from_file_location("_pkg_under_test", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pkg = _load()


def _touch(path: Path, mtime: float = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"id": "x"}\n')
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


# ---------------------------------------------------------------------------
# Both naming conventions are discovered
# ---------------------------------------------------------------------------

def test_eval_mode_underscore_suffix_is_found(tmp_path):
    f = _touch(tmp_path / "M_VANTAGE_VQA_8frame_submission.jsonl")
    found = pkg.find_submission_files(tmp_path)
    assert found == {"vqa": f}


def test_infer_mode_dot_suffix_is_found(tmp_path):
    f = _touch(tmp_path / "M_VANTAGE_VQA_8frame.submission.jsonl")
    found = pkg.find_submission_files(tmp_path)
    assert found == {"vqa": f}


def test_all_eight_tasks_map_to_canonical_keys(tmp_path):
    names = {
        "vqa": "VANTAGE_VQA_8frame",
        "event_verification": "VANTAGE_EventVerification_8frame",
        "temporal": "VANTAGE_Temporal_8frame",
        "dvc": "VANTAGE_DVC_8frame",
        "sot": "VANTAGE_SOT",
        "grounding": "VANTAGE_2DGrounding",
        "pointing": "VANTAGE_2DPointing",
        "astro": "Astro2D",
    }
    expected = {}
    for i, (task, ds) in enumerate(names.items()):
        suffix = "_submission.jsonl" if i % 2 == 0 else ".submission.jsonl"
        expected[task] = _touch(tmp_path / f"Qwen3-VL-2B-Instruct_{ds}{suffix}")
    assert pkg.find_submission_files(tmp_path) == expected


def test_model_names_with_underscores_do_not_confuse_matching(tmp_path):
    f = _touch(tmp_path / "my_model_v2_VANTAGE_2DPointing_submission.jsonl")
    assert pkg.find_submission_files(tmp_path) == {"pointing": f}


def test_unrelated_jsonl_is_ignored(tmp_path):
    _touch(tmp_path / "notes.jsonl")
    _touch(tmp_path / "M_SomethingElse_submission.jsonl")
    assert pkg.find_submission_files(tmp_path) == {}


def test_search_is_recursive(tmp_path):
    f = _touch(tmp_path / "T20260928_Gabcdef12" / "M_Astro2D_submission.jsonl")
    assert pkg.find_submission_files(tmp_path) == {"astro": f}


# ---------------------------------------------------------------------------
# Canonical split wins over variant splits
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("canonical,variant,task", [
    ("M_VANTAGE_2DGrounding_submission.jsonl", "M_VANTAGE_2DGrounding_val_submission.jsonl", "grounding"),
    ("M_VANTAGE_2DGrounding_submission.jsonl", "M_VANTAGE_2DGrounding_small_submission.jsonl", "grounding"),
    ("M_VANTAGE_VQA_8frame_submission.jsonl", "M_VANTAGE_VQA_8frame_200_submission.jsonl", "vqa"),
])
def test_canonical_split_preferred_even_if_variant_is_newer(tmp_path, canonical, variant, task, capsys):
    now = time.time()
    c = _touch(tmp_path / canonical, mtime=now - 3600)
    _touch(tmp_path / variant, mtime=now)  # newer, but a variant split
    found = pkg.find_submission_files(tmp_path)
    assert found[task] == c
    out = capsys.readouterr().out
    assert "multiple submission files" in out
    assert str(c) in out


def test_variant_split_is_used_when_it_is_the_only_match(tmp_path, capsys):
    v = _touch(tmp_path / "M_VANTAGE_2DGrounding_val_submission.jsonl")
    found = pkg.find_submission_files(tmp_path)
    assert found == {"grounding": v}
    out = capsys.readouterr().out
    # The user is told which (non-canonical) file was picked.
    assert "VANTAGE_2DGrounding_val" in out
    assert "variant" in out.lower()


def test_frame_and_fps_suffixes_count_as_canonical():
    ranked = {t: pkg.classify_dataset_component(f"M_{ds}", t) for t, ds in [
        ("vqa", "VANTAGE_VQA_16frame"), ("temporal", "VANTAGE_Temporal_0.5fps"),
        ("sot", "VANTAGE_SOT_16f"), ("dvc", "VANTAGE_DVC_1fps"),
    ]}
    assert all(r == "canonical" for r in ranked.values()), ranked
    assert pkg.classify_dataset_component("M_VANTAGE_VQA_8frame_200", "vqa") == "variant"
    assert pkg.classify_dataset_component("M_VANTAGE_2DGrounding_val", "grounding") == "variant"
    assert pkg.classify_dataset_component("M_Astro2D", "astro") == "canonical"
    assert pkg.classify_dataset_component("M_Other", "astro") is None


# ---------------------------------------------------------------------------
# Ties are broken by modification time, not by sort order
# ---------------------------------------------------------------------------

def test_most_recent_wins_among_equal_rank(tmp_path, capsys):
    now = time.time()
    # Sort order would pick the "zzz" run last; mtime says "aaa" is newer.
    newer = _touch(tmp_path / "aaa_run" / "M_VANTAGE_VQA_8frame_submission.jsonl", mtime=now)
    _touch(tmp_path / "zzz_run" / "M_VANTAGE_VQA_8frame_submission.jsonl", mtime=now - 7200)
    found = pkg.find_submission_files(tmp_path)
    assert found["vqa"] == newer
    assert "Using:" in capsys.readouterr().out


def test_eval_file_preferred_over_older_infer_file_by_mtime(tmp_path):
    now = time.time()
    _touch(tmp_path / "M_VANTAGE_VQA_8frame.submission.jsonl", mtime=now - 60)
    newer = _touch(tmp_path / "M_VANTAGE_VQA_8frame_submission.jsonl", mtime=now)
    assert pkg.find_submission_files(tmp_path)["vqa"] == newer


def test_symlink_and_target_are_deduplicated(tmp_path):
    target = _touch(tmp_path / "run" / "M_VANTAGE_VQA_8frame_submission.jsonl")
    link = tmp_path / "M_VANTAGE_VQA_8frame_submission.jsonl"
    link.symlink_to(target)
    found = pkg.find_submission_files(tmp_path)
    assert found["vqa"].resolve() == target.resolve()
