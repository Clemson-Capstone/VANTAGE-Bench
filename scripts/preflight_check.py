#!/usr/bin/env python3
"""
preflight_check.py — Environment sanity check for someone about to run
VANTAGE-Bench inference (``run_lmudata.py`` and/or ``run.py``).

Emits a machine-readable environment report. Every finding is classified
``ok`` / ``warning`` / ``blocker``:

  - A missing GPU is a ``warning`` — API-only backends (GPT4V, CosmosReason2
    API, ...) remain viable without one.
  - Missing ffmpeg is a ``blocker`` only when the target work explicitly
    needs it (the SOT task's frame extraction, or the CosmosReason2 API
    backend). Otherwise it's a ``warning``. Pass ``--tasks`` and/or
    ``--model`` so this script can tell.
  - A missing/expired HF token is a ``warning`` — only gated data (SOT's
    source videos) and private models need one.
  - ``vlmeval`` failing to import from this repo is a ``blocker`` — nothing
    else works if that's broken.

Usage:
    python scripts/preflight_check.py
    python scripts/preflight_check.py --tasks sot --lmu-root ~/LMUData
    python scripts/preflight_check.py --model cosmos_reason2_api --json

Exit code: 0 if no blocker finding, 1 if any blocker.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

# Make `import vlmeval` work even when this script is invoked as
# `python scripts/preflight_check.py` without an editable install — but do
# this lazily inside check_vlmeval() so a broken vlmeval install doesn't
# crash the rest of the report.
_REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_WORK_DIR = Path("./outputs")
DEFAULT_LMU_ROOT = Path("~/LMUData").expanduser()

# Same conda-env search locations run_lmudata.py's _discover_ffmpeg_dir()
# uses, so this check reports the same answer the prep script would act on.
FFMPEG_ENV_GLOBS = [
    "~/miniconda3/envs/*/bin/ffmpeg",
    "~/anaconda3/envs/*/bin/ffmpeg",
    "/opt/conda/envs/*/bin/ffmpeg",
    "~/miniconda3/bin/ffmpeg",
    "~/anaconda3/bin/ffmpeg",
    "/opt/conda/bin/ffmpeg",
]

MIN_PYTHON = (3, 10)  # matches setup.py python_requires

STATUS_ORDER = {"ok": 0, "warning": 1, "blocker": 2}


class Report:
    def __init__(self):
        self.findings = []  # list[{"check", "status", "message"}]

    def add(self, check: str, status: str, message: str) -> None:
        assert status in STATUS_ORDER
        self.findings.append({"check": check, "status": status, "message": message})

    def worst_status(self) -> str:
        if not self.findings:
            return "ok"
        return max((f["status"] for f in self.findings), key=lambda s: STATUS_ORDER[s])


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_platform(report: Report) -> None:
    py_ver = tuple(sys.version_info[:2])
    conda_env = os.environ.get("CONDA_DEFAULT_ENV")
    msg = (f"platform={sys.platform} python={platform.python_version()} "
           f"conda_env={conda_env or '(none)'}")
    if py_ver < MIN_PYTHON:
        report.add("platform", "warning",
                    msg + f" — python {'.'.join(map(str, MIN_PYTHON))}+ is required by setup.py")
    else:
        report.add("platform", "ok", msg)


def _crash_note(returncode: int) -> str:
    """Describe a nonzero subprocess exit code as a native crash or a normal error.

    POSIX reports a signal-killed child as a negative returncode (-N for signal N).
    Windows instead reports NT exception codes (segfault, stack overflow, ...) as
    large positive values >= 0x80000000 (e.g. 3221225477 == 0xC0000005,
    STATUS_ACCESS_VIOLATION) — never negative. Checking only "negative" therefore
    silently misses every real crash on Windows; check both forms explicitly.
    """
    if returncode < 0:
        return f" (killed by signal {-returncode} — a native crash)"
    if returncode >= 0x80000000:
        return f" (0x{returncode:08X} — looks like a native crash, not a normal exit)"
    return ""


_TORCH_PROBE_CODE = (
    "import json\n"
    "try:\n"
    "    import torch\n"
    "except Exception as e:\n"
    "    print(json.dumps({'importable': False, 'error': f'{type(e).__name__}: {e}'}))\n"
    "else:\n"
    "    facts = {'importable': True, 'version': torch.__version__, 'cuda_available': False, 'gpus': []}\n"
    "    try:\n"
    "        facts['cuda_available'] = torch.cuda.is_available()\n"
    "        if facts['cuda_available']:\n"
    "            for i in range(torch.cuda.device_count()):\n"
    "                p = torch.cuda.get_device_properties(i)\n"
    "                facts['gpus'].append({'index': i, 'name': p.name,\n"
    "                                       'vram_gb': round(p.total_memory / (1024 ** 3), 1)})\n"
    "    except Exception as e:\n"
    "        facts['cuda_query_error'] = f'{type(e).__name__}: {e}'\n"
    "    print(json.dumps(facts))\n"
)


def check_torch(report: Report) -> Optional[dict]:
    """Returns a dict of torch/cuda facts (or None if torch isn't usable).

    Runs in a *subprocess*, same reasoning as check_vlmeval(): torch pulls
    in the same native numpy stack that segfaulted this very tool during
    development on one host, so an in-process import risks taking the whole
    preflight run down instead of producing a warning finding.
    """
    try:
        proc = subprocess.run([sys.executable, "-c", _TORCH_PROBE_CODE],
                               capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        report.add("torch", "warning", "torch import probe timed out after 60s.")
        return None
    except OSError as e:
        report.add("torch", "warning", f"failed to launch torch probe subprocess: {e}")
        return None

    if proc.returncode != 0:
        tail_source = proc.stderr or proc.stdout
        tail = " | ".join(tail_source.strip().splitlines()[-5:]) if tail_source.strip() else ""
        crash_note = _crash_note(proc.returncode)
        report.add("torch", "warning",
                    f"torch probe subprocess exited {proc.returncode}{crash_note}: {tail}. "
                    f"API-only backends remain viable regardless.")
        return None

    try:
        facts = json.loads(proc.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        report.add("torch", "warning", f"torch probe produced unexpected output: {proc.stdout!r}")
        return None

    if not facts.get("importable"):
        report.add("torch", "warning",
                    f"torch not importable ({facts.get('error')}). Required for local model "
                    f"inference; API-only backends (GPT4V, CosmosReason2 API, ...) don't need it.")
        return None

    if "cuda_query_error" in facts:
        report.add("torch", "warning",
                    f"torch {facts['version']} installed but CUDA query failed: {facts['cuda_query_error']}")
    elif facts["cuda_available"]:
        names = ", ".join(f"{g['name']} ({g['vram_gb']} GB)" for g in facts["gpus"])
        report.add("torch", "ok",
                    f"torch {facts['version']}, CUDA available, {len(facts['gpus'])} GPU(s): {names}")
    else:
        report.add("torch", "warning",
                    f"torch {facts['version']} installed but CUDA is not available. Local GPU "
                    f"inference will be unavailable/CPU-only; API backends remain viable.")
    return facts


def check_nvidia_smi(report: Report, torch_facts: Optional[dict]) -> None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
    except FileNotFoundError:
        report.add("nvidia-smi", "warning",
                    "nvidia-smi not found on PATH — cannot cross-check GPU inventory "
                    "(fine on non-NVIDIA hosts / API-only environments).")
        return
    except (subprocess.TimeoutExpired, OSError) as e:
        report.add("nvidia-smi", "warning", f"nvidia-smi invocation failed: {e}")
        return

    if out.returncode != 0:
        report.add("nvidia-smi", "warning",
                    f"nvidia-smi exited {out.returncode}: {out.stderr.strip()[:200]}")
        return

    lines = [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
    smi_count = len(lines)
    torch_count = len(torch_facts["gpus"]) if torch_facts else 0
    detail = "; ".join(lines) if lines else "(no GPUs reported)"
    if torch_facts is not None and torch_facts["cuda_available"] and smi_count != torch_count:
        report.add("nvidia-smi", "warning",
                    f"nvidia-smi reports {smi_count} GPU(s) but torch.cuda sees {torch_count} "
                    f"(CUDA_VISIBLE_DEVICES may be restricting torch's view). nvidia-smi: {detail}")
    else:
        report.add("nvidia-smi", "ok", f"{smi_count} GPU(s) reported: {detail}")


def _nearest_existing(path: Path) -> Path:
    p = path
    while not p.exists() and p.parent != p:
        p = p.parent
    return p


def check_disk(report: Report, check: str, path: Path, warn_below_gb: float) -> None:
    probe = _nearest_existing(path)
    try:
        usage = shutil.disk_usage(probe)
    except OSError as e:
        report.add(check, "warning", f"could not stat disk usage at {probe}: {e}")
        return
    free_gb = usage.free / (1024 ** 3)
    msg = f"{free_gb:.1f} GB free at {probe} (target: {path})"
    if free_gb < warn_below_gb:
        report.add(check, "warning", msg + f" — below the {warn_below_gb:.0f} GB soft threshold")
    else:
        report.add(check, "ok", msg)


def _discover_ffmpeg_dir() -> Optional[str]:
    """Mirrors run_lmudata.py's _discover_ffmpeg_dir(): PATH first, then the
    same common conda-env bin dirs (that prep script's underlying
    prepare_sot_dataset.py find_ffmpeg() does not look inside conda envs)."""
    found = shutil.which("ffmpeg")
    if found:
        return os.path.dirname(found)
    for pat in FFMPEG_ENV_GLOBS:
        hits = sorted(glob.glob(os.path.expanduser(pat)))
        if hits:
            return os.path.dirname(hits[0])
    return None


def check_ffmpeg(report: Report, tasks: List[str], model: Optional[str]) -> None:
    ffmpeg_dir = _discover_ffmpeg_dir()
    sot_targeted = "sot" in tasks or "all" in tasks
    model_norm = (model or "").lower().replace("_", "").replace("-", "").replace(" ", "")
    cosmos_api_targeted = "cosmosreason2api" in model_norm or "cosmosreason2" in model_norm and "api" in model_norm
    required = sot_targeted or cosmos_api_targeted

    if ffmpeg_dir:
        report.add("ffmpeg", "ok", f"found at {ffmpeg_dir}")
        return

    if required:
        report.add("ffmpeg", "blocker",
                    "ffmpeg not found on PATH or in common conda-env bin dirs — required for "
                    "the SOT task's frame extraction and/or the CosmosReason2 API backend.")
    else:
        report.add("ffmpeg", "warning",
                    "ffmpeg not found on PATH or in common conda-env bin dirs. Only required for "
                    "the SOT task and the CosmosReason2 API backend; pass --tasks/--model if you "
                    "are targeting either so this becomes a hard blocker.")


def _resolve_cached_hf_token() -> Optional[str]:
    """Best-effort across huggingface_hub API generations."""
    try:
        from huggingface_hub import get_token  # newer API (>=0.19)
        tok = get_token()
        if tok:
            return tok
    except Exception:
        pass
    try:
        from huggingface_hub import HfFolder  # older API
        tok = HfFolder.get_token()
        if tok:
            return tok
    except Exception:
        pass
    return None


def check_huggingface_hub(report: Report) -> None:
    try:
        import huggingface_hub
        hub_version = getattr(huggingface_hub, "__version__", "unknown")
    except ImportError as e:
        report.add("huggingface_hub", "warning", f"huggingface_hub not importable ({e}).")
        return

    env_token = bool(os.environ.get("HF_TOKEN"))
    cached_token = bool(_resolve_cached_hf_token())
    if env_token or cached_token:
        source = "HF_TOKEN env var" if env_token else "cached (`hf auth login`)"
        report.add("huggingface_hub", "ok", f"huggingface_hub {hub_version}; token present ({source}).")
    else:
        report.add("huggingface_hub", "warning",
                    f"huggingface_hub {hub_version}; no token found (HF_TOKEN unset, no cached "
                    f"`hf auth login`). Only required for gated data (SOT source videos) and "
                    f"gated/private models.")


def check_vlmeval(report: Report) -> None:
    """Import vlmeval.config in a *subprocess*, not in-process.

    A broken native dependency (a mismatched numpy/torch wheel, for
    instance) can segfault the interpreter on import rather than raising a
    catchable Python exception — that would otherwise take the whole
    preflight tool down with it and hide every other finding. Isolating the
    import means a crash here is just one more (blocker) finding.
    """
    code = (
        f"import sys; sys.path.insert(0, {str(_REPO_ROOT)!r}); "
        f"from vlmeval.config import supported_VLM; print(len(supported_VLM))"
    )
    try:
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        report.add("vlmeval", "blocker", "vlmeval.config import timed out after 120s.")
        return
    except OSError as e:
        report.add("vlmeval", "blocker", f"failed to launch import-check subprocess: {e}")
        return

    if proc.returncode != 0:
        tail_source = proc.stderr or proc.stdout
        tail = " | ".join(tail_source.strip().splitlines()[-5:]) if tail_source.strip() else ""
        crash_note = _crash_note(proc.returncode)
        report.add("vlmeval", "blocker",
                    f"vlmeval.config failed to import from {_REPO_ROOT} (subprocess exit "
                    f"{proc.returncode}{crash_note}): {tail}")
        return

    try:
        n = int(proc.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        report.add("vlmeval", "blocker",
                    f"vlmeval.config import subprocess produced unexpected output: {proc.stdout!r}")
        return

    if n == 0:
        report.add("vlmeval", "blocker", "vlmeval imported but supported_VLM is empty — install looks broken.")
    else:
        report.add("vlmeval", "ok", f"vlmeval importable, {n} models registered in supported_VLM.")


def check_network(report: Report) -> None:
    try:
        import requests
    except ImportError as e:
        report.add("network", "warning", f"requests not importable ({e}); skipping reachability checks.")
        return

    for check, url in (("network:huggingface.co", "https://huggingface.co"),
                        ("network:github.com", "https://github.com")):
        try:
            resp = requests.head(url, timeout=4, allow_redirects=True)
            if resp.status_code >= 400:
                resp = requests.get(url, timeout=4)
            if resp.status_code < 400:
                report.add(check, "ok", f"reachable ({resp.status_code})")
            else:
                report.add(check, "warning", f"reachable but returned HTTP {resp.status_code}")
        except Exception as e:
            report.add(check, "warning", f"unreachable within 4s: {type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _resolve_lmu_root(cli_value: Optional[str]) -> Path:
    if cli_value:
        return Path(cli_value).expanduser().resolve()
    env = os.environ.get("LMUData")
    if env and Path(env).exists():
        return Path(env).resolve()
    return DEFAULT_LMU_ROOT


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="preflight_check.py",
        description="Environment sanity check before running VANTAGE-Bench inference.",
    )
    p.add_argument("--lmu-root", type=str, default=None,
                   help=f"LMUData root to check disk space at (default: $LMUData or {DEFAULT_LMU_ROOT})")
    p.add_argument("--work-dir", type=str, default=str(DEFAULT_WORK_DIR),
                   help=f"run.py output directory to check disk space at (default: {DEFAULT_WORK_DIR})")
    p.add_argument("--tasks", type=str, default=None,
                   help="comma-separated task short-keys you intend to run (e.g. 'sot,vqa'). "
                        "Sharpens the ffmpeg blocker/warning classification.")
    p.add_argument("--model", type=str, default=None,
                   help="model/config name you intend to run (e.g. 'cosmos_reason2_api'). "
                        "Sharpens the ffmpeg blocker/warning classification.")
    p.add_argument("--json", action="store_true", help="print a single JSON report object instead of text")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    tasks = [t.strip().lower() for t in args.tasks.split(",")] if args.tasks else []

    report = Report()
    check_platform(report)
    torch_facts = check_torch(report)
    check_nvidia_smi(report, torch_facts)
    check_disk(report, "disk:lmu-root", _resolve_lmu_root(args.lmu_root), warn_below_gb=20.0)
    check_disk(report, "disk:work-dir", Path(args.work_dir).expanduser().resolve(), warn_below_gb=5.0)
    check_ffmpeg(report, tasks, args.model)
    check_huggingface_hub(report)
    check_vlmeval(report)
    check_network(report)

    status = report.worst_status()

    if args.json:
        print(json.dumps({"status": status, "findings": report.findings}, indent=2))
    else:
        print("=" * 78)
        print("VANTAGE-Bench preflight check")
        print("=" * 78)
        width = max(len(f["check"]) for f in report.findings) + 2
        for f in report.findings:
            tag = {"ok": "OK", "warning": "WARN", "blocker": "BLOCK"}[f["status"]]
            print(f"[{tag:<5}] {f['check']:<{width}} {f['message']}")
        print("-" * 78)
        print(f"Overall status: {status.upper()}")
        if status == "blocker":
            print("At least one blocker was found — fix it before running inference.")
        print("=" * 78)

    return 0 if status != "blocker" else 1


if __name__ == "__main__":
    sys.exit(main())
