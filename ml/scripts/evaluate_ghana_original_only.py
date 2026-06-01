#!/usr/bin/env python3
"""Evalúa modelo o ensemble en test Ghana **original-only** (un PNG por sujeto)."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_ML_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = _ML_ROOT.parent


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Eval Ghana test original-only.")
    p.add_argument("--model-path", type=Path, default=None)
    p.add_argument("--ensemble", action="store_true")
    p.add_argument("--calibration-json", type=Path, required=True)
    p.add_argument("--ghana-raw", type=Path, default=_ML_ROOT / "data_raw" / "ghana")
    p.add_argument("--output-json", type=Path, default=None)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    with tempfile.TemporaryDirectory(prefix="ghana_test_orig_") as tmp:
        out = Path(tmp) / "ghana_orig"
        prep = [
            sys.executable,
            str(_ML_ROOT / "scripts" / "prepare_ghana_dataset.py"),
            "--input-root",
            str(args.ghana_raw.resolve()),
            "--output-dir",
            str(out),
        ]
        if subprocess.run(prep, cwd=_ML_ROOT, check=False).returncode != 0:
            return 1
        test_dir = out / "test"
        if args.ensemble:
            if not args.model_path:
                print("ensemble requiere --model-path repetido vía evaluate_ensemble", file=sys.stderr)
                return 1
            cmd = [
                sys.executable,
                str(_ML_ROOT / "scripts" / "evaluate_ensemble.py"),
                "--test-dir",
                str(test_dir),
                "--calibration-json",
                str(args.calibration_json.resolve()),
                "--dataset-label",
                "ghana_test_original_only",
            ]
            for mp in [args.model_path] if isinstance(args.model_path, Path) else []:
                cmd.extend(["--model-path", str(mp)])
        else:
            cmd = [
                sys.executable,
                str(_ML_ROOT / "scripts" / "evaluate_dir.py"),
                "--test-dir",
                str(test_dir),
                "--calibration-json",
                str(args.calibration_json.resolve()),
                "--dataset-label",
                "ghana_test_original_only",
            ]
            if args.model_path:
                cmd.extend(["--model-path", str(args.model_path.resolve())])
        rc = subprocess.run(cmd, cwd=_ML_ROOT, check=False)
        if args.output_json and rc.returncode == 0:
            runs = sorted((_ML_ROOT / "artifacts" / "runs").glob("eval_*original_only*.json"))
            if runs:
                shutil.copy2(runs[-1], args.output_json.resolve())
        return rc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
