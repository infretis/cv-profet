import os
import subprocess
import sys
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent
SYNTHETIC_ROOT = HERE / "data" / "synthetic_repo"


@pytest.mark.heavy
def test_cli_builder_smoke_runs_with_synthetic_repo(tmp_path):
    env = os.environ.copy()
    env["CV_PROFET_ROOT"] = str(SYNTHETIC_ROOT)

    cmd = [
        sys.executable,
        "-m",
        "profet",
        "build",
        "--toml",
        "infretis.toml",
        "--h5-input",
        "fake.h5",
        "--load-dir",
        "load",
        "--data",
        "infretis_data.txt",
    ]

    proc = subprocess.run(
        cmd,
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr


@pytest.mark.heavy
def test_cli_analyze_smoke_runs_all_modes_with_synthetic_repo(tmp_path):
    env = os.environ.copy()
    env["CV_PROFET_ROOT"] = str(SYNTHETIC_ROOT)

    modes = ["screen", "tmat", "diagnose", "optimize"]
    for mode in modes:
        cmd = [sys.executable, "-m", "profet", mode, "--toml", "infretis.toml"]
        proc = subprocess.run(
            cmd,
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, f"mode={mode}\n{proc.stderr}"
