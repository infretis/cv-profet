#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys


def _run_module(module_name: str, args: list[str]) -> int:
    cmd = [sys.executable, "-m", module_name, *args]
    return int(subprocess.call(cmd, cwd="."))


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="profet",
        description=(
            "cv-profet command line interface\n\n"
            "Recommended workflow order:\n"
            "1) build     -> always first; builds the CV matrix\n"
            "2) screen    -> estimates predictive power T for CVs\n"
            "3) diagnose  -> generates CV diagnostics as a textfile report, overlap and CV distribution plots, and their integrals\n"
            "4) tmat      -> T-matrix over varying lambda_c and lambda_r\n"
            "5) optimize  -> optimize linear combinations of CVs by predictive power T"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Subcommands:\n"
            "  build     CV matrix builder. This should always be run first.\n"
            "  screen    Determines predictive power of each CV (T score).\n"
            "  diagnose  Generates CV diagnostics as a textfile report, overlap and CV distribution plots, and their integrals.\n"
            "  tmat      Builds a T-matrix for varying lambda_c and lambda_r combinations.\n"
            "  optimize  Builds linear combinations of CVs and optimizes them by T.\n\n"
            "Examples:\n"
            "  profet build --toml infretis.toml\n"
            "  profet screen --toml infretis.toml\n"
            "  profet diagnose --toml infretis.toml\n"
            "  profet tmat --toml infretis.toml\n"
            "  profet optimize --toml infretis.toml"
        ),
    )
    parser.add_argument(
        "command",
        choices=["build", "check_active", "screen", "diagnose", "tmat", "optimize"],
        help="Subcommand to run",
    )
    parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the selected subcommand",
    )

    ns = parser.parse_args()
    forwarded = list(ns.args)

    if ns.command == "build":
        return _run_module("cv_builder", forwarded)

    if ns.command == "check_active":
        return _run_module("cv_builder", ["--check-active", *forwarded])

    # Convenience wrappers for common PPA modes.
    mode_flag = {
        "screen": "--screen",
        "diagnose": "--diagnose",
        "tmat": "--tmat",
        "optimize": "--optimize",
    }[ns.command]
    return _run_module("cv_analyze", [mode_flag, *forwarded])


if __name__ == "__main__":
    raise SystemExit(main())
