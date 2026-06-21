#!/usr/bin/env python3
import json
import os
import subprocess
import sys
from importlib import metadata
from pathlib import Path

ENTRYPOINT_REL = Path("CV_builder") / "main.py"


def _resolve_repo_root() -> Path | None:
    # 1) Explicit override for HPC/module environments.
    env_root = os.environ.get("CV_PROFET_ROOT") or os.environ.get("CV_ANA_ROOT")
    if env_root:
        root = Path(env_root).expanduser().resolve()
        if (root / ENTRYPOINT_REL).exists():
            return root

    # 2) Standard editable-install metadata (direct_url.json).
    try:
        dist = metadata.distribution("cv-profet")
        raw = dist.read_text("direct_url.json")
        if raw:
            data = json.loads(raw)
            url = data.get("url", "")
            if url.startswith("file://"):
                root = Path(url[7:]).resolve()
                if (root / ENTRYPOINT_REL).exists():
                    return root
    except Exception:
        pass

    # 3) Fallback: running directly from the repository root.
    cwd = Path.cwd().resolve()
    if (cwd / ENTRYPOINT_REL).exists():
        return cwd

    return None


def main() -> int:
    run_root = _resolve_repo_root()
    entrypoint = (run_root / ENTRYPOINT_REL) if run_root else None

    if entrypoint is None or not entrypoint.exists():
        print(
            "Error: could not locate cv-profet root for CV_builder/main.py. "
            "Set CV_PROFET_ROOT=/absolute/path/to/cv-profet.",
            file=sys.stderr,
        )
        return 2

    cmd = [sys.executable, str(entrypoint), *sys.argv[1:]]
    return int(subprocess.call(cmd, cwd=str(Path.cwd())))


if __name__ == "__main__":
    raise SystemExit(main())
