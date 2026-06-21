from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
CV_BUILDER_ROOT = ROOT / "CV_builder"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(CV_BUILDER_ROOT) not in sys.path:
    sys.path.insert(0, str(CV_BUILDER_ROOT))
