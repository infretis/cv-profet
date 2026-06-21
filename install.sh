#!/usr/bin/env bash
set -euo pipefail

python3 -m pip install --upgrade pip
python3 -m pip install -e .

echo "Installation complete."
echo "You can now run: cv_builder --help"
echo "You can also run: cv_analyze --help"