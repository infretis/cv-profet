#!/usr/bin/env python3
import argparse


def main() -> int:
    parser = argparse.ArgumentParser(prog="synthetic-cv-builder")
    parser.add_argument("--toml", default="infretis.toml")
    parser.add_argument("--h5-input", default=None)
    parser.add_argument("--load-dir", default=None)
    parser.add_argument("--data", default=None)
    parser.parse_args()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
