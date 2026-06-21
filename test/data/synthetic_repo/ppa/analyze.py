#!/usr/bin/env python3
import argparse


def main() -> int:
    parser = argparse.ArgumentParser(prog="synthetic-cv-analyze")
    parser.add_argument("--toml", default="infretis.toml")
    parser.add_argument("--screen", action="store_true")
    parser.add_argument("--tmat", action="store_true")
    parser.add_argument("--diagnose", action="store_true")
    parser.add_argument("--optimize", action="store_true")
    parser.parse_args()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
