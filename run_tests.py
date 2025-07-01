#!/usr/bin/env python3
"""
Test runner script for ModelConverterTool
Run different test categories easily
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def run_pytest(marker=None, verbose=False, coverage=False):
    """Run pytest with specified options"""
    cmd = ["python", "-m", "pytest"]

    if marker:
        cmd.extend(["-m", marker])

    if verbose:
        cmd.append("-v")

    if coverage:
        cmd.extend(["--cov=model_converter_tool", "--cov-report=term-missing"])

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Run ModelConverterTool tests")
    parser.add_argument(
        "--category",
        choices=["basic", "quantization", "batch", "api", "all"],
        default="all",
        help="Test category to run",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument(
        "--coverage", "-c", action="store_true", help="Run with coverage report"
    )
    parser.add_argument(
        "--fast", action="store_true", help="Run only fast tests (exclude slow tests)"
    )

    args = parser.parse_args()

    # Create test output directories
    test_dirs = [
        "test_outputs/basic_conversions",
        "test_outputs/quantization",
        "test_outputs/batch_conversion",
        "test_outputs/api_usage",
    ]

    for test_dir in test_dirs:
        Path(test_dir).mkdir(parents=True, exist_ok=True)

    # Determine marker
    if args.category == "all":
        marker = None
    elif args.fast:
        marker = "not slow"
    else:
        marker = args.category

    # Run tests
    exit_code = run_pytest(marker, args.verbose, args.coverage)

    if exit_code == 0:
        print("\n✅ All tests passed!")
    else:
        print(f"\n❌ Tests failed with exit code {exit_code}")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
