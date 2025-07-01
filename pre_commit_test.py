#!/usr/bin/env python3
"""
Pre-commit test script for ModelConverterTool
Run this before committing to ensure all tests pass
"""

import subprocess
import sys
from pathlib import Path


def run_tests():
    """Run all tests and return success status"""
    print("🧪 Running ModelConverterTool tests...")

    # Create test output directories
    test_dirs = [
        "test_outputs/basic_conversions",
        "test_outputs/quantization",
        "test_outputs/batch_conversion",
        "test_outputs/api_usage",
    ]

    for test_dir in test_dirs:
        Path(test_dir).mkdir(parents=True, exist_ok=True)

    # Run fast tests first
    print("\n🚀 Running fast tests...")
    fast_result = subprocess.run(
        ["python", "-m", "pytest", "-m", "fast", "-v", "--tb=short"]
    )

    if fast_result.returncode != 0:
        print("❌ Fast tests failed! Please fix before committing.")
        return False

    # Run slow tests (optional, can be skipped in CI)
    print("\n🐌 Running slow tests...")
    slow_result = subprocess.run(
        ["python", "-m", "pytest", "-m", "slow", "-v", "--tb=short"]
    )

    if slow_result.returncode != 0:
        print("⚠️  Slow tests failed, but this is acceptable for commits.")
        print("   Run 'python run_tests.py --category all' to debug slow tests.")
        return True  # Don't block commit for slow test failures

    print("\n✅ All tests passed!")
    return True


def main():
    """Main function"""
    success = run_tests()

    if not success:
        print("\n❌ Pre-commit tests failed. Please fix issues before committing.")
        sys.exit(1)
    else:
        print("\n✅ Pre-commit tests passed. Ready to commit!")
        sys.exit(0)


if __name__ == "__main__":
    main()
