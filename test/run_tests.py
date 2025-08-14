"""
Test suite runner and configuration.
"""

import pytest
import sys
from pathlib import Path


def run_all_tests():
    """Run all tests in the test suite."""
    test_args = [
        str(Path(__file__).parent),
        "-v",  # Verbose output
        "--tb=short",  # Short traceback format
        "--durations=10",  # Show 10 slowest tests
        "--strict-markers",  # Treat unknown markers as errors
        "--strict-config",  # Treat unknown config options as errors
    ]

    return pytest.main(test_args)


def run_unit_tests():
    """Run only unit tests (fast tests)."""
    test_args = [
        str(Path(__file__).parent / "test_feature_extractor.py"),
        str(Path(__file__).parent / "test_checkpoint_utils.py"),
        str(Path(__file__).parent / "test_cli_and_tools.py"),
        "-v",
        "-m",
        "not slow",  # Exclude slow tests
        "--tb=short",
    ]

    return pytest.main(test_args)


def run_integration_tests():
    """Run integration tests."""
    test_args = [
        str(Path(__file__).parent / "test_integration.py"),
        str(Path(__file__).parent / "test_beats_model.py"),
        "-v",
        "--tb=short",
    ]

    return pytest.main(test_args)


def run_quick_tests():
    """Run a quick subset of tests for development."""
    test_args = [
        str(Path(__file__).parent),
        "-v",
        "-x",  # Stop on first failure
        "--tb=line",  # One line per failure
        "-k",
        "not (slow or gpu)",  # Exclude slow and GPU tests
    ]

    return pytest.main(test_args)


def run_tests_with_coverage():
    """Run tests with coverage reporting."""
    try:
        import importlib.util

        if importlib.util.find_spec("pytest_cov") is None:
            raise ImportError
    except ImportError:
        print("pytest-cov not installed. Install with: pip install pytest-cov")
        return 1
        print("pytest-cov not installed. Install with: pip install pytest-cov")
        return 1

    test_args = [
        str(Path(__file__).parent),
        "--cov=beats_trainer",  # Coverage for the main package
        "--cov-report=html",  # HTML coverage report
        "--cov-report=term",  # Terminal coverage report
        "--cov-fail-under=50",  # Fail if coverage below 50%
        "-v",
    ]

    return pytest.main(test_args)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "unit":
            exit_code = run_unit_tests()
        elif command == "integration":
            exit_code = run_integration_tests()
        elif command == "quick":
            exit_code = run_quick_tests()
        elif command == "coverage":
            exit_code = run_tests_with_coverage()
        elif command == "all":
            exit_code = run_all_tests()
        else:
            print(f"Unknown command: {command}")
            print("Available commands: unit, integration, quick, coverage, all")
            exit_code = 1
    else:
        # Default: run all tests
        exit_code = run_all_tests()

    sys.exit(exit_code)
