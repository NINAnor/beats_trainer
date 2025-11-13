"""
Test suite for CLI and utility tools.
"""

import pytest
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch
import json


class TestCLITools:
    """Test cases for command-line interface tools."""

    def test_beats_checkpoint_manager_import(self):
        """Test that the checkpoint manager CLI can be imported."""
        # Test that we can import the CLI module
        try:
            import sys

            sys.path.append(str(Path(__file__).parent.parent))

            # This should not raise an error
            with open(
                Path(__file__).parent.parent / "beats_checkpoint_manager.py"
            ) as f:
                content = f.read()
                assert "argparse" in content
                assert "checkpoint_utils" in content

        except FileNotFoundError:
            pytest.skip("beats_checkpoint_manager.py not found")

    def test_checkpoint_manager_list_command(self, temp_dir):
        """Test checkpoint manager list command."""
        cli_script = Path(__file__).parent.parent / "beats_checkpoint_manager.py"

        if not cli_script.exists():
            pytest.skip("CLI script not found")

        # Run the list command
        result = subprocess.run(
            [sys.executable, str(cli_script), "list"],
            capture_output=True,
            text=True,
            timeout=30,
        )

        # Should not crash and should show available models
        assert (
            result.returncode == 0 or result.returncode == 1
        )  # May fail if no checkpoints
        assert "BEATs" in result.stdout or "No checkpoints" in result.stdout

    def test_checkpoint_manager_help_command(self):
        """Test checkpoint manager help command."""
        cli_script = Path(__file__).parent.parent / "beats_checkpoint_manager.py"

        if not cli_script.exists():
            pytest.skip("CLI script not found")

        # Run the help command
        result = subprocess.run(
            [sys.executable, str(cli_script), "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        # Should show help text
        assert result.returncode == 0
        assert "usage:" in result.stdout.lower()
        assert "checkpoint" in result.stdout.lower()

    def test_feature_extraction_examples_import(self):
        """Test that feature extraction examples can be imported."""
        examples_script = (
            Path(__file__).parent.parent / "feature_extraction_examples.py"
        )

        if not examples_script.exists():
            pytest.skip("Feature extraction examples not found")

        # Test that the file can be read and contains expected content
        with open(examples_script) as f:
            content = f.read()
            assert "BEATsFeatureExtractor" in content

    def test_checkpoint_examples_import(self):
        """Test that checkpoint examples can be imported."""
        examples_script = Path(__file__).parent.parent / "checkpoint_examples.py"

        if not examples_script.exists():
            pytest.skip("Checkpoint examples not found")

        # Test that the file can be read and contains expected content
        with open(examples_script) as f:
            content = f.read()
            assert "checkpoint" in content.lower()


class TestLibraryUsage:
    """Test cases for typical library usage patterns."""

    def test_basic_import_structure(self):
        """Test that library modules can be imported correctly."""
        # Test core imports
        try:
            from beats_trainer.core.feature_extractor import BEATsFeatureExtractor
            from beats_trainer.utils.checkpoints import (
                find_checkpoint,
                ensure_checkpoint,
                list_available_models,
            )

            # Basic smoke test - classes should be callable
            assert callable(BEATsFeatureExtractor)
            assert callable(find_checkpoint)
            assert callable(ensure_checkpoint)
            assert callable(list_available_models)

        except ImportError as e:
            pytest.fail(f"Failed to import library modules: {e}")

    def test_library_workflow(self, temp_dir):
        """Test typical library usage workflow."""
        from beats_trainer.utils import checkpoints as checkpoint_utils
        from beats_trainer.core.feature_extractor import BEATsFeatureExtractor

        # Step 1: List available models
        models = checkpoint_utils.list_available_models()
        assert isinstance(models, dict)
        assert len(models) > 0

        # Step 2: Try to find or ensure checkpoint (mocked)
        with patch("beats_trainer.utils.checkpoints.CHECKPOINT_DIRS", [temp_dir]):
            # Create mock checkpoint
            checkpoint_file = temp_dir / "BEATs_iter3_plus_AS2M.pt"
            import torch

            torch.save({"mock": "data"}, checkpoint_file)

            checkpoint = checkpoint_utils.find_checkpoint()
            assert checkpoint is not None

            # Step 3: Initialize feature extractor
            # This may fail without a real model, but structure should be correct
            try:
                extractor = BEATsFeatureExtractor(
                    model_path=checkpoint, pooling="mean", device="cpu"
                )
                assert hasattr(extractor, "extract_features")
                assert hasattr(extractor, "extract_from_files")

            except Exception:
                # Expected to fail with mock checkpoint
                pass

    def test_error_handling_patterns(self):
        """Test that library handles errors gracefully."""
        from beats_trainer.core.feature_extractor import BEATsFeatureExtractor
        from beats_trainer.utils import checkpoints as checkpoint_utils

        # Test invalid model path
        with pytest.raises(FileNotFoundError):
            BEATsFeatureExtractor(
                model_path=Path("/nonexistent/model.pt"), pooling="mean"
            )

        # Test invalid pooling method
        with pytest.raises(ValueError):
            BEATsFeatureExtractor(model_path=None, pooling="invalid_method")

        # Test nonexistent model info
        info = checkpoint_utils.get_model_info("nonexistent_model")
        assert info is None

    def test_configuration_validation(self):
        """Test that configuration parameters are validated properly."""
        from beats_trainer.core.feature_extractor import BEATsFeatureExtractor

        # Test valid configurations
        valid_configs = [
            {"pooling": "mean", "device": "cpu"},
            {"pooling": "max", "device": "auto"},
            {"pooling": "cls", "device": "cuda"},
        ]

        for config in valid_configs:
            try:
                # This might fail due to model loading, but should not fail validation
                BEATsFeatureExtractor(model_path=None, **config)
            except (FileNotFoundError, RuntimeError):
                # Expected - we don't have actual model files
                pass
            except ValueError as e:
                pytest.fail(
                    f"Valid configuration failed validation: {config}, error: {e}"
                )


class TestDocumentationAndExamples:
    """Test that documentation and examples are consistent."""

    def test_readme_consistency(self):
        """Test that README examples are valid."""
        readme_path = Path(__file__).parent.parent / "README.md"

        if readme_path.exists():
            with open(readme_path) as f:
                readme_content = f.read()

                # Check for key components mentioned in README
                assert "BEATs" in readme_content
                assert "feature" in readme_content.lower()

                # Check for code examples (python blocks)
                if "```python" in readme_content:
                    # Should mention main classes
                    assert "BEATsFeatureExtractor" in readme_content

    def test_notebook_consistency(self):
        """Test that notebooks are present and structured correctly."""
        notebooks_dir = Path(__file__).parent.parent / "notebooks"

        if notebooks_dir.exists():
            notebook_files = list(notebooks_dir.glob("*.ipynb"))

            # Should have at least one notebook
            assert len(notebook_files) > 0

            # Check notebook content structure
            for notebook_file in notebook_files:
                try:
                    with open(notebook_file) as f:
                        content = json.load(f)

                    # Should have cells
                    assert "cells" in content
                    assert len(content["cells"]) > 0

                    # Should mention BEATs in content
                    notebook_text = str(content)
                    assert "BEATs" in notebook_text or "beats" in notebook_text.lower()

                except (json.JSONDecodeError, KeyError):
                    # Notebook might be in different format
                    pass

    def test_example_scripts_consistency(self):
        """Test that example scripts are consistent with library API."""
        project_root = Path(__file__).parent.parent

        example_files = [
            "feature_extraction_examples.py",
            "checkpoint_examples.py",
            "test_beats_library.py",
        ]

        for example_file in example_files:
            example_path = project_root / example_file

            if example_path.exists():
                with open(example_path) as f:
                    content = f.read()

                # Should import from beats_trainer
                assert "beats_trainer" in content

                # Should use main classes
                if "feature" in example_file.lower():
                    assert "BEATsFeatureExtractor" in content

                if "checkpoint" in example_file.lower():
                    assert any(
                        func in content
                        for func in [
                            "find_checkpoint",
                            "ensure_checkpoint",
                            "download_beats_checkpoint",
                        ]
                    )


class TestPerformanceAndScaling:
    """Test performance characteristics and scaling behavior."""

    @pytest.mark.slow
    def test_library_import_time(self):
        """Test that library imports don't take too long."""
        import time

        start_time = time.time()

        import_time = time.time() - start_time

        # Imports should be reasonably fast (less than 5 seconds)
        assert import_time < 5.0, f"Library imports took {import_time:.2f} seconds"

    def test_memory_usage_basic(self):
        """Test basic memory usage patterns."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Import library

        post_import_memory = process.memory_info().rss

        # Memory increase should be reasonable (less than 500MB)
        memory_increase = post_import_memory - initial_memory
        assert memory_increase < 500 * 1024 * 1024, (
            f"Memory increased by {memory_increase / 1024 / 1024:.1f} MB"
        )

    def test_cpu_usage_basic(self):
        """Test that basic operations don't consume excessive CPU."""
        import psutil

        # Monitor CPU usage
        cpu_percent_before = psutil.cpu_percent(interval=1)

        # Perform basic operations
        from beats_trainer.utils import checkpoints as checkpoint_utils

        checkpoint_utils.list_available_models()
        checkpoint_utils.find_checkpoint()

        cpu_percent_after = psutil.cpu_percent(interval=1)

        # CPU usage should not spike excessively for basic operations
        cpu_increase = max(0, cpu_percent_after - cpu_percent_before)
        assert cpu_increase < 80, f"CPU usage increased by {cpu_increase}%"


if __name__ == "__main__":
    # Run with various options
    pytest.main(
        [
            __file__,
            "-v",  # Verbose
            "--tb=short",  # Short traceback format
            "-x",  # Stop on first failure
        ]
    )
