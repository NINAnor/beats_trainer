"""
Test suite for BEATs checkpoint management functionality.
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from beats_trainer.checkpoint_utils import (
    list_available_models,
    find_checkpoint,
    download_beats_checkpoint,
    ensure_checkpoint,
    get_model_info,
    validate_checkpoint,
)


class TestCheckpointUtils:
    """Test cases for checkpoint utilities."""

    def test_list_available_models(self):
        """Test listing available models."""
        models = list_available_models()

        assert isinstance(models, dict)
        assert len(models) > 0

        # Check expected model is present
        assert "BEATs_iter3_plus_AS2M" in models

        # Check model info structure
        for model_name, info in models.items():
            assert "hf_repo" in info
            assert "hf_filename" in info
            assert "description" in info
            assert "size_mb" in info
            assert isinstance(info["size_mb"], (int, float))

    def test_get_model_info(self):
        """Test getting model information."""
        # Test known model
        info = get_model_info("BEATs_iter3_plus_AS2M")

        assert info is not None
        assert "hf_repo" in info
        assert "hf_filename" in info
        assert "description" in info
        assert "size_mb" in info

        # Test unknown model
        with pytest.raises(ValueError):
            get_model_info("nonexistent_model")

    def test_find_checkpoint_existing(self, temp_dir):
        """Test finding existing checkpoint."""
        # Create a mock checkpoint file
        checkpoint_dir = temp_dir / "checkpoints"
        checkpoint_dir.mkdir()

        checkpoint_file = checkpoint_dir / "BEATs_iter3_plus_AS2M.pt"
        checkpoint_file.write_text("mock checkpoint data")

        # Mock the default checkpoint directories to include our temp dir
        with patch("beats_trainer.checkpoint_utils.CHECKPOINT_DIRS", [checkpoint_dir]):
            found_checkpoint = find_checkpoint()

            assert found_checkpoint is not None
            assert found_checkpoint.exists()
            assert found_checkpoint.name == "BEATs_iter3_plus_AS2M.pt"

    def test_find_checkpoint_nonexistent(self, temp_dir):
        """Test finding checkpoint when none exists."""
        # Use empty directory
        empty_dir = temp_dir / "empty"
        empty_dir.mkdir()

        with patch("beats_trainer.checkpoint_utils.CHECKPOINT_DIRS", [empty_dir]):
            found_checkpoint = find_checkpoint()
            assert found_checkpoint is None

    def test_find_checkpoint_with_specific_name(self, temp_dir):
        """Test finding checkpoint with specific model name."""
        checkpoint_dir = temp_dir / "checkpoints"
        checkpoint_dir.mkdir()

        # Create multiple checkpoint files
        checkpoint1 = checkpoint_dir / "BEATs_iter3_plus_AS2M.pt"
        checkpoint2 = checkpoint_dir / "other_model.pt"
        checkpoint1.write_text("mock data 1")
        checkpoint2.write_text("mock data 2")

        with patch("beats_trainer.checkpoint_utils.CHECKPOINT_DIRS", [checkpoint_dir]):
            # Find specific model
            found_checkpoint = find_checkpoint("other_model")
            assert found_checkpoint.name == "other_model.pt"

            # Find default model
            found_checkpoint = find_checkpoint()
            assert found_checkpoint.name == "BEATs_iter3_plus_AS2M.pt"

    @patch("urllib.request.urlopen")
    @patch(
        "beats_trainer.checkpoint_utils.CHECKPOINT_DIRS",
        [Path("/tmp/test_checkpoints")],
    )
    def test_download_beats_checkpoint_success(self, mock_urlopen, temp_dir):
        """Test successful checkpoint download."""
        # Setup mock response for urllib
        mock_response = MagicMock()
        mock_response.headers = {"content-length": "100"}
        # Mock the read method to return data in chunks
        read_data = [b"mock_data_chunk"] * 10 + [b""]  # 10 chunks + empty to signal end
        mock_response.read.side_effect = read_data
        mock_response.__enter__ = lambda self: mock_response
        mock_response.__exit__ = lambda self, *args: None
        mock_urlopen.return_value = mock_response

        # Create temporary checkpoint directory
        checkpoint_dir = temp_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)

        with patch("beats_trainer.checkpoint_utils.CHECKPOINT_DIRS", [checkpoint_dir]):
            # Download checkpoint (force download to test actual download)
            checkpoint_path = download_beats_checkpoint(
                "BEATs_iter3_plus_AS2M",
                cache_dir=str(checkpoint_dir),
                force_download=True,
            )

            assert checkpoint_path is not None
            assert checkpoint_path.exists()
            assert checkpoint_path.name == "BEATs_iter3_plus_AS2M.pt"

            # Verify content
            content = checkpoint_path.read_bytes()
            expected_content = b"mock_data_chunk" * 10
            assert content == expected_content

    @patch("requests.get")
    def test_download_beats_checkpoint_failure(self, mock_get):
        """Test checkpoint download failure."""
        # Test with non-existent model (should raise ValueError)
        with pytest.raises(ValueError):
            download_beats_checkpoint("nonexistent_model")

    @patch("urllib.request.urlopen")
    def test_download_beats_checkpoint_network_error(self, mock_urlopen, temp_dir):
        """Test checkpoint download with network error."""
        # Setup mock network error
        mock_urlopen.side_effect = Exception("Network error")

        # Create clean test directory
        checkpoint_dir = temp_dir / "net_error_checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)

        # Test download failure (should raise exception)
        with pytest.raises(Exception, match="Network error"):
            download_beats_checkpoint(
                "BEATs_iter3_plus_AS2M",
                cache_dir=str(checkpoint_dir),
                force_download=True,
            )

    def test_validate_checkpoint_valid(self, temp_dir):
        """Test checkpoint validation with valid file."""
        # Create mock valid checkpoint
        checkpoint_file = temp_dir / "valid_checkpoint.pt"

        # Create a mock torch state dict
        mock_state_dict = {
            "model_state_dict": {"layer1.weight": "data"},
            "config": {"some_config": "value"},
        }

        import torch

        torch.save(mock_state_dict, checkpoint_file)

        # Validate checkpoint
        is_valid = validate_checkpoint(checkpoint_file)
        assert is_valid is True

    def test_validate_checkpoint_invalid(self, temp_dir):
        """Test checkpoint validation with invalid file."""
        # Create invalid checkpoint (not a torch file)
        checkpoint_file = temp_dir / "invalid_checkpoint.pt"
        checkpoint_file.write_text("not a torch file")

        # Validation should fail
        is_valid = validate_checkpoint(checkpoint_file)
        assert is_valid is False

    def test_validate_checkpoint_nonexistent(self):
        """Test checkpoint validation with nonexistent file."""
        nonexistent_file = Path("/nonexistent/checkpoint.pt")

        is_valid = validate_checkpoint(nonexistent_file)
        assert is_valid is False

    def test_ensure_checkpoint_existing(self, temp_dir):
        """Test ensure_checkpoint when checkpoint already exists."""
        # Create existing checkpoint
        checkpoint_dir = temp_dir / "checkpoints"
        checkpoint_dir.mkdir()
        checkpoint_file = checkpoint_dir / "BEATs_iter3_plus_AS2M.pt"

        # Create a valid mock checkpoint
        import torch

        torch.save({"model": "data"}, checkpoint_file)

        with patch("beats_trainer.checkpoint_utils.CHECKPOINT_DIRS", [checkpoint_dir]):
            result = ensure_checkpoint()

            assert result is not None
            assert result.exists()
            assert result == checkpoint_file

    @patch("beats_trainer.checkpoint_utils.download_beats_checkpoint")
    def test_ensure_checkpoint_download(self, mock_download, temp_dir):
        """Test ensure_checkpoint when download is needed."""
        # Setup mock download
        checkpoint_dir = temp_dir / "checkpoints"
        checkpoint_dir.mkdir()
        checkpoint_file = checkpoint_dir / "BEATs_iter3_plus_AS2M.pt"

        mock_download.return_value = checkpoint_file

        with patch("beats_trainer.checkpoint_utils.CHECKPOINT_DIRS", [checkpoint_dir]):
            # Ensure no existing checkpoints
            with patch(
                "beats_trainer.checkpoint_utils.find_checkpoint", return_value=None
            ):
                result = ensure_checkpoint()

                assert result is not None
                mock_download.assert_called_once()

    @patch("beats_trainer.checkpoint_utils.download_beats_checkpoint")
    @patch("beats_trainer.checkpoint_utils.find_checkpoint")
    def test_ensure_checkpoint_download_failure(self, mock_find, mock_download):
        """Test ensure_checkpoint when download fails."""
        mock_find.return_value = None  # No existing checkpoint
        mock_download.return_value = None  # Download fails

        result = ensure_checkpoint()
        assert result is None

    def test_ensure_checkpoint_with_specific_model(self, temp_dir):
        """Test ensure_checkpoint with specific model name."""
        checkpoint_dir = temp_dir / "checkpoints"
        checkpoint_dir.mkdir()

        # Create specific model checkpoint
        specific_checkpoint = checkpoint_dir / "specific_model.pt"
        import torch

        torch.save({"model": "data"}, specific_checkpoint)

        with patch("beats_trainer.checkpoint_utils.CHECKPOINT_DIRS", [checkpoint_dir]):
            result = ensure_checkpoint("specific_model")

            assert result is not None
            assert result.name == "specific_model.pt"

    def test_checkpoint_dirs_priority(self, temp_dir):
        """Test that checkpoint directories are searched in correct priority order."""
        # Create multiple directories with checkpoints
        dir1 = temp_dir / "dir1"
        dir2 = temp_dir / "dir2"
        dir1.mkdir()
        dir2.mkdir()

        # Create checkpoints in both directories
        checkpoint1 = dir1 / "BEATs_iter3_plus_AS2M.pt"
        checkpoint2 = dir2 / "BEATs_iter3_plus_AS2M.pt"
        checkpoint1.write_text("checkpoint in dir1")
        checkpoint2.write_text("checkpoint in dir2")

        # dir1 should have priority (first in list)
        with patch("beats_trainer.checkpoint_utils.CHECKPOINT_DIRS", [dir1, dir2]):
            found = find_checkpoint()

            assert found == checkpoint1
            assert found.read_text() == "checkpoint in dir1"


class TestCheckpointIntegration:
    """Integration tests for checkpoint management."""

    def test_checkpoint_workflow_complete(self, temp_dir):
        """Test complete checkpoint management workflow."""
        checkpoint_dir = temp_dir / "checkpoints"
        checkpoint_dir.mkdir()

        with patch("beats_trainer.checkpoint_utils.CHECKPOINT_DIRS", [checkpoint_dir]):
            # Step 1: No checkpoint exists initially
            assert find_checkpoint() is None

            # Step 2: Mock download
            with patch(
                "beats_trainer.checkpoint_utils.download_beats_checkpoint"
            ) as mock_download:
                # Create mock downloaded checkpoint
                checkpoint_file = checkpoint_dir / "BEATs_iter3_plus_AS2M.pt"
                import torch

                torch.save({"model": "data"}, checkpoint_file)
                mock_download.return_value = checkpoint_file

                # Step 3: Ensure checkpoint (should download)
                result = ensure_checkpoint()

                assert result is not None
                assert result.exists()
                mock_download.assert_called_once()

                # Step 4: Ensure checkpoint again (should find existing)
                mock_download.reset_mock()
                result2 = ensure_checkpoint()

                assert result2 == result
                mock_download.assert_not_called()  # Should not download again

                # Step 5: Validate the checkpoint
                assert validate_checkpoint(result) is True

    def test_model_info_consistency(self):
        """Test that model info is consistent across functions."""
        models = list_available_models()

        for model_name in models.keys():
            # Get info through different methods
            info1 = models[model_name]
            info2 = get_model_info(model_name)

            # Should be identical
            assert info1 == info2

            # Required fields should be present
            assert "url" in info1
            assert "description" in info1
            assert "size_mb" in info1


if __name__ == "__main__":
    pytest.main([__file__])
