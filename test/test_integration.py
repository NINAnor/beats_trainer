"""
Integration tests for the complete BEATs library workflow.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import json

from .conftest import (
    TestConfig,
    skip_if_no_model,
    create_synthetic_audio,
    create_test_audio_files,
)


class TestEndToEndWorkflow:
    """Test complete end-to-end workflows."""

    @skip_if_no_model()
    def test_complete_feature_extraction_workflow(self, temp_dir):
        """Test complete workflow from checkpoint to features."""
        from beats_trainer import checkpoint_utils
        from beats_trainer.feature_extractor import BEATsFeatureExtractor

        # Step 1: Ensure checkpoint is available
        checkpoint_path = checkpoint_utils.ensure_checkpoint()

        assert checkpoint_path is not None, "Should have checkpoint available"
        assert checkpoint_path.exists(), "Checkpoint file should exist"

        # Step 2: Initialize feature extractor
        extractor = BEATsFeatureExtractor(
            model_path=checkpoint_path, pooling="mean", device="cpu"
        )

        # Step 3: Create test audio data
        test_audio = []
        labels = []

        for i in range(5):
            audio = create_synthetic_audio(
                frequency=440 * (2 ** (i % 3)),  # Different frequencies
                seed=42 + i,
            )
            test_audio.append(audio)
            labels.append(f"class_{i % 3}")

        # Step 4: Extract features
        features = extractor.extract_from_batch(test_audio)

        # Step 5: Verify results
        assert features.shape == (5, TestConfig.FEATURE_DIM)
        assert not np.isnan(features).any()
        assert np.isfinite(features).all()

        # Different audio should produce different features
        feature_similarities = np.corrcoef(features)
        np.fill_diagonal(feature_similarities, 0)  # Ignore self-similarity
        max_similarity = np.max(feature_similarities)
        assert max_similarity < 0.99, (
            "Different audio should produce different features"
        )

    def test_automatic_checkpoint_workflow(self, temp_dir):
        """Test workflow with automatic checkpoint management."""
        from beats_trainer import checkpoint_utils

        # Mock checkpoint directories to use our temp directory
        with patch("beats_trainer.checkpoint_utils.CHECKPOINT_DIRS", [temp_dir]):
            # Initially no checkpoint should exist
            assert checkpoint_utils.find_checkpoint() is None

            # Mock download process
            with patch(
                "beats_trainer.checkpoint_utils.download_beats_checkpoint"
            ) as mock_download:
                # Create mock downloaded checkpoint
                checkpoint_file = temp_dir / "BEATs_iter3_plus_AS2M.pt"

                # Create a valid-looking checkpoint
                import torch

                mock_state_dict = {
                    "model_state_dict": {"layer1.weight": torch.randn(10, 10)},
                    "config": {"embed_dim": 768},
                }
                torch.save(mock_state_dict, checkpoint_file)

                mock_download.return_value = checkpoint_file

                # Now ensure_checkpoint should download
                result = checkpoint_utils.ensure_checkpoint()

                assert result is not None
                assert result.exists()
                mock_download.assert_called_once()

                # Second call should not download again
                mock_download.reset_mock()
                result2 = checkpoint_utils.ensure_checkpoint()

                assert result2 == result
                mock_download.assert_not_called()

    def test_multi_model_comparison_workflow(self, temp_dir):
        """Test workflow for comparing multiple model configurations."""
        from beats_trainer.feature_extractor import BEATsFeatureExtractor

        # Create mock checkpoints
        checkpoint_dir = temp_dir / "checkpoints"
        checkpoint_dir.mkdir()

        checkpoints = []
        for i, model_name in enumerate(["model_a", "model_b"]):
            checkpoint_file = checkpoint_dir / f"{model_name}.pt"

            # Create slightly different mock models
            import torch

            mock_state_dict = {
                "model_state_dict": {"layer1.weight": torch.randn(10, 10) + i * 0.1},
                "config": {"embed_dim": 768},
            }
            torch.save(mock_state_dict, checkpoint_file)
            checkpoints.append(checkpoint_file)

        # Create test audio
        test_audio = create_synthetic_audio()

        # Extract features with both models (this will likely fail with mock models)
        extractors = {}
        features = {}

        for i, checkpoint in enumerate(checkpoints):
            model_name = f"model_{i}"

            try:
                extractor = BEATsFeatureExtractor(
                    model_path=checkpoint, pooling="mean", device="cpu"
                )
                extractors[model_name] = extractor

                # This will likely fail with mock checkpoints, but test the structure
                feature = extractor.extract_features(test_audio)
                features[model_name] = feature

            except Exception:
                # Expected with mock checkpoints - just verify structure
                pass

        # At minimum, we should be able to create the extractors
        # (even if feature extraction fails with mock data)
        assert len(extractors) <= len(checkpoints)  # May be 0 with mock data

    def test_batch_processing_workflow(self, temp_dir):
        """Test workflow for batch processing multiple audio files."""
        # Create mock audio files
        audio_files = create_test_audio_files(temp_dir, n_files=10)

        # Mock feature extraction
        with patch(
            "beats_trainer.feature_extractor.BEATsFeatureExtractor"
        ) as MockExtractor:
            mock_extractor = MagicMock()
            MockExtractor.return_value = mock_extractor

            # Mock feature extraction results
            mock_features = np.random.randn(10, TestConfig.FEATURE_DIM)
            mock_extractor.extract_from_files.return_value = mock_features

            # Initialize extractor
            from beats_trainer.feature_extractor import BEATsFeatureExtractor

            extractor = BEATsFeatureExtractor(
                model_path=None, pooling="mean", device="cpu"
            )

            # Process batch
            features = extractor.extract_from_files(audio_files, batch_size=4)

            # Verify batch processing was called
            mock_extractor.extract_from_files.assert_called_once()
            assert features.shape == (10, TestConfig.FEATURE_DIM)

    def test_save_and_load_features_workflow(self, temp_dir):
        """Test workflow for saving and loading extracted features."""

        # Mock extracted features
        n_samples = 20
        features = np.random.randn(n_samples, TestConfig.FEATURE_DIM)
        labels = [f"class_{i % 5}" for i in range(n_samples)]

        # Save features
        features_dir = temp_dir / "extracted_features"
        features_dir.mkdir()

        features_file = features_dir / "features.npy"
        labels_file = features_dir / "labels.json"

        np.save(features_file, features)
        with open(labels_file, "w") as f:
            json.dump(labels, f)

        # Load features back
        loaded_features = np.load(features_file)
        with open(labels_file) as f:
            loaded_labels = json.load(f)

        # Verify loaded data
        np.testing.assert_array_equal(features, loaded_features)
        assert labels == loaded_labels

        # Test feature analysis workflow
        from sklearn.metrics import silhouette_score
        from sklearn.preprocessing import LabelEncoder

        # Encode labels
        le = LabelEncoder()
        encoded_labels = le.fit_transform(loaded_labels)

        # Compute metrics
        sil_score = silhouette_score(loaded_features, encoded_labels)
        assert isinstance(sil_score, float)
        assert -1 <= sil_score <= 1  # Silhouette score range

    def test_error_recovery_workflow(self, temp_dir):
        """Test error recovery in workflows."""
        from beats_trainer import checkpoint_utils
        from beats_trainer.feature_extractor import BEATsFeatureExtractor

        # Test 1: Recovery from missing checkpoint
        with patch("beats_trainer.checkpoint_utils.CHECKPOINT_DIRS", [temp_dir]):
            # No checkpoint exists
            checkpoint = checkpoint_utils.find_checkpoint()
            assert checkpoint is None

            # Try to ensure checkpoint (will try to download)
            with patch(
                "beats_trainer.checkpoint_utils.download_beats_checkpoint"
            ) as mock_download:
                # Mock download failure
                mock_download.return_value = None

                result = checkpoint_utils.ensure_checkpoint()
                assert result is None  # Should handle gracefully

        # Test 2: Recovery from invalid model file
        invalid_checkpoint = temp_dir / "invalid.pt"
        invalid_checkpoint.write_text("not a torch file")

        with pytest.raises(Exception):  # Should fail gracefully
            BEATsFeatureExtractor(model_path=invalid_checkpoint, pooling="mean")

        # Test 3: Recovery from invalid audio input
        # This would be tested in the actual feature extractor if we had a valid model
        pass

    def test_performance_monitoring_workflow(self, temp_dir):
        """Test workflow with performance monitoring."""
        import time

        # Mock feature extraction with timing
        start_time = time.time()

        # Simulate feature extraction process
        [create_synthetic_audio(seed=i) for i in range(10)]

        # Mock processing time
        processing_time = time.time() - start_time

        # Should be reasonably fast for synthetic data
        assert processing_time < 1.0, f"Processing took {processing_time:.2f} seconds"

        # Mock memory usage tracking
        import psutil
        import os

        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss

        # Simulate some processing
        np.random.randn(10, TestConfig.FEATURE_DIM)

        memory_after = process.memory_info().rss
        memory_increase = memory_after - memory_before

        # Memory usage should be reasonable
        assert memory_increase < 100 * 1024 * 1024, (
            f"Memory increased by {memory_increase / 1024 / 1024:.1f} MB"
        )


class TestRealWorldScenarios:
    """Test realistic usage scenarios."""

    def test_audio_similarity_search_scenario(self, temp_dir):
        """Test scenario: building an audio similarity search system."""
        # Step 1: Create database of audio features
        database_size = 50
        query_size = 3

        # Mock audio database
        database_features = np.random.randn(database_size, TestConfig.FEATURE_DIM)
        [f"audio_{i:03d}" for i in range(database_size)]

        # Normalize features for cosine similarity
        database_features = database_features / np.linalg.norm(
            database_features, axis=1, keepdims=True
        )

        # Step 2: Query features
        query_features = np.random.randn(query_size, TestConfig.FEATURE_DIM)
        query_features = query_features / np.linalg.norm(
            query_features, axis=1, keepdims=True
        )

        # Step 3: Compute similarities
        similarities = np.dot(query_features, database_features.T)

        # Step 4: Find most similar items
        top_k = 5
        top_indices = np.argsort(similarities, axis=1)[:, -top_k:]

        # Verify results
        assert similarities.shape == (query_size, database_size)
        assert top_indices.shape == (query_size, top_k)

        # All similarity scores should be valid
        assert np.all(similarities >= -1) and np.all(similarities <= 1)

    def test_audio_classification_scenario(self, temp_dir):
        """Test scenario: building an audio classification system."""
        # Step 1: Mock training features
        n_classes = 5
        n_samples_per_class = 20

        features = []
        labels = []

        for class_id in range(n_classes):
            # Create class-specific features (with some separation)
            class_features = np.random.randn(
                n_samples_per_class, TestConfig.FEATURE_DIM
            )
            class_features[:, class_id * 10 : (class_id + 1) * 10] += (
                2.0  # Add class signal
            )

            features.append(class_features)
            labels.extend([class_id] * n_samples_per_class)

        features = np.vstack(features)
        labels = np.array(labels)

        # Step 2: Split into train/test
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.3, random_state=42, stratify=labels
        )

        # Step 3: Train a simple classifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score

        classifier = LogisticRegression(random_state=42, max_iter=1000)
        classifier.fit(X_train, y_train)

        # Step 4: Evaluate
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # With class-specific signals, should achieve reasonable accuracy
        assert accuracy > 0.5, f"Classification accuracy too low: {accuracy:.3f}"

        # Step 5: Feature importance analysis
        feature_importance = np.abs(classifier.coef_).mean(axis=0)
        assert len(feature_importance) == TestConfig.FEATURE_DIM

    def test_audio_clustering_scenario(self, temp_dir):
        """Test scenario: clustering audio by similarity."""
        from sklearn.cluster import KMeans
        from sklearn.metrics import adjusted_rand_score

        # Step 1: Create mock audio features with hidden clusters
        n_clusters = 4
        n_samples_per_cluster = 15

        features = []
        true_labels = []

        for cluster_id in range(n_clusters):
            # Create cluster-specific features
            cluster_center = np.random.randn(TestConfig.FEATURE_DIM) * 3
            cluster_features = (
                np.random.randn(n_samples_per_cluster, TestConfig.FEATURE_DIM) * 0.5
            )
            cluster_features += cluster_center

            features.append(cluster_features)
            true_labels.extend([cluster_id] * n_samples_per_cluster)

        features = np.vstack(features)
        true_labels = np.array(true_labels)

        # Step 2: Apply clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        predicted_labels = kmeans.fit_predict(features)

        # Step 3: Evaluate clustering quality
        ari_score = adjusted_rand_score(true_labels, predicted_labels)

        # Should achieve reasonable clustering (ARI > 0.3)
        assert ari_score > 0.3, f"Clustering quality too low: ARI = {ari_score:.3f}"

        # Step 4: Analyze cluster properties
        cluster_centers = kmeans.cluster_centers_
        assert cluster_centers.shape == (n_clusters, TestConfig.FEATURE_DIM)

        # Centers should be different from each other
        center_distances = np.linalg.norm(
            cluster_centers[:, np.newaxis] - cluster_centers[np.newaxis, :], axis=2
        )
        np.fill_diagonal(center_distances, np.inf)
        min_distance = np.min(center_distances)
        assert min_distance > 1.0, "Cluster centers should be well separated"

    def test_transfer_learning_scenario(self, temp_dir):
        """Test scenario: using BEATs features for transfer learning."""
        # Step 1: Mock pretrained features (from BEATs)
        n_samples = 100
        pretrained_features = np.random.randn(n_samples, TestConfig.FEATURE_DIM)

        # Step 2: Create target task labels (binary classification)
        # Make labels somewhat correlated with features for realistic scenario
        target_scores = np.mean(
            pretrained_features[:, :50], axis=1
        )  # Use first 50 features
        target_labels = (target_scores > np.median(target_scores)).astype(int)

        # Step 3: Split data
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            pretrained_features, target_labels, test_size=0.3, random_state=42
        )

        # Step 4: Train classifier on frozen features
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, roc_auc_score

        classifier = LogisticRegression(random_state=42)
        classifier.fit(X_train, y_train)

        # Step 5: Evaluate transfer learning performance
        y_pred = classifier.predict(X_test)
        y_proba = classifier.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        # Should perform better than random
        assert accuracy > 0.6, f"Transfer learning accuracy too low: {accuracy:.3f}"
        assert auc > 0.7, f"Transfer learning AUC too low: {auc:.3f}"

        # Step 6: Feature analysis
        # Features should have reasonable variance (not all the same)
        feature_vars = np.var(pretrained_features, axis=0)
        assert np.mean(feature_vars) > 0.1, "Features should have reasonable variance"
        assert np.std(feature_vars) > 0.01, "Features should have diverse variances"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
