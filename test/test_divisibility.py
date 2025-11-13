#!/usr/bin/env python3

import torch
import numpy as np
import sys

sys.path.append(".")
from beats_trainer.core.feature_extractor import BEATsFeatureExtractor


def test_divisibility_requirement():
    """Test what happens when time dimension is not divisible by 16"""

    print("=== Testing BEATs Time Dimension Divisibility ===\n")

    # Create extractor
    extractor = BEATsFeatureExtractor()

    # Test different audio lengths
    sample_rate = 16000
    test_durations = [
        0.5,  # 8000 samples  -> ~50 time frames  -> 50/16 = 3.125 (not divisible)
        1.0,  # 16000 samples -> ~100 time frames -> 100/16 = 6.25 (not divisible)
        1.6,  # 25600 samples -> ~160 time frames -> 160/16 = 10 (divisible!)
        2.5,  # 40000 samples -> ~250 time frames -> 250/16 = 15.625 (not divisible)
        3.2,  # 51200 samples -> ~320 time frames -> 320/16 = 20 (divisible!)
    ]

    for duration in test_durations:
        print(f"\n--- Testing {duration}s audio ---")

        # Create synthetic audio
        samples = int(duration * sample_rate)
        audio = np.random.randn(samples).astype(np.float32)

        print(f"Audio samples: {samples}")

        # Convert to tensor and preprocess like BEATs does
        audio_tensor = torch.tensor(audio).unsqueeze(0)  # Add batch dim

        # Simulate the filterbank processing
        # BEATs uses 25ms frame_length, 10ms frame_shift
        # This gives us: (duration_seconds / 0.01) frames approximately
        expected_frames = int(duration / 0.01)
        print(f"Expected filterbank frames: ~{expected_frames}")
        print(f"Frames / 16 = {expected_frames / 16:.3f}")

        if expected_frames % 16 == 0:
            print("✅ Divisible by 16 - should work perfectly")
        else:
            print("⚠️  NOT divisible by 16 - potential truncation")
            print(f"   Would truncate to: {(expected_frames // 16) * 16} frames")
            print(f"   Lost frames: {expected_frames % 16}")

        # Actually extract features to see what happens
        try:
            features = extractor.extract_features(audio_tensor)
            print(f"✅ Feature extraction successful: {features.shape}")
        except Exception as e:
            print(f"❌ Feature extraction failed: {e}")


if __name__ == "__main__":
    test_divisibility_requirement()
