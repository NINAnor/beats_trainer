#!/usr/bin/env python3

import torch
import torch.nn as nn


def test_conv2d_truncation():
    """Test exactly how Conv2d handles non-divisible dimensions"""

    print("=== Conv2d Truncation Behavior ===\n")

    # Simulate the BEATs patch embedding layer
    patch_embedding = nn.Conv2d(
        in_channels=1,
        out_channels=512,
        kernel_size=16,  # patch size
        stride=16,  # non-overlapping
        bias=False,
    )

    # Test cases: different time dimensions
    test_cases = [
        (50, 128),  # 50/16 = 3.125 -> should give 3 patches
        (100, 128),  # 100/16 = 6.25 -> should give 6 patches
        (160, 128),  # 160/16 = 10 -> should give 10 patches
        (250, 128),  # 250/16 = 15.625 -> should give 15 patches
        (1000, 128),  # 1000/16 = 62.5 -> should give 62 patches
    ]

    for time_dim, freq_dim in test_cases:
        print(f"\n--- Input: (1, 1, {time_dim}, {freq_dim}) ---")

        # Create input tensor (batch=1, channels=1, time, freq)
        input_tensor = torch.randn(1, 1, time_dim, freq_dim)

        # Apply conv2d
        output = patch_embedding(input_tensor)

        # Calculate expected vs actual
        expected_time_patches = time_dim // 16
        expected_freq_patches = freq_dim // 16  # 128/16 = 8
        actual_time_patches = output.shape[2]
        actual_freq_patches = output.shape[3]

        print(f"Expected time patches: {expected_time_patches} (from {time_dim}/16)")
        print(f"Actual time patches: {actual_time_patches}")
        print(f"Expected freq patches: {expected_freq_patches} (from {freq_dim}/16)")
        print(f"Actual freq patches: {actual_freq_patches}")
        print(f"Output shape: {output.shape}")

        # Check for truncation
        lost_time_frames = time_dim % 16
        lost_freq_bins = freq_dim % 16

        if lost_time_frames > 0:
            print(f"⚠️  TRUNCATED: Lost {lost_time_frames} time frames")
            print(f"   Lost audio duration: ~{lost_time_frames * 10}ms")
        else:
            print("✅ No time truncation")

        if lost_freq_bins > 0:
            print(f"⚠️  TRUNCATED: Lost {lost_freq_bins} frequency bins")
        else:
            print("✅ No frequency truncation")

        # Total sequence length after flattening
        total_patches = actual_time_patches * actual_freq_patches
        print(f"Total sequence length: {total_patches} patches")


if __name__ == "__main__":
    test_conv2d_truncation()
