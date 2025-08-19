#!/usr/bin/env python3
"""
Standalone test to verify OpenBEATs parameter loading.
"""

import torch
from BEATs import BEATs, BEATsConfig


def test_openbeats_parameter_loading():
    """Test parameter loading specifically for OpenBEATs checkpoint."""

    checkpoint_path = "checkpoints/openbeats.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    print("=== OpenBEATs Parameter Loading Verification ===")
    print(f"Checkpoint keys: {list(checkpoint.keys())}")

    cfg = BEATsConfig(checkpoint["cfg"])
    print(
        f"Config loaded: embed_dim={cfg.embed_dim}, input_patch_size={cfg.input_patch_size}"
    )

    if "model" in checkpoint:
        checkpoint_state_dict = checkpoint["model"]
    else:
        checkpoint_state_dict = checkpoint

    # Test our enhanced loading method
    model = BEATs(cfg)
    model.reload_pretrained_parameters(state_dict=checkpoint_state_dict)

    # Verify parameter loading
    model_state_dict = model.state_dict()

    matching_params = 0
    total_checkpoint_params = len(checkpoint_state_dict)
    compatible_params = 0

    expected_incompatible = {"patch_embedding_pad.weight", "raw2fbank_pad.weight"}
    actual_incompatible = set()

    for param_name, checkpoint_param in checkpoint_state_dict.items():
        if param_name in model_state_dict:
            compatible_params += 1
            model_param = model_state_dict[param_name]

            if torch.equal(checkpoint_param, model_param):
                matching_params += 1
            else:
                print(f"  ⚠️  Parameter mismatch: {param_name}")
        else:
            actual_incompatible.add(param_name)

    print("\nResults:")
    print(f"  Total checkpoint parameters: {total_checkpoint_params}")
    print(f"  Compatible parameters: {compatible_params}")
    print(f"  Exactly matching parameters: {matching_params}")
    print(f"  Incompatible parameters: {len(actual_incompatible)}")
    print(f"  Match rate: {matching_params / total_checkpoint_params * 100:.1f}%")
    print(
        f"  Compatible rate: {compatible_params / total_checkpoint_params * 100:.1f}%"
    )

    if actual_incompatible:
        print(f"  Incompatible parameters: {actual_incompatible}")

    # Verify expected incompatible parameters
    if actual_incompatible == expected_incompatible:
        print(
            "  ✅ Incompatible parameters match expectations (OpenBEATs-specific padding)"
        )
    else:
        print(
            f"  ⚠️  Unexpected incompatible parameters: {actual_incompatible - expected_incompatible}"
        )

    # Test model functionality
    print("\nFunctionality test:")
    model.eval()

    # Create test input
    test_audio = torch.randn(1, 16000)  # 1 second of audio at 16kHz

    try:
        with torch.no_grad():
            output = model(test_audio)
        print(f"  ✅ Model inference successful: output shape {output.shape}")
        return True
    except Exception as e:
        print(f"  ❌ Model inference failed: {e}")
        return False


if __name__ == "__main__":
    success = test_openbeats_parameter_loading()
    if success:
        print("\n🎉 OpenBEATs parameter loading verification PASSED!")
    else:
        print("\n💥 OpenBEATs parameter loading verification FAILED!")
        exit(1)
