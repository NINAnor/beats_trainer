#!/usr/bin/env python3

import torch


def main():
    print("Loading BEATs checkpoint configuration...")

    try:
        # Load checkpoint
        checkpoint_path = "checkpoints/BEATs_iter3_plus_AS2M.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        cfg = checkpoint["cfg"]

        print("\n=== BEATs Model Configuration ===")
        print(f"input_patch_size: {cfg.get('input_patch_size', 'Not specified')}")
        print(f"embed_dim: {cfg.get('embed_dim', 'Not specified')}")
        print(f"encoder_embed_dim: {cfg.get('encoder_embed_dim', 'Not specified')}")
        print(f"encoder_layers: {cfg.get('encoder_layers', 'Not specified')}")
        print(
            f"encoder_attention_heads: {cfg.get('encoder_attention_heads', 'Not specified')}"
        )

        print("\n=== All Configuration Keys ===")
        for key in sorted(cfg.keys()):
            print(f"{key}: {cfg[key]}")

    except Exception as e:
        print(f"Error: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
