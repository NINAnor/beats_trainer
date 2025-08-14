#!/usr/bin/env python3
"""
BEATs Trainer Installation Test Script

Run this script after installation to verify everything works correctly.
"""

import sys
import importlib
import numpy as np


def test_imports():
    """Test that all required modules can be imported."""
    print("🔍 Testing imports...")

    required_modules = [
        "beats_trainer",
        "beats_trainer.feature_extractor",
        "beats_trainer.checkpoint_utils",
    ]

    optional_modules = [
        "beats_trainer.trainer",
        "beats_trainer.model",
        "beats_trainer.data_module",
    ]

    for module in required_modules:
        try:
            importlib.import_module(module)
            print(f"  ✅ {module}")
        except ImportError as e:
            print(f"  ❌ {module}: {e}")
            return False

    for module in optional_modules:
        try:
            importlib.import_module(module)
            print(f"  ✅ {module} (optional)")
        except ImportError:
            print(f"  ⚠️  {module} (optional - not available)")

    return True


def test_feature_extractor():
    """Test the BEATs feature extractor."""
    print("\n🧪 Testing BEATs Feature Extractor...")

    try:
        from beats_trainer import BEATsFeatureExtractor

        # Create extractor (this will download model if needed)
        print("  📥 Initializing extractor (may download model)...")
        extractor = BEATsFeatureExtractor()

        # Test with synthetic audio
        print("  🔊 Testing with synthetic audio...")
        audio = np.random.randn(16000).astype(np.float32)  # 1 second
        features = extractor.extract_from_array(audio)

        print("  ✅ Feature extraction successful!")
        print(f"     Input: {audio.shape} (1 second of audio)")
        print(f"     Output: {features.shape} (feature vector)")
        print(f"     Feature dimension: {extractor.get_feature_dim()}")

        # Test model info
        info = extractor.get_model_info()
        print(f"     Model layers: {info['num_layers']}")
        print(f"     Attention heads: {info['attention_heads']}")
        print(f"     Device: {info['device']}")

        return True

    except Exception as e:
        print(f"  ❌ Feature extractor test failed: {e}")
        return False


def test_checkpoint_utils():
    """Test checkpoint management utilities."""
    print("\n📦 Testing Checkpoint Utilities...")

    try:
        from beats_trainer import list_available_models, ensure_checkpoint

        # Test listing models
        models = list_available_models()
        print(f"  ✅ Available models: {list(models.keys())}")

        # Test ensure checkpoint
        checkpoint_path = ensure_checkpoint()
        print(f"  ✅ Checkpoint ensured at: {checkpoint_path}")

        return True

    except Exception as e:
        print(f"  ❌ Checkpoint utilities test failed: {e}")
        return False


def test_version_info():
    """Test version and package info."""
    print("\n📋 Package Information...")

    try:
        import beats_trainer

        print(f"  📌 Version: {beats_trainer.__version__}")
        print(f"  👤 Author: {beats_trainer.__author__}")
        print(f"  📧 Email: {beats_trainer.__email__}")
        print(f"  🌐 Homepage: {getattr(beats_trainer, '__uri__', 'Not available')}")
        return True
    except Exception as e:
        print(f"  ❌ Version info test failed: {e}")
        return False


def main():
    """Run all installation tests."""
    print("🚀 BEATs Trainer Installation Test")
    print("=" * 50)

    tests = [
        test_version_info,
        test_imports,
        test_checkpoint_utils,
        test_feature_extractor,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1

    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} passed")

    if passed == total:
        print("🎉 All tests passed! BEATs Trainer is ready to use.")
        print("\n🎯 Next Steps:")
        print("  • Try: python -c 'from beats_trainer import BEATsFeatureExtractor'")
        print("  • Explore notebooks: git clone <repo> && cd notebooks/")
        print(
            "  • Check documentation: https://github.com/benjamin-cretois/beats-trainer"
        )
        return True
    else:
        print(f"⚠️  {total - passed} test(s) failed. Please check the installation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
