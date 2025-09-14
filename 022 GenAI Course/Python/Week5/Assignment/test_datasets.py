#!/usr/bin/env python3
"""Test script to verify datasets package installation"""

try:
    import datasets
    print(f"✅ SUCCESS: datasets package imported successfully!")
    print(f"📦 Version: {datasets.__version__}")
    print(f"📍 Location: {datasets.__file__}")
    
    # Test basic functionality
    from datasets import load_dataset
    print("✅ SUCCESS: load_dataset function imported successfully!")
    
except ImportError as e:
    print(f"❌ ERROR: Failed to import datasets package")
    print(f"Error details: {e}")
except Exception as e:
    print(f"❌ ERROR: Unexpected error occurred")
    print(f"Error details: {e}")
