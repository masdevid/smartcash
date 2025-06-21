#!/usr/bin/env python3
"""Test script to verify package imports"""

import sys
import os

print("Python path:")
for p in sys.path:
    print(f" - {p}")

print("\nTrying to import smartcash.ui...")
try:
    import smartcash.ui
    print("✅ Successfully imported smartcash.ui")
    print(f"smartcash.ui location: {smartcash.ui.__file__}")
    
    print("\nTrying to import from smartcash.ui.setup.env_config...")
    from smartcash.ui.setup.env_config import env_config_initializer
    print("✅ Successfully imported from smartcash.ui.setup.env_config")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("\nCurrent working directory:", os.getcwd())
    print("Directory contents:")
    for f in os.listdir():
        print(f" - {f}")
