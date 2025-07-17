#!/usr/bin/env python3
"""Simple test script to test individual cell files"""

import sys
import os

# Add current directory to path
sys.path.insert(0, '.')

def test_cell(cell_name, cell_path):
    """Test a single cell file"""
    print(f"\n🧪 Testing {cell_name}...")
    print("=" * 50)
    
    try:
        # Try to import and run the cell
        import importlib.util
        spec = importlib.util.spec_from_file_location(cell_name, cell_path)
        cell_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cell_module)
        print(f"✅ {cell_name} executed successfully")
        return True
    except Exception as e:
        print(f"❌ {cell_name} failed: {e}")
        return False

if __name__ == "__main__":
    # Test a few key cells
    cells = [
        ("cell_1_2_colab", "smartcash/ui/cells/cell_1_2_colab.py"),
        ("cell_1_3_dependency", "smartcash/ui/cells/cell_1_3_dependency.py"),
        ("cell_2_1_downloader", "smartcash/ui/cells/cell_2_1_downloader.py")
    ]
    
    results = []
    for cell_name, cell_path in cells:
        if os.path.exists(cell_path):
            success = test_cell(cell_name, cell_path)
            results.append((cell_name, success))
        else:
            print(f"❌ {cell_name}: File not found - {cell_path}")
            results.append((cell_name, False))
    
    print("\n" + "=" * 50)
    print("📋 Test Results:")
    success_count = 0
    for cell_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"   {cell_name}: {status}")
        if success:
            success_count += 1
    
    print(f"\n🎯 Summary: {success_count}/{len(results)} cells passed")
    
    if success_count == len(results):
        print("🎉 All cells are working correctly!")
    else:
        print("⚠️  Some cells need attention")