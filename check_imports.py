#!/usr/bin/env python3
"""
Script to verify package imports and paths.
"""
import sys
import os

def main():
    print("Python sys.path:")
    for p in sys.path:
        print(f"  - {p}")
    
    print("\nTrying to import smartcash.ui.setup...")
    try:
        import smartcash.ui.setup as setup
        print("✅ Successfully imported smartcash.ui.setup")
        print(f"  - Location: {setup.__file__}")
        
        print("\nTrying to import dependency_initializer...")
        try:
            from smartcash.ui.setup.dependency import dependency_initializer
            print("✅ Successfully imported dependency_initializer")
            print(f"  - Location: {dependency_initializer.__file__}")
        except ImportError as e:
            print(f"❌ Failed to import dependency_initializer: {e}")
            print("  - Checking if directory exists:")
            setup_dir = os.path.join(os.path.dirname(setup.__file__), 'dependency')
            print(f"    - {setup_dir} exists: {os.path.exists(setup_dir)}")
            if os.path.exists(setup_dir):
                print("    - Contents:")
                for f in os.listdir(setup_dir):
                    print(f"      - {f}")
    except ImportError as e:
        print(f"❌ Failed to import smartcash.ui.setup: {e}")
        print("  - Checking if package is installed:")
        try:
            import pkg_resources
            dist = pkg_resources.get_distribution('smartcash')
            print(f"    - Package found: {dist.project_name} {dist.version}")
            print(f"    - Location: {dist.location}")
        except Exception as e:
            print(f"    - Could not find package info: {e}")

if __name__ == "__main__":
    main()
