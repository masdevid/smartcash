"""
Script to verify imports from the smartcash package.
"""
import sys
import os

def verify_imports():
    """Verify that all required modules can be imported."""
    # Add project root to Python path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    print(f"Python path: {sys.path}")
    print("\nAttempting to import smartcash.ui.core.errors.error_component...")
    
    try:
        import smartcash.ui.core.errors.error_component
        print("✅ Successfully imported error_component")
        return True
    except ImportError as e:
        print(f"❌ Failed to import error_component: {e}")
        return False

if __name__ == "__main__":
    verify_imports()
