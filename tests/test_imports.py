"""
Test file to verify module imports.
"""
import sys
import os

def test_imports():
    """Test that all required modules can be imported."""
    # Add project root to Python path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    # Test imports
    try:
        import smartcash.ui.core.errors.error_component
        print("✅ Successfully imported error_component")
        return True
    except ImportError as e:
        print(f"❌ Failed to import error_component: {e}")
        print(f"Python path: {sys.path}")
        return False

if __name__ == "__main__":
    test_imports()
