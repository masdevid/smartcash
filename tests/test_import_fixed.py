"""
Test script to verify that the package can be imported correctly.
"""
import sys
import os

def main():
    # Add project root to Python path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    print(f"Python path: {sys.path}")
    
    # Try importing the package
    try:
        import smartcash
        print(f"✅ Successfully imported smartcash from {smartcash.__file__}")
        
        # Try importing the UI module
        try:
            import smartcash.ui
            print(f"✅ Successfully imported smartcash.ui from {smartcash.ui.__file__}")
            
            # Try importing the core module
            try:
                import smartcash.ui.core
                print(f"✅ Successfully imported smartcash.ui.core from {smartcash.ui.core.__file__}")
                
                # Try importing the errors module
                try:
                    import smartcash.ui.core.errors
                    print(f"✅ Successfully imported smartcash.ui.core.errors from {smartcash.ui.core.errors.__file__}")
                    
                    # Try importing the error component
                    try:
                        from smartcash.ui.core.errors import error_component
                        print(f"✅ Successfully imported error_component from {error_component.__file__}")
                        return True
                    except ImportError as e:
                        print(f"❌ Failed to import error_component: {e}")
                        return False
                        
                except ImportError as e:
                    print(f"❌ Failed to import smartcash.ui.core.errors: {e}")
                    return False
                    
            except ImportError as e:
                print(f"❌ Failed to import smartcash.ui.core: {e}")
                return False
                
        except ImportError as e:
            print(f"❌ Failed to import smartcash.ui: {e}")
            return False
            
    except ImportError as e:
        print(f"❌ Failed to import smartcash: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 All imports successful!")
    else:
        print("\n❌ Some imports failed. Check the output above for details.")
