"""
File: smartcash/ui/setup/dependency/display_ui.py
Deskripsi: Function untuk display dependency UI dengan proper error handling
"""

from typing import Dict, Any, Optional
from IPython.display import display
import ipywidgets as widgets

def display_dependency_ui(env=None, config=None, **kwargs):
    """
    Display dependency installer UI dengan proper error handling
    
    Args:
        env: Environment info (e.g. 'colab')
        config: Configuration dict
        **kwargs: Additional parameters
        
    Returns:
        Displayed UI components dictionary
    """
    try:
        # Import dependency initializer
        from smartcash.ui.setup.dependency.dependency_init import initialize_dependency_ui
        
        # Initialize UI components
        ui_components = initialize_dependency_ui(env=env, config=config, **kwargs)
        
        # Check if error occurred
        if isinstance(ui_components, dict) and ui_components.get('error'):
            print(f"❌ Error: {ui_components.get('message', 'Unknown error')}")
            print(f"📋 Module: {ui_components.get('module', 'dependency')}")
            if ui_components.get('traceback'):
                print("🔍 Traceback:")
                print(ui_components.get('traceback'))
            
            # Try to display error UI if available
            if 'ui' in ui_components:
                display(ui_components['ui'])
            
            return ui_components
        
        # Display main UI if available
        if 'ui' in ui_components:
            display(ui_components['ui'])
            print("✅ Dependency UI berhasil ditampilkan")
        else:
            print("⚠️ Warning: UI component tidak ditemukan dalam hasil")
            print(f"📋 Available keys: {list(ui_components.keys())}")
        
        return ui_components
        
    except Exception as e:
        error_msg = str(e)
        print(f"💥 Critical error displaying dependency UI: {error_msg}")
        
        # Create minimal error display
        error_ui = widgets.VBox([
            widgets.HTML(
                value=f"""
                <div style='background: #ffebee; padding: 16px; border-radius: 8px; border-left: 4px solid #f44336;'>
                    <h3 style='color: #c62828; margin: 0 0 8px 0;'>❌ Critical UI Error</h3>
                    <p style='margin: 0;'><strong>Error:</strong> {error_msg}</p>
                </div>
                """
            )
        ])
        
        display(error_ui)
        
        return {
            'error': True,
            'error_message': error_msg,
            'ui': error_ui
        }

# Convenience function
def setup_dependency():
    """Setup dependency dengan auto-display"""
    return display_dependency_ui()