"""
File: smartcash/ui/setup/env_config/handlers/setup_handlers.py
Deskripsi: Main handler yang menghubungkan semua komponen dengan benar
"""

from typing import Dict, Any
from smartcash.ui.setup.env_config.handlers.setup_handler import SetupHandler
from smartcash.ui.setup.env_config.handlers.ui_update_handler import UIUpdateHandler

def setup_env_config_handlers(ui_components: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """🔧 Setup semua handlers untuk environment configuration dengan integrasi yang benar"""
    
    # 1. Setup UI update handler (untuk status refresh)
    ui_update_handler = UIUpdateHandler(ui_components)
    ui_components['ui_update_handler'] = ui_update_handler
    
    # 2. Setup main setup handler 
    setup_handler = SetupHandler()
    ui_components['setup_handler'] = setup_handler
    
    # 3. Setup button click handler
    def on_setup_click(button):
        """Handler untuk setup button click"""
        try:
            # Notify setup start
            ui_update_handler.on_setup_start()
            
            # Disable button during setup
            button.disabled = True
            button.description = "⚙️ Setting up..."
            
            # Execute setup
            success = setup_handler.execute_setup(ui_components)
            
            # Notify completion
            ui_update_handler.on_setup_complete(success)
            
        except Exception as e:
            ui_update_handler.logger.error(f"❌ Setup failed: {str(e)}")
            ui_update_handler.on_setup_complete(False)
        finally:
            # Re-enable button
            button.disabled = False
            button.description = "🚀 Setup Environment"
    
    # 4. Attach button handler
    if 'setup_button' in ui_components:
        setup_button = ui_components['setup_button']
        setup_button.on_click(on_setup_click)
    
    # 5. Initial status check
    ui_update_handler.refresh_environment_status()
    
    return ui_components