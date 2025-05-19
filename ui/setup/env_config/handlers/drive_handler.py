"""
File: smartcash/ui/setup/env_config/handlers/drive_handler.py
Deskripsi: Handler untuk operasi Google Drive
"""

import os
import shutil
from pathlib import Path
from typing import Dict, Any
import ipywidgets as widgets

from smartcash.ui.setup.env_config.components.state_manager import EnvConfigStateManager

def setup_drive_handler(ui_components: Dict[str, Any], colab_manager: Any) -> None:
    """
    Setup handler untuk tombol drive
    
    Args:
        ui_components: Dictionary UI components
        colab_manager: ColabConfigManager instance
    """
    # Create state manager
    state_manager = EnvConfigStateManager(ui_components, colab_manager)
    
    # Register event handler
    ui_components['drive_button'].on_click(
        lambda b: on_drive_click(b, ui_components, colab_manager, state_manager)
    )

async def on_drive_click(b: widgets.Button, ui_components: Dict[str, Any], colab_manager: Any, state_manager: EnvConfigStateManager) -> None:
    """
    Handler untuk tombol drive
    
    Args:
        b: Button widget
        ui_components: Dictionary UI components
        colab_manager: ColabConfigManager instance
        state_manager: State manager instance
    """
    try:
        # Update state
        state_manager.handle_drive_connection_start()
        
        # Connect to drive
        await colab_manager.connect_to_drive()
        
        # Update progress
        state_manager.tracker.update(0.5, "Setting up drive configs...")
        
        # Replace configs with drive configs
        drive_configs = Path("/content/drive/MyDrive/SmartCash/configs")
        if drive_configs.exists():
            # Backup current configs
            backup_dir = Path("/content/configs_backup")
            if Path("/content/configs").exists():
                shutil.move("/content/configs", backup_dir)
            
            # Create symlink to drive configs
            os.symlink(drive_configs, "/content/configs")
        else:
            # Create drive configs directory
            drive_configs.mkdir(parents=True)
            
            # Copy local configs to drive
            if Path("/content/configs").exists():
                shutil.copytree("/content/configs", drive_configs, dirs_exist_ok=True)
        
        # Update state
        state_manager.handle_drive_connection_success()
        
    except Exception as e:
        state_manager.handle_drive_connection_error(str(e)) 