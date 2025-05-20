"""
File: smartcash/ui/setup/env_config/handlers/directory_handler.py
Deskripsi: Handler untuk operasi directory
"""

import os
import shutil
from pathlib import Path
from typing import Dict, Any
import ipywidgets as widgets

from smartcash.ui.setup.env_config.components.state_manager import EnvConfigStateManager

def setup_directory_handler(ui_components: Dict[str, Any], colab_manager: Any) -> None:
    """
    Setup handler untuk tombol directory
    
    Args:
        ui_components: Dictionary UI components
        colab_manager: ColabConfigManager instance
    """
    # Create state manager
    state_manager = EnvConfigStateManager(ui_components, colab_manager)
    
    # Register event handler
    ui_components['directory_button'].on_click(
        lambda b: on_directory_click(b, ui_components, colab_manager, state_manager)
    )

def on_directory_click(b: widgets.Button, ui_components: Dict[str, Any], colab_manager: Any, state_manager: EnvConfigStateManager) -> None:
    """
    Handler untuk tombol directory
    
    Args:
        b: Button widget
        ui_components: Dictionary UI components
        colab_manager: ColabConfigManager instance
        state_manager: State manager instance
    """
    try:
        # Update state
        state_manager.handle_directory_setup_start()
        
        # Create directory structure
        base_dirs = [
            "/content/data",
            "/content/models",
            "/content/output",
            "/content/logs",
            "/content/exports"
        ]
        
        for dir_path in base_dirs:
            path = Path(dir_path)
            path.mkdir(exist_ok=True)
            state_manager.ui_components['log_output'].append_stdout(f"Created directory: {dir_path}\n")
        
        # Create symlinks to drive if connected
        if colab_manager.is_drive_connected():
            state_manager.update_progress(0.5)
            drive_base = Path("/content/drive/MyDrive/SmartCash")
            drive_dirs = {
                "/content/data": drive_base / "data",
                "/content/models": drive_base / "models",
                "/content/output": drive_base / "output",
                "/content/logs": drive_base / "logs",
                "/content/exports": drive_base / "exports"
            }
            
            for local_path, drive_path in drive_dirs.items():
                # Create drive directory if not exists
                drive_path.mkdir(parents=True, exist_ok=True)
                
                # Remove existing symlink if exists
                local_path = Path(local_path)
                if local_path.is_symlink():
                    local_path.unlink()
                
                # Create new symlink
                os.symlink(drive_path, local_path)
                state_manager.ui_components['log_output'].append_stdout(f"Created symlink: {local_path} -> {drive_path}\n")
        
        # Update state
        state_manager.handle_directory_setup_success()
        
    except Exception as e:
        state_manager.handle_directory_setup_error(str(e)) 