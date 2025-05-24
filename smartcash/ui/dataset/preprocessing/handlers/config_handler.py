"""
File: smartcash/ui/dataset/preprocessing/handlers/config_handler.py
Deskripsi: Simplified config handler yang delegate ke SRP handlers dan environment setup
"""

from typing import Dict, Any
from smartcash.common.environment import get_environment_manager
from smartcash.common.constants.paths import get_paths_for_environment
from smartcash.ui.dataset.preprocessing.handlers.config_save_handler import setup_config_save_handler
from smartcash.ui.dataset.preprocessing.handlers.config_reset_handler import setup_config_reset_handler
from smartcash.ui.dataset.preprocessing.utils import get_config_extractor

def setup_config_handlers(ui_components: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Setup config handlers dengan environment detection dan delegation ke SRP handlers."""
    logger = ui_components.get('logger')
    env_manager = get_environment_manager()
    
    # Update paths berdasarkan environment
    paths = get_paths_for_environment(env_manager.is_colab, env_manager.is_drive_mounted)
    ui_components.update({
        'data_dir': paths['data_root'],
        'preprocessed_dir': f"{paths['data_root']}/preprocessed",
        'environment_info': {
            'is_colab': env_manager.is_colab,
            'is_drive_mounted': env_manager.is_drive_mounted,
            'paths': paths
        }
    })
    
    # Setup individual SRP handlers
    ui_components = setup_config_save_handler(ui_components)
    ui_components = setup_config_reset_handler(ui_components)
    
    # Apply initial config jika ada
    if config:
        config_extractor = get_config_extractor(ui_components)
        errors = config_extractor.apply_config_to_ui(config)
        if errors and logger:
            logger.warning(f"⚠️ Config application warnings: {', '.join(errors)}")
    
    # Log environment info
    if logger:
        logger.debug(f"✅ Config handler setup: Colab={env_manager.is_colab}, Drive={env_manager.is_drive_mounted}")
    
    return ui_components