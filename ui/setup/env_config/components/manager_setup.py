"""
File: smartcash/ui/setup/env_config/components/manager_setup.py
Deskripsi: Setup untuk manager environment config
"""

import shutil
from pathlib import Path
from typing import Tuple, Any

from smartcash.common.config.manager import ConfigManager
from smartcash.common.config.colab_manager import ColabConfigManager
from smartcash.common.utils import is_colab

def setup_managers() -> Tuple[ConfigManager, ColabConfigManager, Path, Path]:
    """
    Setup configuration managers
    
    Returns:
        Tuple of (config_manager, colab_manager, base_dir, config_dir)
    """
    # Determine base directory
    if is_colab():
        base_dir = Path("/content")
        config_dir = base_dir / "configs"
        
        # Copy configs from smartcash/configs if not exists
        if not config_dir.exists():
            source_configs = Path("/content/smartcash/configs")
            if source_configs.exists():
                shutil.copytree(source_configs, config_dir)
    else:
        base_dir = Path.home() / "SmartCash"
        config_dir = base_dir / "configs"
    
    # Initialize managers
    config_manager = ConfigManager(
        base_dir=str(base_dir),
        config_file=str(config_dir / "base_config.yaml")
    )
    
    colab_manager = ColabConfigManager(
        base_dir=str(base_dir),
        config_file=str(config_dir / "base_config.yaml")
    )
    
    return config_manager, colab_manager, base_dir, config_dir 