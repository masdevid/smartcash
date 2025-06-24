# File: smartcash/ui/setup/env_config/handlers/config_handler.py  
# Deskripsi: Handler untuk configuration management

import os
import shutil
from smartcash.common.config.manager import get_config_manager

class ConfigHandler:
    """📋 Handler untuk configuration setup"""
    
    def __init__(self):
        self.config_manager = get_config_manager()
    
    def setup_configurations(self, logger=None):
        """Setup file konfigurasi dengan sync manager"""
        try:
            repo_config_path = "/content/smartcash/configs"
            drive_config_path = "/content/drive/MyDrive/SmartCash/configs"
            
            if os.path.exists(repo_config_path) and os.path.exists("/content/drive/MyDrive/SmartCash"):
                if not os.path.exists(drive_config_path):
                    shutil.copytree(repo_config_path, drive_config_path)
                    if logger:
                        logger.success("📋 Konfigurasi berhasil disalin ke Drive")
                else:
                    if logger:
                        logger.info("📋 Konfigurasi sudah ada di Drive")
                
                # Verifikasi dengan config manager
                self._verify_configs_with_manager(logger)
            
        except Exception as e:
            if logger:
                logger.warning(f"⚠️ Warning setup configs: {str(e)}")
    
    def _verify_configs_with_manager(self, logger=None):
        """Verifikasi config dengan config manager"""
        try:
            # Use existing config manager sync method
            self.config_manager.sync_configs_to_drive()
            if logger:
                logger.success("✅ Config manager verified")
        except Exception as e:
            if logger:
                logger.warning(f"⚠️ Config manager sync warning: {str(e)}")
