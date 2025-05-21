"""
File: smartcash/ui/setup/env_config/handlers/base_handler.py
Deskripsi: Handler dasar yang berisi fungsionalitas umum untuk semua handler env_config
"""

from typing import Dict, Any, Optional, Callable
import logging

from smartcash.common.logger import get_logger
from smartcash.ui.utils.ui_logger_namespace import ENV_CONFIG_LOGGER_NAMESPACE

class BaseHandler:
    """
    Handler dasar dengan fungsionalitas umum untuk semua handler environment config
    """
    
    def __init__(self, ui_callback: Optional[Dict[str, Callable]] = None, 
                logger_name: str = ENV_CONFIG_LOGGER_NAMESPACE):
        """
        Inisialisasi handler dasar
        
        Args:
            ui_callback: Dictionary callback untuk update UI
            logger_name: Nama logger untuk handler ini
        """
        self.ui_callback = ui_callback or {}
        self.logger = get_logger(logger_name)
    
    def _log_message(self, message: str, level: str = "info", icon: str = None):
        """
        Log message ke logger dan UI (jika callback tersedia)
        
        Args:
            message: Pesan yang akan dilog
            level: Level log (info, success, warning, error)
            icon: Ikon opsional untuk ditampilkan di UI
        """
        # Log ke logger object
        if level == "error":
            self.logger.error(message)
        elif level == "warning":
            self.logger.warning(message)
        elif level == "success":
            self.logger.info(f"✅ {message}")
        else:
            self.logger.info(message)
            
        # Gunakan callback jika ada
        if 'log_message' in self.ui_callback:
            self.ui_callback['log_message'](message, level, icon)
    
    def _update_status(self, message: str, status_type: str = "info"):
        """
        Update status UI jika callback tersedia
        
        Args:
            message: Pesan status
            status_type: Tipe status (info, success, warning, error)
        """
        if 'update_status' in self.ui_callback:
            self.ui_callback['update_status'](message, status_type)
    
    def _update_progress(self, value: float, message: str = ""):
        """
        Update progress UI jika callback tersedia
        
        Args:
            value: Nilai progress (0.0 - 1.0)
            message: Pesan progress opsional
        """
        if 'update_progress' in self.ui_callback:
            self.ui_callback['update_progress'](value, message) 