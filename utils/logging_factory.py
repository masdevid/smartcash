# File: smartcash/utils/logging_factory.py
# Author: Alfrida Sabar
# Deskripsi: Factory untuk pembuatan dan konfigurasi sistem logging terpusat

import logging
import os
from typing import Dict, Optional, List, Union, Any
from pathlib import Path

from smartcash.utils.logger import SmartCashLogger, get_logger

class LoggingFactory:
    """Factory untuk konfigurasi dan pembuatan logger secara terpusat."""
    
    # Instance cache untuk reuse logger
    _loggers: Dict[str, SmartCashLogger] = {}
    
    # Konfigurasi default
    DEFAULT_CONFIG = {
        'logs_dir': 'logs',
        'log_level': 'INFO',
        'log_to_file': True,
        'log_to_console': True,
        'use_colors': True,
        'use_emojis': True,
        'rotate_logs': True,
        'backup_count': 30
    }
    
    @classmethod
    def configure(cls, config: Dict[str, Any]) -> None:
        """
        Konfigurasi global untuk semua logger yang akan dibuat.
        
        Args:
            config: Dictionary konfigurasi logger
        """
        # Update default config
        cls.DEFAULT_CONFIG.update(config)
        
        # Siapkan direktori logs
        logs_dir = Path(cls.DEFAULT_CONFIG['logs_dir'])
        logs_dir.mkdir(exist_ok=True)
        
        # Set level logging global
        level_name = cls.DEFAULT_CONFIG['log_level'].upper()
        level = getattr(logging, level_name, logging.INFO)
        logging.basicConfig(level=level)
        
    @classmethod
    def get_logger(
        cls, 
        name: str, 
        config: Optional[Dict[str, Any]] = None
    ) -> SmartCashLogger:
        """
        Dapatkan logger dengan nama dan konfigurasi tertentu.
        Menggunakan cache untuk reuse logger yang sama.
        
        Args:
            name: Nama logger
            config: Konfigurasi kustom (opsional)
            
        Returns:
            Instance SmartCashLogger
        """
        # Reuse logger yang sudah ada jika tidak ada konfigurasi kustom
        if name in cls._loggers and config is None:
            return cls._loggers[name]
        
        # Gunakan konfigurasi default dengan override dari konfigurasi kustom
        final_config = cls.DEFAULT_CONFIG.copy()
        if config:
            final_config.update(config)
        
        # Buat logger menggunakan get_logger
        logger = get_logger(name)
        
        # Konfigurasi tambahan jika dibutuhkan
        if final_config.get('log_to_file', True):
            logs_dir = Path(final_config['logs_dir'])
            logs_dir.mkdir(exist_ok=True)
            
            # Log handler sudah diurus oleh get_logger
            
        # Simpan di cache
        cls._loggers[name] = logger
        
        return logger
    
    @classmethod
    def reset_loggers(cls) -> None:
        """Reset semua logger yang disimpan dalam cache."""
        cls._loggers = {}
        
    @classmethod
    def get_all_loggers(cls) -> Dict[str, SmartCashLogger]:
        """
        Dapatkan semua logger yang telah dibuat.
        
        Returns:
            Dictionary dari nama logger ke instance SmartCashLogger
        """
        return cls._loggers.copy()