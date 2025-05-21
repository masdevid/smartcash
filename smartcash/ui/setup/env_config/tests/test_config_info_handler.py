"""
File: smartcash/ui/setup/env_config/tests/test_config_info_handler.py
Deskripsi: Unit test untuk config_info_handler.py
"""

import unittest
from unittest.mock import MagicMock, patch, call
from pathlib import Path
import os
import sys
import platform
import importlib.metadata

from smartcash.ui.setup.env_config.handlers.config_info_handler import (
    display_config_info,
    log_env_info, 
    log_system_info,
    log_colab_config,
    log_available_configs
)
from smartcash.common.config.manager import SimpleConfigManager

class TestConfigInfoHandler(unittest.TestCase):
    """
    Test untuk config_info_handler
    """
    
    def setUp(self):
        """
        Setup untuk test
        """
        # Setup mock UI components
        self.logger = MagicMock()
        self.ui_components = {
            'logger': self.logger
        }
        
        # Setup mock untuk SimpleConfigManager
        self.mock_config_manager = MagicMock(spec=SimpleConfigManager)
        self.mock_config_manager.get_config_path.return_value = Path("/mock/config/path.yaml")
        
        # Ganti get_all_config dengan get_config yang sesuai
        self.mock_config_manager.get_config.return_value = {
            "data_dir": "/mock/data",
            "model_dir": "/mock/models",
            "results_dir": "/mock/results"
        }
        
        # Setup mock paths
        self.mock_base_dir = Path("/mock/base/dir")
        self.mock_config_dir = Path("/mock/config/dir")
    
    def test_display_config_info(self):
        """
        Test display_config_info
        """
        # Mock untuk log_* functions
        with patch('smartcash.ui.setup.env_config.handlers.config_info_handler.log_env_info') as mock_log_env_info, \
            patch('smartcash.ui.setup.env_config.handlers.config_info_handler.log_system_info') as mock_log_system_info, \
            patch('smartcash.ui.setup.env_config.handlers.config_info_handler.log_colab_config') as mock_log_colab_config, \
            patch('smartcash.ui.setup.env_config.handlers.config_info_handler.log_available_configs') as mock_log_available_configs:
            
            # Call function to test
            display_config_info(self.ui_components, self.mock_config_manager, self.mock_base_dir, self.mock_config_dir)
            
            # Verify all log functions were called
            mock_log_env_info.assert_called_once_with(self.logger, self.mock_base_dir, self.mock_config_dir)
            mock_log_system_info.assert_called_once_with(self.logger)
            mock_log_colab_config.assert_called_once_with(self.logger, self.mock_config_manager)
            mock_log_available_configs.assert_called_once_with(self.logger, self.mock_config_manager)
    
    def test_log_env_info(self):
        """
        Test log_env_info
        """
        # Setting islink to False for mock_config_dir
        with patch('pathlib.Path.is_symlink', return_value=False):
            # Call function to test
            log_env_info(self.logger, self.mock_base_dir, self.mock_config_dir)
            
            # Verify logger calls
            self.logger.success.assert_called_once_with("Environment berhasil dikonfigurasi")
            self.logger.info.assert_any_call(f"📁 Base directory: {self.mock_base_dir}")
            self.logger.info.assert_any_call(f"📁 Config directory: {self.mock_config_dir}")
            
        # Setting islink to True for mock_config_dir
        with patch('pathlib.Path.is_symlink', return_value=True), \
             patch('pathlib.Path.resolve', return_value=Path("/mock/resolved/dir")), \
             patch('pathlib.Path.exists', return_value=True):
            
            # Reset mock
            self.logger.reset_mock()
            
            # Call function to test
            log_env_info(self.logger, self.mock_base_dir, self.mock_config_dir)
            
            # Verify logger calls
            self.logger.success.assert_called_once_with("Environment berhasil dikonfigurasi")
            self.logger.info.assert_any_call(f"📁 Base directory: {self.mock_base_dir}")
            self.logger.info.assert_any_call(f"📁 Config directory: {self.mock_config_dir}")
            self.logger.info.assert_any_call("🔗 Config directory adalah symlink ke: /mock/resolved/dir")
    
    @patch('smartcash.ui.setup.env_config.handlers.config_info_handler.is_colab')
    @patch('smartcash.ui.setup.env_config.handlers.config_info_handler.sys')
    def test_log_system_info(self, mock_sys, mock_is_colab):
        """
        Test log_system_info
        """
        # Setup mock
        mock_sys.version = "3.8.0 (default, Jan 1 2021, 00:00:01)"
        mock_is_colab.return_value = True
        
        # Call function to test
        log_system_info(self.logger)
        
        # Verify logger calls
        self.logger.info.assert_any_call("📊 Informasi Environment:")
        self.logger.info.assert_any_call("🐍 Python version: 3.8.0")
        self.logger.info.assert_any_call("💻 Running di Google Colab: Ya")
    
    @patch('smartcash.ui.setup.env_config.handlers.config_info_handler.is_colab')
    def test_log_colab_config_not_in_colab(self, mock_is_colab):
        """
        Test log_colab_config when not in Colab
        """
        # Setup mock
        mock_is_colab.return_value = False
        
        # Call function to test
        log_colab_config(self.logger, self.mock_config_manager)
        
        # Verify config_manager.get_config not called
        self.mock_config_manager.get_config.assert_not_called()
    
    @patch('smartcash.ui.setup.env_config.handlers.config_info_handler.is_colab')
    def test_log_colab_config_in_colab(self, mock_is_colab):
        """
        Test log_colab_config when in Colab
        """
        # Setup mock
        mock_is_colab.return_value = True
        
        colab_config = {
            'drive': {
                'use_drive': True,
                'sync_strategy': 'drive_priority',
                'symlinks': True,
                'paths': {
                    'smartcash_dir': 'SmartCash',
                    'configs_dir': 'configs'
                }
            },
            'model': {
                'use_gpu': True,
                'use_tpu': False,
                'precision': 'float16'
            },
            'performance': {
                'auto_garbage_collect': True,
                'checkpoint_to_drive': True
            }
        }
        
        self.mock_config_manager.get_config.return_value = colab_config
        
        # Call function to test
        log_colab_config(self.logger, self.mock_config_manager)
        
        # Verify config_manager.get_config called
        self.mock_config_manager.get_config.assert_called_once_with('colab')
        
        # Verify logger info calls
        self.logger.info.assert_any_call("🗄️ Pengaturan Google Drive:")
        self.logger.info.assert_any_call("- Sinkronisasi aktif: True")
        self.logger.info.assert_any_call("- Strategi sinkronisasi: drive_priority")
        self.logger.info.assert_any_call("- Gunakan symlinks: True")
        self.logger.info.assert_any_call("- SmartCash dir: SmartCash")
        self.logger.info.assert_any_call("- Configs dir: configs")
        self.logger.info.assert_any_call("⚡ Pengaturan Hardware:")
        self.logger.info.assert_any_call("- Gunakan GPU: True")
        self.logger.info.assert_any_call("- Gunakan TPU: False")
        self.logger.info.assert_any_call("- Precision: float16")
        self.logger.info.assert_any_call("🚀 Pengaturan Performa:")
        self.logger.info.assert_any_call("- Auto garbage collect: True")
        self.logger.info.assert_any_call("- Simpan checkpoint ke Drive: True")
    
    def test_log_available_configs(self):
        """
        Test log_available_configs
        """
        # Setup mock
        available_configs = ["config1.yaml", "config2.yaml", "config3.yaml"]
        self.mock_config_manager.get_available_configs.return_value = available_configs
        
        # Call function to test
        log_available_configs(self.logger, self.mock_config_manager)
        
        # Verify config_manager.get_available_configs called
        self.mock_config_manager.get_available_configs.assert_called_once()
        
        # Verify logger info calls
        self.logger.info.assert_any_call("📝 File Konfigurasi Tersedia:")
        self.logger.info.assert_any_call("- config1.yaml")
        self.logger.info.assert_any_call("- config2.yaml")
        self.logger.info.assert_any_call("- config3.yaml")
        
        # Test with empty configs
        self.logger.reset_mock()
        self.mock_config_manager.get_available_configs.return_value = []
        
        # Call function to test
        log_available_configs(self.logger, self.mock_config_manager)
        
        # Verify warning logged
        self.logger.warning.assert_called_once_with("Tidak ada file konfigurasi yang ditemukan.")

if __name__ == '__main__':
    unittest.main() 