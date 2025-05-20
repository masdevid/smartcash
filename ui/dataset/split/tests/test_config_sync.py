"""
File: smartcash/ui/dataset/split/tests/test_config_sync.py
Deskripsi: Test untuk memastikan sinkronisasi konfigurasi split dataset setelah save/reset
"""

import unittest
import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock
import ipywidgets as widgets

from smartcash.ui.dataset.split.split_initializer import initialize_split_ui
from smartcash.ui.dataset.split.handlers.config_handlers import load_config, save_config, get_default_split_config
from smartcash.common.config import get_config_manager


class TestConfigSync(unittest.TestCase):
    """Test untuk memastikan sinkronisasi konfigurasi split dataset."""
    
    def setUp(self):
        """Setup untuk test."""
        # Buat direktori temporary untuk test
        self.test_dir = tempfile.mkdtemp()
        
        # Mock environment manager
        self.mock_env = MagicMock()
        self.mock_env.base_dir = self.test_dir
        
        # Setup config awal
        self.config_manager = get_config_manager(base_dir=self.test_dir)
        self.initial_config = get_default_split_config()
        self.config_manager.save_module_config('split', self.initial_config)
        
        # Buat UI components mock
        self.ui_components = {
            'enabled_checkbox': widgets.Checkbox(value=True),
            'train_ratio_slider': widgets.FloatSlider(value=0.7),
            'val_ratio_slider': widgets.FloatSlider(value=0.15),
            'test_ratio_slider': widgets.FloatSlider(value=0.15),
            'random_seed_input': widgets.IntText(value=42),
            'stratify_checkbox': widgets.Checkbox(value=True),
            'save_button': widgets.Button(),
            'reset_button': widgets.Button(),
            'split_button': widgets.Button(),
            'output_log': widgets.Output(),
            'logger': MagicMock()
        }
    
    def tearDown(self):
        """Cleanup setelah test."""
        # Hapus direktori temporary
        shutil.rmtree(self.test_dir)
    
    def test_save_button_sync(self):
        """Test sinkronisasi konfigurasi setelah menekan tombol save."""
        # Ubah nilai di UI
        self.ui_components['train_ratio_slider'].value = 0.8
        self.ui_components['val_ratio_slider'].value = 0.1
        self.ui_components['test_ratio_slider'].value = 0.1
        self.ui_components['random_seed_input'].value = 100
        
        # Simulasikan klik tombol save
        from smartcash.ui.dataset.split.handlers.button_handlers import setup_button_handlers
        ui_components = setup_button_handlers(self.ui_components, self.mock_env)
        ui_components['save_button']._click_handlers(ui_components['save_button'])
        
        # Load config dan verifikasi nilai tersimpan dengan benar
        config = self.config_manager.get_module_config('split')
        self.assertEqual(config['split']['train_ratio'], 0.8)
        self.assertEqual(config['split']['val_ratio'], 0.1)
        self.assertEqual(config['split']['test_ratio'], 0.1)
        self.assertEqual(config['split']['random_seed'], 100)
        
        # Verifikasi UI tetap konsisten dengan config
        self.assertEqual(self.ui_components['train_ratio_slider'].value, 0.8)
        self.assertEqual(self.ui_components['val_ratio_slider'].value, 0.1)
        self.assertEqual(self.ui_components['test_ratio_slider'].value, 0.1)
        self.assertEqual(self.ui_components['random_seed_input'].value, 100)
    
    def test_reset_button_sync(self):
        """Test sinkronisasi konfigurasi setelah menekan tombol reset."""
        # Ubah nilai di UI dan simpan
        self.ui_components['train_ratio_slider'].value = 0.8
        self.ui_components['val_ratio_slider'].value = 0.1
        self.ui_components['test_ratio_slider'].value = 0.1
        self.ui_components['random_seed_input'].value = 100
        
        # Simpan perubahan terlebih dahulu
        from smartcash.ui.dataset.split.handlers.button_handlers import setup_button_handlers
        ui_components = setup_button_handlers(self.ui_components, self.mock_env)
        ui_components['save_button']._click_handlers(ui_components['save_button'])
        
        # Simulasikan klik tombol reset
        ui_components['reset_button']._click_handlers(ui_components['reset_button'])
        
        # Load config dan verifikasi nilai kembali ke default
        config = self.config_manager.get_module_config('split')
        default_config = get_default_split_config()
        self.assertEqual(config['split']['train_ratio'], default_config['split']['train_ratio'])
        self.assertEqual(config['split']['val_ratio'], default_config['split']['val_ratio'])
        self.assertEqual(config['split']['test_ratio'], default_config['split']['test_ratio'])
        self.assertEqual(config['split']['random_seed'], default_config['split']['random_seed'])
        
        # Verifikasi UI juga kembali ke default
        self.assertEqual(self.ui_components['train_ratio_slider'].value, default_config['split']['train_ratio'])
        self.assertEqual(self.ui_components['val_ratio_slider'].value, default_config['split']['val_ratio'])
        self.assertEqual(self.ui_components['test_ratio_slider'].value, default_config['split']['test_ratio'])
        self.assertEqual(self.ui_components['random_seed_input'].value, default_config['split']['random_seed'])
    
    def test_initialize_ui_loads_config(self):
        """Test bahwa UI diinisialisasi dengan konfigurasi yang benar."""
        # Ubah config dan simpan
        custom_config = {
            'split': {
                'enabled': False,
                'train_ratio': 0.6,
                'val_ratio': 0.2,
                'test_ratio': 0.2,
                'random_seed': 123,
                'stratify': False
            }
        }
        self.config_manager.save_module_config('split', custom_config)
        
        # Mock display untuk mencegah error saat menampilkan UI
        with patch('IPython.display.display'):
            # Inisialisasi UI dengan config yang telah diubah
            ui_components = initialize_split_ui(env=self.mock_env)
            
            # Verifikasi UI diinisialisasi dengan config yang benar
            self.assertEqual(ui_components['enabled_checkbox'].value, False)
            self.assertEqual(ui_components['train_ratio_slider'].value, 0.6)
            self.assertEqual(ui_components['val_ratio_slider'].value, 0.2)
            self.assertEqual(ui_components['test_ratio_slider'].value, 0.2)
            self.assertEqual(ui_components['random_seed_input'].value, 123)
            self.assertEqual(ui_components['stratify_checkbox'].value, False)


if __name__ == '__main__':
    unittest.main() 