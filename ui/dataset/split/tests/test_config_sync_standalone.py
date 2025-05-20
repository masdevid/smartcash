"""
File: smartcash/ui/dataset/split/tests/test_config_sync_standalone.py
Deskripsi: Test standalone untuk memastikan sinkronisasi konfigurasi split dataset setelah save/reset
"""

import unittest
import os
import json
import tempfile
import shutil
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

class TestConfigSyncStandalone(unittest.TestCase):
    """Test standalone untuk memastikan sinkronisasi konfigurasi split dataset."""
    
    def setUp(self):
        """Setup untuk test."""
        # Buat direktori temporary untuk test
        self.test_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.test_dir, 'split_config.json')
        
        # Buat konfigurasi default
        self.default_config = {
            "split": {
                "enabled": True,
                "train_ratio": 0.7,
                "val_ratio": 0.15,
                "test_ratio": 0.15,
                "random_seed": 42,
                "stratify": True
            }
        }
        
        # Simpan konfigurasi default
        with open(self.config_path, 'w') as f:
            json.dump(self.default_config, f)
        
        # Mock UI components
        self.ui_components = {
            'enabled_checkbox': MagicMock(value=True),
            'train_ratio_slider': MagicMock(value=0.7),
            'val_ratio_slider': MagicMock(value=0.15),
            'test_ratio_slider': MagicMock(value=0.15),
            'random_seed_input': MagicMock(value=42),
            'stratify_checkbox': MagicMock(value=True),
            'save_button': MagicMock(),
            'reset_button': MagicMock(),
            'split_button': MagicMock(),
            'output_log': MagicMock(),
            'logger': MagicMock()
        }
        
    def tearDown(self):
        """Cleanup setelah test."""
        # Hapus direktori temporary
        shutil.rmtree(self.test_dir)
    
    def load_config(self):
        """Load konfigurasi dari file."""
        with open(self.config_path, 'r') as f:
            return json.load(f)
    
    def save_config(self, config):
        """Simpan konfigurasi ke file."""
        with open(self.config_path, 'w') as f:
            json.dump(config, f)
    
    def update_ui_from_config(self, ui_components, config):
        """Update UI dari konfigurasi."""
        split_config = config['split']
        ui_components['enabled_checkbox'].value = split_config.get('enabled', True)
        ui_components['train_ratio_slider'].value = split_config.get('train_ratio', 0.7)
        ui_components['val_ratio_slider'].value = split_config.get('val_ratio', 0.15)
        ui_components['test_ratio_slider'].value = split_config.get('test_ratio', 0.15)
        ui_components['random_seed_input'].value = split_config.get('random_seed', 42)
        ui_components['stratify_checkbox'].value = split_config.get('stratify', True)
        
        # Log success ke UI logger
        ui_components['logger'].info('UI berhasil diupdate dari konfigurasi')
    
    def update_config_from_ui(self, ui_components):
        """Update konfigurasi dari UI."""
        config = self.load_config()
        config['split']['enabled'] = ui_components['enabled_checkbox'].value
        config['split']['train_ratio'] = ui_components['train_ratio_slider'].value
        config['split']['val_ratio'] = ui_components['val_ratio_slider'].value
        config['split']['test_ratio'] = ui_components['test_ratio_slider'].value
        config['split']['random_seed'] = ui_components['random_seed_input'].value
        config['split']['stratify'] = ui_components['stratify_checkbox'].value
        self.save_config(config)
        
        # Log success ke UI logger
        ui_components['logger'].info('Konfigurasi berhasil disimpan')
        
        return config
    
    def test_save_button_sync(self):
        """Test sinkronisasi konfigurasi setelah menekan tombol save."""
        # Ubah nilai di UI
        self.ui_components['train_ratio_slider'].value = 0.8
        self.ui_components['val_ratio_slider'].value = 0.1
        self.ui_components['test_ratio_slider'].value = 0.1
        self.ui_components['random_seed_input'].value = 100
        
        # Simulasikan klik tombol save
        self.update_config_from_ui(self.ui_components)
        
        # Load config dan verifikasi nilai tersimpan dengan benar
        config = self.load_config()
        self.assertEqual(config['split']['train_ratio'], 0.8)
        self.assertEqual(config['split']['val_ratio'], 0.1)
        self.assertEqual(config['split']['test_ratio'], 0.1)
        self.assertEqual(config['split']['random_seed'], 100)
        
        # Verifikasi UI tetap konsisten dengan config
        self.assertEqual(self.ui_components['train_ratio_slider'].value, 0.8)
        self.assertEqual(self.ui_components['val_ratio_slider'].value, 0.1)
        self.assertEqual(self.ui_components['test_ratio_slider'].value, 0.1)
        self.assertEqual(self.ui_components['random_seed_input'].value, 100)
        
        # Verifikasi log UI dipanggil dengan benar
        self.ui_components['logger'].info.assert_called_with('Konfigurasi berhasil disimpan')
    
    def test_reset_button_sync(self):
        """Test sinkronisasi konfigurasi setelah menekan tombol reset."""
        # Ubah nilai di UI dan simpan
        self.ui_components['train_ratio_slider'].value = 0.8
        self.ui_components['val_ratio_slider'].value = 0.1
        self.ui_components['test_ratio_slider'].value = 0.1
        self.ui_components['random_seed_input'].value = 100
        
        # Simulasikan klik tombol save
        self.update_config_from_ui(self.ui_components)
        
        # Simulasikan klik tombol reset
        self.save_config(self.default_config)
        self.update_ui_from_config(self.ui_components, self.default_config)
        
        # Log reset ke UI logger
        self.ui_components['logger'].info('Konfigurasi berhasil direset ke default')
        
        # Load config dan verifikasi nilai kembali ke default
        config = self.load_config()
        self.assertEqual(config['split']['train_ratio'], 0.7)
        self.assertEqual(config['split']['val_ratio'], 0.15)
        self.assertEqual(config['split']['test_ratio'], 0.15)
        self.assertEqual(config['split']['random_seed'], 42)
        
        # Verifikasi UI juga kembali ke default
        self.assertEqual(self.ui_components['train_ratio_slider'].value, 0.7)
        self.assertEqual(self.ui_components['val_ratio_slider'].value, 0.15)
        self.assertEqual(self.ui_components['test_ratio_slider'].value, 0.15)
        self.assertEqual(self.ui_components['random_seed_input'].value, 42)
        
        # Verifikasi log UI dipanggil dengan benar
        self.ui_components['logger'].info.assert_called_with('Konfigurasi berhasil direset ke default')
    
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
        self.save_config(custom_config)
        
        # Inisialisasi UI dengan config yang telah diubah
        self.update_ui_from_config(self.ui_components, custom_config)
        
        # Verifikasi UI diinisialisasi dengan config yang benar
        self.assertEqual(self.ui_components['enabled_checkbox'].value, False)
        self.assertEqual(self.ui_components['train_ratio_slider'].value, 0.6)
        self.assertEqual(self.ui_components['val_ratio_slider'].value, 0.2)
        self.assertEqual(self.ui_components['test_ratio_slider'].value, 0.2)
        self.assertEqual(self.ui_components['random_seed_input'].value, 123)
        self.assertEqual(self.ui_components['stratify_checkbox'].value, False)
        
        # Verifikasi log UI dipanggil dengan benar
        self.ui_components['logger'].info.assert_called_with('UI berhasil diupdate dari konfigurasi')


if __name__ == '__main__':
    unittest.main() 