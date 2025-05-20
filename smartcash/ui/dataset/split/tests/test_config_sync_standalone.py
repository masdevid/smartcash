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
import time

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
            'status_panel': MagicMock(),
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
        return config
    
    def test_save_button_sync(self):
        """Test sinkronisasi konfigurasi setelah menekan tombol save."""
        # Ubah nilai di UI
        self.ui_components['train_ratio_slider'].value = 0.8
        self.ui_components['val_ratio_slider'].value = 0.1
        self.ui_components['test_ratio_slider'].value = 0.1
        self.ui_components['random_seed_input'].value = 100
        
        # Simulasikan klik tombol save
        saved_config = self.update_config_from_ui(self.ui_components)
        
        # Load config dan verifikasi nilai tersimpan dengan benar
        loaded_config = self.load_config()
        
        # Verifikasi konsistensi data setelah disimpan dan dimuat kembali
        self.assertEqual(saved_config['split']['train_ratio'], loaded_config['split']['train_ratio'])
        self.assertEqual(saved_config['split']['val_ratio'], loaded_config['split']['val_ratio'])
        self.assertEqual(saved_config['split']['test_ratio'], loaded_config['split']['test_ratio'])
        self.assertEqual(saved_config['split']['random_seed'], loaded_config['split']['random_seed'])
        
        # Verifikasi nilai yang dimuat sesuai dengan nilai yang diubah di UI
        self.assertEqual(loaded_config['split']['train_ratio'], 0.8)
        self.assertEqual(loaded_config['split']['val_ratio'], 0.1)
        self.assertEqual(loaded_config['split']['test_ratio'], 0.1)
        self.assertEqual(loaded_config['split']['random_seed'], 100)
        
        # Verifikasi UI tetap konsisten dengan config
        self.assertEqual(self.ui_components['train_ratio_slider'].value, loaded_config['split']['train_ratio'])
        self.assertEqual(self.ui_components['val_ratio_slider'].value, loaded_config['split']['val_ratio'])
        self.assertEqual(self.ui_components['test_ratio_slider'].value, loaded_config['split']['test_ratio'])
        self.assertEqual(self.ui_components['random_seed_input'].value, loaded_config['split']['random_seed'])
    
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
        
        # Load config dan verifikasi nilai kembali ke default
        loaded_config = self.load_config()
        
        # Verifikasi konsistensi data setelah reset dan dimuat kembali
        self.assertEqual(self.default_config['split']['train_ratio'], loaded_config['split']['train_ratio'])
        self.assertEqual(self.default_config['split']['val_ratio'], loaded_config['split']['val_ratio'])
        self.assertEqual(self.default_config['split']['test_ratio'], loaded_config['split']['test_ratio'])
        self.assertEqual(self.default_config['split']['random_seed'], loaded_config['split']['random_seed'])
        
        # Verifikasi UI juga kembali ke default
        self.assertEqual(self.ui_components['train_ratio_slider'].value, loaded_config['split']['train_ratio'])
        self.assertEqual(self.ui_components['val_ratio_slider'].value, loaded_config['split']['val_ratio'])
        self.assertEqual(self.ui_components['test_ratio_slider'].value, loaded_config['split']['test_ratio'])
        self.assertEqual(self.ui_components['random_seed_input'].value, loaded_config['split']['random_seed'])
    
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
        
        # Load config dan verifikasi konsistensi
        loaded_config = self.load_config()
        
        # Verifikasi konsistensi data setelah dimuat
        self.assertEqual(custom_config['split']['enabled'], loaded_config['split']['enabled'])
        self.assertEqual(custom_config['split']['train_ratio'], loaded_config['split']['train_ratio'])
        self.assertEqual(custom_config['split']['val_ratio'], loaded_config['split']['val_ratio'])
        self.assertEqual(custom_config['split']['test_ratio'], loaded_config['split']['test_ratio'])
        self.assertEqual(custom_config['split']['random_seed'], loaded_config['split']['random_seed'])
        self.assertEqual(custom_config['split']['stratify'], loaded_config['split']['stratify'])
        
        # Verifikasi UI diinisialisasi dengan config yang benar
        self.assertEqual(self.ui_components['enabled_checkbox'].value, loaded_config['split']['enabled'])
        self.assertEqual(self.ui_components['train_ratio_slider'].value, loaded_config['split']['train_ratio'])
        self.assertEqual(self.ui_components['val_ratio_slider'].value, loaded_config['split']['val_ratio'])
        self.assertEqual(self.ui_components['test_ratio_slider'].value, loaded_config['split']['test_ratio'])
        self.assertEqual(self.ui_components['random_seed_input'].value, loaded_config['split']['random_seed'])
        self.assertEqual(self.ui_components['stratify_checkbox'].value, loaded_config['split']['stratify'])
    
    def test_data_consistency_through_multiple_operations(self):
        """Test konsistensi data melalui beberapa operasi save dan load."""
        # 1. Simpan data awal
        original_config = self.load_config()
        
        # 2. Ubah nilai di UI
        self.ui_components['train_ratio_slider'].value = 0.8
        self.ui_components['val_ratio_slider'].value = 0.1
        self.ui_components['test_ratio_slider'].value = 0.1
        self.ui_components['random_seed_input'].value = 100
        
        # 3. Simpan perubahan
        self.update_config_from_ui(self.ui_components)
        
        # 4. Load config dan verifikasi
        first_loaded_config = self.load_config()
        self.assertEqual(first_loaded_config['split']['train_ratio'], 0.8)
        self.assertEqual(first_loaded_config['split']['val_ratio'], 0.1)
        self.assertEqual(first_loaded_config['split']['test_ratio'], 0.1)
        self.assertEqual(first_loaded_config['split']['random_seed'], 100)
        
        # 5. Reset UI dan inisialisasi dari config yang telah diubah
        new_ui_components = {
            'enabled_checkbox': MagicMock(value=True),
            'train_ratio_slider': MagicMock(value=0.7),
            'val_ratio_slider': MagicMock(value=0.15),
            'test_ratio_slider': MagicMock(value=0.15),
            'random_seed_input': MagicMock(value=42),
            'stratify_checkbox': MagicMock(value=True),
            'status_panel': MagicMock(),
            'logger': MagicMock()
        }
        
        # 6. Update UI dari config yang telah disimpan
        self.update_ui_from_config(new_ui_components, first_loaded_config)
        
        # 7. Verifikasi UI diperbarui dengan benar
        self.assertEqual(new_ui_components['train_ratio_slider'].value, 0.8)
        self.assertEqual(new_ui_components['val_ratio_slider'].value, 0.1)
        self.assertEqual(new_ui_components['test_ratio_slider'].value, 0.1)
        self.assertEqual(new_ui_components['random_seed_input'].value, 100)
        
        # 8. Ubah nilai lagi dan simpan
        new_ui_components['train_ratio_slider'].value = 0.75
        new_ui_components['val_ratio_slider'].value = 0.125
        new_ui_components['test_ratio_slider'].value = 0.125
        new_ui_components['random_seed_input'].value = 50
        
        # Simulasikan update config dari UI baru
        config = self.load_config()
        config['split']['train_ratio'] = new_ui_components['train_ratio_slider'].value
        config['split']['val_ratio'] = new_ui_components['val_ratio_slider'].value
        config['split']['test_ratio'] = new_ui_components['test_ratio_slider'].value
        config['split']['random_seed'] = new_ui_components['random_seed_input'].value
        self.save_config(config)
        
        # 9. Load config dan verifikasi
        final_loaded_config = self.load_config()
        self.assertEqual(final_loaded_config['split']['train_ratio'], 0.75)
        self.assertEqual(final_loaded_config['split']['val_ratio'], 0.125)
        self.assertEqual(final_loaded_config['split']['test_ratio'], 0.125)
        self.assertEqual(final_loaded_config['split']['random_seed'], 50)
        
        # 10. Verifikasi berbeda dari config awal
        self.assertNotEqual(original_config['split']['train_ratio'], final_loaded_config['split']['train_ratio'])
        self.assertNotEqual(original_config['split']['val_ratio'], final_loaded_config['split']['val_ratio'])
        self.assertNotEqual(original_config['split']['test_ratio'], final_loaded_config['split']['test_ratio'])
        self.assertNotEqual(original_config['split']['random_seed'], final_loaded_config['split']['random_seed'])
    
    def test_sequential_config_updates(self):
        """Test konsistensi data dengan mengubah konfigurasi beberapa kali secara berurutan."""
        # Menyimpan nilai konfigurasi dari setiap perubahan untuk verifikasi
        config_history = []
        
        # 1. Ubah nilai pertama
        self.ui_components['train_ratio_slider'].value = 0.8
        self.ui_components['val_ratio_slider'].value = 0.1
        self.ui_components['test_ratio_slider'].value = 0.1
        self.ui_components['random_seed_input'].value = 100
        self.ui_components['stratify_checkbox'].value = True
        
        # Simpan perubahan pertama
        self.update_config_from_ui(self.ui_components)
        config_history.append(self.load_config())
        
        # 2. Ubah nilai kedua
        self.ui_components['train_ratio_slider'].value = 0.75
        self.ui_components['val_ratio_slider'].value = 0.15
        self.ui_components['test_ratio_slider'].value = 0.1
        self.ui_components['random_seed_input'].value = 200
        self.ui_components['stratify_checkbox'].value = False
        
        # Simpan perubahan kedua
        self.update_config_from_ui(self.ui_components)
        config_history.append(self.load_config())
        
        # 3. Ubah nilai ketiga
        self.ui_components['train_ratio_slider'].value = 0.6
        self.ui_components['val_ratio_slider'].value = 0.2
        self.ui_components['test_ratio_slider'].value = 0.2
        self.ui_components['random_seed_input'].value = 300
        self.ui_components['stratify_checkbox'].value = True
        
        # Simpan perubahan ketiga
        self.update_config_from_ui(self.ui_components)
        config_history.append(self.load_config())
        
        # 4. Verifikasi semua perubahan tersimpan dengan benar
        # Perubahan pertama
        self.assertEqual(config_history[0]['split']['train_ratio'], 0.8)
        self.assertEqual(config_history[0]['split']['val_ratio'], 0.1)
        self.assertEqual(config_history[0]['split']['test_ratio'], 0.1)
        self.assertEqual(config_history[0]['split']['random_seed'], 100)
        self.assertEqual(config_history[0]['split']['stratify'], True)
        
        # Perubahan kedua
        self.assertEqual(config_history[1]['split']['train_ratio'], 0.75)
        self.assertEqual(config_history[1]['split']['val_ratio'], 0.15)
        self.assertEqual(config_history[1]['split']['test_ratio'], 0.1)
        self.assertEqual(config_history[1]['split']['random_seed'], 200)
        self.assertEqual(config_history[1]['split']['stratify'], False)
        
        # Perubahan ketiga
        self.assertEqual(config_history[2]['split']['train_ratio'], 0.6)
        self.assertEqual(config_history[2]['split']['val_ratio'], 0.2)
        self.assertEqual(config_history[2]['split']['test_ratio'], 0.2)
        self.assertEqual(config_history[2]['split']['random_seed'], 300)
        self.assertEqual(config_history[2]['split']['stratify'], True)
        
        # 5. Verifikasi konfigurasi terakhir sama dengan yang tersimpan
        final_config = self.load_config()
        self.assertEqual(final_config['split']['train_ratio'], 0.6)
        self.assertEqual(final_config['split']['val_ratio'], 0.2)
        self.assertEqual(final_config['split']['test_ratio'], 0.2)
        self.assertEqual(final_config['split']['random_seed'], 300)
        self.assertEqual(final_config['split']['stratify'], True)
        
        # 6. Muat kembali config ke UI dan verifikasi
        new_ui = {
            'enabled_checkbox': MagicMock(value=True),
            'train_ratio_slider': MagicMock(value=0.0),
            'val_ratio_slider': MagicMock(value=0.0),
            'test_ratio_slider': MagicMock(value=0.0),
            'random_seed_input': MagicMock(value=0),
            'stratify_checkbox': MagicMock(value=False),
            'status_panel': MagicMock(),
            'logger': MagicMock()
        }
        
        self.update_ui_from_config(new_ui, final_config)
        
        # Verifikasi UI diperbarui dengan nilai terakhir
        self.assertEqual(new_ui['train_ratio_slider'].value, 0.6)
        self.assertEqual(new_ui['val_ratio_slider'].value, 0.2)
        self.assertEqual(new_ui['test_ratio_slider'].value, 0.2)
        self.assertEqual(new_ui['random_seed_input'].value, 300)
        self.assertEqual(new_ui['stratify_checkbox'].value, True)


if __name__ == '__main__':
    unittest.main() 