"""
File: smartcash/ui/dataset/split/tests/test_config_sync_standalone.py
Deskripsi: Test standalone untuk verifikasi sinkronisasi konfigurasi dataset split
"""

import os
import sys
import unittest
from pathlib import Path
import tempfile
import shutil
from unittest.mock import patch, MagicMock, PropertyMock

# Tambahkan path ke smartcash
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))

from smartcash.ui.dataset.split.handlers.config_handlers import (
    get_default_split_config, is_colab_environment
)

class TestConfigSync(unittest.TestCase):
    """Test sinkronisasi konfigurasi dataset split."""

    def setUp(self):
        """Setup untuk test."""
        # Buat direktori temp untuk test
        self.temp_dir = tempfile.mkdtemp()
        
        # Buat mock UI components
        self.ui_components = {
            'logger': MagicMock(),
            'status_panel': MagicMock()
        }
        
        # Setup patch untuk get_default_base_dir
        self.base_dir_patcher = patch('smartcash.ui.dataset.split.handlers.config_handlers.get_default_base_dir')
        self.mock_base_dir = self.base_dir_patcher.start()
        self.mock_base_dir.return_value = self.temp_dir
        
        # Setup patch untuk is_colab_environment
        self.colab_patcher = patch('smartcash.ui.dataset.split.handlers.config_handlers.is_colab_environment')
        self.mock_colab = self.colab_patcher.start()
        self.mock_colab.return_value = False  # Default: bukan di Colab
        
        # Setup patch untuk config manager
        self.config_manager_patcher = patch('smartcash.ui.dataset.split.handlers.config_handlers.get_config_manager')
        self.mock_config_manager = self.config_manager_patcher.start()
        mock_manager = MagicMock()
        mock_manager.save_module_config.return_value = True
        mock_manager.get_module_config.return_value = get_default_split_config()
        mock_manager.sync_to_drive.return_value = (True, "Success")
        self.mock_config_manager.return_value = mock_manager
        
        # Pastikan direktori config ada
        os.makedirs(os.path.join(self.temp_dir, 'configs'), exist_ok=True)
        
        # Buat default config
        self.default_config = get_default_split_config()

    def tearDown(self):
        """Teardown untuk test."""
        # Stop patch
        self.base_dir_patcher.stop()
        self.colab_patcher.stop()
        self.config_manager_patcher.stop()
        
        # Hapus direktori temp
        shutil.rmtree(self.temp_dir)

    def test_save_and_load_config(self):
        """Test save dan load konfigurasi."""
        # Import fungsi yang akan diuji
        from smartcash.ui.dataset.split.handlers.config_handlers import save_config, load_config
        
        # Simpan konfigurasi
        saved_config = save_config(self.default_config, self.ui_components)
        
        # Verifikasi save_module_config dipanggil
        self.mock_config_manager.return_value.save_module_config.assert_called_with('split', self.default_config)
        
        # Load konfigurasi
        loaded_config = load_config()
        
        # Verifikasi konfigurasi berhasil disimpan dan dimuat
        self.assertEqual(saved_config, loaded_config)
        
        # Verifikasi nilai dalam konfigurasi
        self.assertTrue('split' in loaded_config)
        self.assertTrue(loaded_config['split']['enabled'])
        self.assertEqual(loaded_config['split']['train_ratio'], 0.7)
        self.assertEqual(loaded_config['split']['val_ratio'], 0.15)
        self.assertEqual(loaded_config['split']['test_ratio'], 0.15)
        self.assertEqual(loaded_config['split']['random_seed'], 42)
        self.assertTrue(loaded_config['split']['stratify'])

    def test_config_consistency_after_update(self):
        """Test konsistensi konfigurasi setelah update."""
        from smartcash.ui.dataset.split.handlers.config_handlers import save_config, load_config
        
        # Simpan konfigurasi default
        save_config(self.default_config, self.ui_components)
        
        # Buat konfigurasi yang dimodifikasi
        modified_config = get_default_split_config()
        modified_config['split']['train_ratio'] = 0.8
        modified_config['split']['val_ratio'] = 0.1
        modified_config['split']['test_ratio'] = 0.1
        modified_config['split']['random_seed'] = 123
        
        # Update mock untuk load_config
        self.mock_config_manager.return_value.get_module_config.return_value = modified_config
        
        # Simpan konfigurasi baru
        saved_config = save_config(modified_config, self.ui_components)
        
        # Verifikasi save_module_config dipanggil
        self.mock_config_manager.return_value.save_module_config.assert_called_with('split', modified_config)
        
        # Load konfigurasi
        loaded_config = load_config()
        
        # Verifikasi konfigurasi berhasil diupdate
        self.assertEqual(saved_config, loaded_config)
        self.assertEqual(loaded_config['split']['train_ratio'], 0.8)
        self.assertEqual(loaded_config['split']['val_ratio'], 0.1)
        self.assertEqual(loaded_config['split']['test_ratio'], 0.1)
        self.assertEqual(loaded_config['split']['random_seed'], 123)

    def test_sequential_config_updates(self):
        """Test update konfigurasi secara berurutan."""
        from smartcash.ui.dataset.split.handlers.config_handlers import save_config, load_config
        
        # Update 1: Ubah train_ratio
        config1 = get_default_split_config()
        config1['split']['train_ratio'] = 0.8
        
        # Set mock untuk return config1
        self.mock_config_manager.return_value.get_module_config.return_value = config1
        
        save_config(config1, self.ui_components)
        loaded1 = load_config()
        self.assertEqual(loaded1['split']['train_ratio'], 0.8)
        
        # Update 2: Ubah val_ratio
        config2 = loaded1.copy()
        config2['split']['val_ratio'] = 0.1
        
        # Set mock untuk return config2
        self.mock_config_manager.return_value.get_module_config.return_value = config2
        
        save_config(config2, self.ui_components)
        loaded2 = load_config()
        self.assertEqual(loaded2['split']['train_ratio'], 0.8)
        self.assertEqual(loaded2['split']['val_ratio'], 0.1)
        
        # Update 3: Ubah random_seed
        config3 = loaded2.copy()
        config3['split']['random_seed'] = 123
        
        # Set mock untuk return config3
        self.mock_config_manager.return_value.get_module_config.return_value = config3
        
        save_config(config3, self.ui_components)
        loaded3 = load_config()
        self.assertEqual(loaded3['split']['train_ratio'], 0.8)
        self.assertEqual(loaded3['split']['val_ratio'], 0.1)
        self.assertEqual(loaded3['split']['random_seed'], 123)
        
        # Verifikasi semua perubahan bertahan
        final_config = load_config()
        self.assertEqual(final_config['split']['train_ratio'], 0.8)
        self.assertEqual(final_config['split']['val_ratio'], 0.1)
        self.assertEqual(final_config['split']['test_ratio'], 0.15)  # Tidak diubah
        self.assertEqual(final_config['split']['random_seed'], 123)
        self.assertTrue(final_config['split']['stratify'])  # Tidak diubah

    def test_colab_sync_simulation(self):
        """Menguji simulasi sinkronisasi Colab."""
        # Set simulasi Colab
        self.mock_colab.return_value = True
        
        # Modifikasi config untuk test
        test_config = get_default_split_config()
        test_config['split']['train_ratio'] = 0.75
        
        # Import fungsi yang akan diuji
        from smartcash.ui.dataset.split.handlers.config_handlers import save_config
        
        # Patch save_config untuk menguji interaksi sinkronisasi
        with patch('smartcash.ui.dataset.split.handlers.config_handlers.sync_with_drive') as mock_sync:
            # Set mock untuk sync_with_drive
            mock_sync.return_value = test_config
            
            # Simpan konfigurasi dengan sinkronisasi
            saved_config = save_config(test_config, self.ui_components)
            
            # Verifikasi sync_with_drive dipanggil
            mock_sync.assert_called_once()
            
            # Verifikasi konfigurasi tetap konsisten
            self.assertEqual(saved_config['split']['train_ratio'], 0.75)

    def test_multiple_save_operations(self):
        """Menguji beberapa operasi save berurutan."""
        from smartcash.ui.dataset.split.handlers.config_handlers import save_config, load_config
        
        config_values = [
            {'train_ratio': 0.6, 'val_ratio': 0.2, 'test_ratio': 0.2},
            {'train_ratio': 0.7, 'val_ratio': 0.15, 'test_ratio': 0.15},
            {'train_ratio': 0.8, 'val_ratio': 0.1, 'test_ratio': 0.1}
        ]
        
        for i, values in enumerate(config_values):
            # Buat config baru
            config = get_default_split_config()
            config['split']['train_ratio'] = values['train_ratio']
            config['split']['val_ratio'] = values['val_ratio']
            config['split']['test_ratio'] = values['test_ratio']
            
            # Set mock untuk mengembalikan config ini
            self.mock_config_manager.return_value.get_module_config.return_value = config
            
            # Simpan dan verifikasi
            save_config(config, self.ui_components)
            loaded = load_config()
            
            # Verifikasi nilai tersimpan dengan benar
            self.assertEqual(loaded['split']['train_ratio'], values['train_ratio'])
            self.assertEqual(loaded['split']['val_ratio'], values['val_ratio'])
            self.assertEqual(loaded['split']['test_ratio'], values['test_ratio'])

    def test_colab_save_and_verify(self):
        """Menguji proses penyimpanan dan verifikasi di lingkungan Colab simulasi."""
        # Buat konfigurasi yang dimodifikasi
        modified_config = get_default_split_config()
        modified_config['split']['train_ratio'] = 0.65
        modified_config['split']['val_ratio'] = 0.2
        
        # Set simulasi Colab
        self.mock_colab.return_value = True
        
        # Import fungsi yang akan diuji
        from smartcash.ui.dataset.split.handlers.config_handlers import save_config, load_config
        
        # Patch sync_with_drive untuk menguji sinkronisasi
        with patch('smartcash.ui.dataset.split.handlers.config_handlers.sync_with_drive') as mock_sync:
            # Setup mock untuk sync_with_drive
            mock_sync.return_value = modified_config
            
            # Simpan konfigurasi dengan sinkronisasi
            saved_config = save_config(modified_config, self.ui_components)
            
            # Verifikasi sync_with_drive dipanggil
            mock_sync.assert_called_once()
            
            # Set mock untuk get_module_config untuk load_config
            self.mock_config_manager.return_value.get_module_config.return_value = modified_config
            
            # Load konfigurasi untuk verifikasi
            loaded_config = load_config()
            
            # Verifikasi konfigurasi hasil sinkronisasi
            self.assertEqual(saved_config['split']['train_ratio'], 0.65)
            self.assertEqual(saved_config['split']['val_ratio'], 0.2)
            self.assertEqual(loaded_config['split']['train_ratio'], 0.65)
            self.assertEqual(loaded_config['split']['val_ratio'], 0.2)

    def test_force_sync_with_drive(self):
        """Menguji sinkronisasi paksa dengan Google Drive."""
        # Setup konfigurasi
        modified_config = get_default_split_config()
        modified_config['split']['train_ratio'] = 0.85
        
        # Set simulasi Colab
        self.mock_colab.return_value = True
        
        # Buat konfigurasi awal
        config = get_default_split_config()
        config['split']['train_ratio'] = 0.75
        
        # Import fungsi yang akan diuji
        from smartcash.ui.dataset.split.handlers.config_handlers import sync_with_drive
        
        # Patch force_sync untuk test
        with patch('smartcash.common.config.force_sync.sync_with_drive') as mock_force_sync:
            # Setup mock
            mock_force_sync.return_value = modified_config
            
            # Sinkronisasi konfigurasi
            synced_config = sync_with_drive(config, self.ui_components)
            
            # Verifikasi force_sync dipanggil
            mock_force_sync.assert_called_once()
            
            # Verifikasi konfigurasi hasil sinkronisasi
            self.assertEqual(synced_config['split']['train_ratio'], 0.85)


if __name__ == '__main__':
    unittest.main() 