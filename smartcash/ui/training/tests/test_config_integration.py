"""
File: smartcash/ui/training/tests/test_config_integration.py
Deskripsi: Pengujian integrasi konfigurasi antara modul training dan modul lainnya
"""

import sys
import os
from typing import Dict, Any
from unittest.mock import MagicMock, patch, call
import ipywidgets as widgets
import traceback

# Import modul yang akan diuji
from smartcash.ui.training.training_init import TrainingInitializer, get_training_initializer
from smartcash.ui.training.components.config_tabs import update_config_tabs
from smartcash.common.config.manager import SimpleConfigManager, get_config_manager


class ConfigUpdateTester:
    """
    Tester untuk memastikan perubahan konfigurasi pada modul lain 
    tercermin dalam informasi training UI
    """
    
    def __init__(self):
        self.results = {
            'success': [],
            'failure': []
        }
        
        # Mock objects
        self.mocks = {}
        
    def setup_mocks(self):
        """Setup mocks untuk pengujian"""
        print("🔧 Menyiapkan mock objects...")
        
        # Mock SimpleConfigManager
        self.mocks['config_manager'] = MagicMock()
        self.mocks['config_manager'].get_config.return_value = {}
        
        # Mock ipywidgets
        self.mocks['HTML'] = MagicMock()
        self.mocks['VBox'] = MagicMock()
        self.mocks['HBox'] = MagicMock()
        self.mocks['Output'] = MagicMock()
        self.mocks['Tab'] = MagicMock()
        
        # Mock logger
        self.mocks['logger'] = MagicMock()
        
        # Patch functions
        self.patches = [
            patch('smartcash.common.config.manager.get_config_manager', 
                  return_value=self.mocks['config_manager']),
            patch('ipywidgets.HTML', return_value=self.mocks['HTML']),
            patch('ipywidgets.VBox', return_value=self.mocks['VBox']),
            patch('ipywidgets.HBox', return_value=self.mocks['HBox']),
            patch('ipywidgets.Output', return_value=self.mocks['Output']),
            patch('ipywidgets.Tab', return_value=self.mocks['Tab'])
        ]
        
        # Start all patches
        for p in self.patches:
            p.start()
            
        print("✅ Mock objects berhasil disiapkan")
        
    def teardown_mocks(self):
        """Cleanup mock objects"""
        print("🧹 Membersihkan mock objects...")
        
        for p in self.patches:
            p.stop()
            
        print("✅ Mock objects berhasil dibersihkan")

    def setup_ui_components(self):
        """Setup komponen UI dasar untuk pengujian"""
        print("🔧 Menyiapkan komponen UI training...")
        
        # Buat config dasar
        self.test_config = {
            'model': {
                'name': 'efficientnet-b4',
                'backbone': 'efficientnet-b4',
                'detection_layers': ['banknote']
            },
            'training': {
                'batch_size': 16,
                'epochs': 100,
                'learning_rate': 0.001
            }
        }
        
        # Buat config tabs
        config_tabs = create_config_tabs(self.test_config)
        
        # Mocked tab widget
        self.mocks['Tab'].children = [self.mocks['HTML']] * 4
        config_tabs = self.mocks['Tab']
        
        # Setup UI components
        self.ui_components = {
            'config_tabs': config_tabs,
            'info_display': config_tabs,
            'main_container': self.mocks['VBox'],
            'start_button': MagicMock(),
            'stop_button': MagicMock(),
            'status_panel': self.mocks['HTML'],
            'logger': self.mocks['logger']
        }
        
        print("✅ Komponen UI training berhasil disiapkan")
        return self.ui_components

    def test_config_callback_registration(self):
        """Test registrasi callback konfigurasi"""
        print("\n🧪 Pengujian registrasi callback konfigurasi...")
        
        try:
            # Setup initializer
            initializer = TrainingInitializer('training', 'smartcash.ui.training')
            
            # Setup config callback
            initializer._setup_config_callback(self.ui_components)
            
            # Verifikasi callback tersimpan
            if hasattr(initializer, 'config_update_callbacks') and len(initializer.config_update_callbacks) > 0:
                self.results['success'].append("✅ Callback konfigurasi berhasil terdaftar")
            else:
                self.results['failure'].append("❌ Callback konfigurasi tidak terdaftar")
                
            # Verifikasi callback tersimpan di UI components
            if 'config_update_callback' in self.ui_components:
                self.results['success'].append("✅ Callback konfigurasi tersedia di UI components")
            else:
                self.results['failure'].append("❌ Callback konfigurasi tidak tersedia di UI components")
        
        except Exception as e:
            self.results['failure'].append(f"❌ Error saat registrasi callback: {str(e)}")
        
        print("✅ Pengujian registrasi callback selesai")
        
    def test_config_update_propagation(self):
        """Test propagasi perubahan konfigurasi"""
        print("\n🧪 Pengujian propagasi perubahan konfigurasi...")
        
        try:
            # Setup initializer
            initializer = TrainingInitializer('training', 'smartcash.ui.training')
            
            # Setup config callback
            initializer._setup_config_callback(self.ui_components)
            
            # Verifikasi callback ada
            if not hasattr(initializer, 'config_update_callbacks') or len(initializer.config_update_callbacks) == 0:
                self.results['failure'].append("❌ Callback konfigurasi tidak terdaftar")
                return
                
            # Set config baru dari modul lain
            new_model_config = {
                'model': {
                    'name': 'yolov5-modified',
                    'backbone': 'efficientnet-b5',
                    'detection_layers': ['banknote', 'serial_number'],
                    'num_classes': 10
                }
            }
            
            # Trigger update dari modul lain
            callback = initializer.config_update_callbacks[0]
            callback(new_model_config)
            
            # Verifikasi config_tabs diupdate
            update_config_tabs_mock = patch('smartcash.ui.training.components.config_tabs.update_config_tabs')
            with update_config_tabs_mock as mock:
                # Trigger update lagi untuk memeriksa apakah update_config_tabs dipanggil
                callback(new_model_config)
                if mock.called:
                    self.results['success'].append("✅ Update config tabs dipanggil saat config berubah")
                else:
                    self.results['failure'].append("❌ Update config tabs tidak dipanggil saat config berubah")
            
            # Verifikasi logger dipanggil
            if self.mocks['logger'].info.called:
                self.results['success'].append("✅ Logger info dipanggil saat config berubah")
            else:
                self.results['failure'].append("❌ Logger info tidak dipanggil saat config berubah")
        
        except Exception as e:
            self.results['failure'].append(f"❌ Error saat pengujian propagasi: {str(e)}")
        
        print("✅ Pengujian propagasi perubahan konfigurasi selesai")
        
    def test_multiple_module_config_merging(self):
        """Test penggabungan konfigurasi dari beberapa modul"""
        print("\n🧪 Pengujian penggabungan konfigurasi dari beberapa modul...")
        
        try:
            # Setup initializer
            initializer = TrainingInitializer('training', 'smartcash.ui.training')
            
            # Setup config callback
            initializer._setup_config_callback(self.ui_components)
            
            # Setup mock config untuk beberapa modul
            backbone_config = {
                'name': 'efficient-optimized',
                'backbone': 'efficientnet-b4',
                'channels': 1280
            }
            
            training_config = {
                'batch_size': 32,
                'epochs': 150,
                'learning_rate': 0.0005
            }
            
            detector_config = {
                'detection_layers': ['banknote', 'serial_number', 'security_feature'],
                'anchors': '4,5, 6,7, 8,9'
            }
            
            # Simulasi config dari berbagai modul
            self.mocks['config_manager'].get_config.side_effect = lambda module: {
                'backbone': backbone_config,
                'training': training_config,
                'detector': detector_config
            }.get(module, {})
            
            # Trigger update dengan config baru
            data_config = {
                'data': {
                    'train_path': '/path/to/train',
                    'val_path': '/path/to/val',
                    'classes': ['Rp1000', 'Rp2000', 'Rp5000', 'Rp10000', 'Rp20000', 'Rp50000', 'Rp100000']
                }
            }
            
            # Ambil callback dan eksekusi
            callback = initializer.config_update_callbacks[0]
            
            # Mock update_config_tabs untuk memeriksa merged config
            merged_config_captured = {}
            
            def capture_merged_config(tabs, config):
                nonlocal merged_config_captured
                merged_config_captured = config
                
            with patch('smartcash.ui.training.components.config_tabs.update_config_tabs', 
                       side_effect=capture_merged_config):
                callback(data_config)
                
                # Verifikasi config dari berbagai modul digabung
                keys_expected = ['model', 'training', 'data']
                missing_keys = [k for k in keys_expected if k not in merged_config_captured]
                
                if not missing_keys:
                    self.results['success'].append("✅ Config dari berbagai modul berhasil digabung")
                else:
                    self.results['failure'].append(f"❌ Config tidak lengkap, missing keys: {missing_keys}")
                
                # Verifikasi data config masuk ke merged config
                if 'data' in merged_config_captured and 'classes' in merged_config_captured.get('data', {}):
                    self.results['success'].append("✅ Config data berhasil ditambahkan ke merged config")
                else:
                    self.results['failure'].append("❌ Config data tidak berhasil ditambahkan")
        
        except Exception as e:
            self.results['failure'].append(f"❌ Error saat pengujian penggabungan: {str(e)}")
        
        print("✅ Pengujian penggabungan konfigurasi selesai")
        
    def test_ui_update_on_config_change(self):
        """Test pembaruan UI saat konfigurasi berubah"""
        print("\n🧪 Pengujian pembaruan UI saat konfigurasi berubah...")
        
        try:
            # Setup initializer dengan mock
            initializer = TrainingInitializer('training', 'smartcash.ui.training')
            
            # Setup UI dengan tab yang dapat dimonitor
            self.ui_components['config_tabs'] = MagicMock()
            self.ui_components['config_tabs'].children = [MagicMock(), MagicMock()]
            
            # Setup config callback
            initializer._setup_config_callback(self.ui_components)
            
            # Simulasi perubahan config dari modul lain
            new_config = {
                'training': {
                    'batch_size': 64,
                    'epochs': 200,
                    'learning_rate': 0.0002,
                    'scheduler': 'cosine'
                }
            }
            
            # Trigger update
            with patch('smartcash.ui.training.components.config_tabs.update_config_tabs') as mock_update:
                callback = initializer.config_update_callbacks[0]
                callback(new_config)
                
                if mock_update.called:
                    call_args = mock_update.call_args
                    if call_args and len(call_args[0]) >= 2:
                        tabs_arg, config_arg = call_args[0][0], call_args[0][1]
                        
                        # Verifikasi tabs yang dikirim ke update adalah tabs dari UI components
                        if tabs_arg == self.ui_components['config_tabs']:
                            self.results['success'].append("✅ UI tabs diperbarui dengan benar")
                        else:
                            self.results['failure'].append("❌ UI tabs tidak diperbarui dengan benar")
                            
                        # Verifikasi config baru masuk ke update
                        if 'training' in config_arg and config_arg['training'].get('scheduler') == 'cosine':
                            self.results['success'].append("✅ Config baru berhasil dimasukkan ke update")
                        else:
                            self.results['failure'].append("❌ Config baru tidak masuk ke update")
                else:
                    self.results['failure'].append("❌ Update UI tidak dipanggil saat config berubah")
        
        except Exception as e:
            self.results['failure'].append(f"❌ Error saat pengujian UI update: {str(e)}")
        
        print("✅ Pengujian pembaruan UI selesai")
        
    def run_all_tests(self):
        """Jalankan semua pengujian"""
        print("\n🚀 Memulai rangkaian pengujian integrasi konfigurasi...")
        
        try:
            # Setup
            self.setup_mocks()
            self.setup_ui_components()
            
            # Run tests
            self.test_config_callback_registration()
            self.test_config_update_propagation()
            self.test_multiple_module_config_merging()
            self.test_ui_update_on_config_change()
            
        finally:
            # Cleanup
            self.teardown_mocks()
        
        # Tampilkan hasil
        self.display_results()
        
    def display_results(self):
        """Tampilkan hasil pengujian"""
        print("\n" + "="*70)
        print("📊 HASIL PENGUJIAN INTEGRASI KONFIGURASI")
        print("="*70)
        
        print("\n✅ SUKSES:")
        for result in self.results['success']:
            print(f"  {result}")
            
        if self.results['failure']:
            print("\n❌ GAGAL:")
            for result in self.results['failure']:
                print(f"  {result}")
        
        print("\n" + "="*70)
        total = len(self.results['success']) + len(self.results['failure'])
        success_rate = (len(self.results['success']) / total * 100) if total > 0 else 0
        print(f"Total: {total} tes")
        print(f"Berhasil: {len(self.results['success'])} ({success_rate:.1f}%)")
        print(f"Gagal: {len(self.results['failure'])}")
        print("="*70)
        
        # Return result status for integration
        return len(self.results['failure']) == 0


def run_config_integration_test():
    """Jalankan pengujian integrasi konfigurasi"""
    tester = ConfigUpdateTester()
    success = tester.run_all_tests()
    
    return {
        'success': success,
        'results': tester.results
    }


if __name__ == "__main__":
    run_config_integration_test()
