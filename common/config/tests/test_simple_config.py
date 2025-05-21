"""
File: smartcash/common/config/tests/test_simple_config.py
Deskripsi: Script untuk menguji implementasi SimpleConfigManager baru
"""

import os
import yaml
import shutil
from pathlib import Path
import tempfile

from smartcash.common.config.manager import SimpleConfigManager, get_config_manager

def setup_test_environment():
    """Setup lingkungan untuk pengujian dengan direktori sementara"""
    # Buat direktori temp
    temp_dir = Path(tempfile.mkdtemp())
    # Buat direktori configs di dalam temp_dir
    config_dir = temp_dir / 'configs'
    config_dir.mkdir(parents=True, exist_ok=True)
    
    # Buat config test
    test_config = {
        'app_name': 'SmartCash',
        'version': '1.0.0',
        'settings': {
            'debug': True,
            'log_level': 'INFO'
        }
    }
    
    # Simpan ke file
    test_config_path = config_dir / 'test_config.yaml'
    with open(test_config_path, 'w') as f:
        yaml.dump(test_config, f)
    
    return temp_dir, test_config

def cleanup_test_environment(temp_dir):
    """Bersihkan lingkungan pengujian"""
    if temp_dir.exists():
        shutil.rmtree(temp_dir)

def test_config_manager_init():
    """Pengujian inisialisasi SimpleConfigManager"""
    print("=== Pengujian Inisialisasi SimpleConfigManager ===")
    
    temp_dir, _ = setup_test_environment()
    try:
        # Buat instance config manager
        config_manager = SimpleConfigManager(base_dir=temp_dir)
        
        # Cek apakah direktori config ada
        assert config_manager.config_dir.exists(), "Direktori config tidak ada"
        
        print("✅ Inisialisasi SimpleConfigManager berhasil")
    finally:
        cleanup_test_environment(temp_dir)

def test_config_loading():
    """Pengujian loading konfigurasi"""
    print("\n=== Pengujian Loading Konfigurasi ===")
    
    temp_dir, test_config = setup_test_environment()
    try:
        # Buat instance config manager
        config_manager = SimpleConfigManager(base_dir=temp_dir)
        
        # Load konfigurasi
        config = config_manager.load_config('test_config')
        
        # Cek apakah konfigurasi berhasil dimuat
        if config:
            print(f"Loaded config: {config}")
            print(f"Expected config: {test_config}")
            
            # Cek cache
            if 'test_config' in config_manager.config_cache:
                print("✅ Konfigurasi di-cache")
            else:
                print("❌ Konfigurasi tidak di-cache")
                
            print("✅ Loading konfigurasi berhasil")
        else:
            print("❌ Konfigurasi tidak dapat dimuat")
    finally:
        cleanup_test_environment(temp_dir)

def test_config_saving():
    """Pengujian penyimpanan konfigurasi"""
    print("\n=== Pengujian Penyimpanan Konfigurasi ===")
    
    temp_dir, _ = setup_test_environment()
    try:
        # Buat instance config manager
        config_manager = SimpleConfigManager(base_dir=temp_dir)
        
        # Buat konfigurasi baru
        new_config = {
            'app_name': 'SmartCash',
            'version': '1.1.0',
            'settings': {
                'debug': False,
                'log_level': 'ERROR'
            }
        }
        
        # Simpan konfigurasi
        success = config_manager.save_config(new_config, 'new_config')
        assert success, "Penyimpanan konfigurasi gagal"
        
        # Cek apakah file ada
        config_path = config_manager.get_config_path('new_config')
        assert config_path.exists(), "File konfigurasi tidak dibuat"
        
        # Load dan verifikasi
        loaded_config = config_manager.load_config('new_config')
        assert loaded_config == new_config, "Konfigurasi yang dimuat tidak sesuai dengan yang disimpan"
        
        print("✅ Penyimpanan konfigurasi berhasil")
    finally:
        cleanup_test_environment(temp_dir)

def test_config_update():
    """Pengujian update konfigurasi"""
    print("\n=== Pengujian Update Konfigurasi ===")
    
    temp_dir, test_config = setup_test_environment()
    try:
        # Buat instance config manager
        config_manager = SimpleConfigManager(base_dir=temp_dir)
        
        # Update konfigurasi
        update_dict = {
            'version': '1.0.1',
            'settings': {
                'log_level': 'DEBUG'
            }
        }
        
        success = config_manager.update_config(update_dict, 'test_config')
        assert success, "Update konfigurasi gagal"
        
        # Load dan verifikasi
        updated_config = config_manager.load_config('test_config')
        assert updated_config['version'] == '1.0.1', "Version tidak diupdate"
        assert updated_config['settings']['log_level'] == 'DEBUG', "Log level tidak diupdate"
        assert updated_config['settings']['debug'] == True, "Setting lainnya hilang"
        
        print("✅ Update konfigurasi berhasil")
    finally:
        cleanup_test_environment(temp_dir)

def test_observer_pattern():
    """Pengujian observer pattern"""
    print("\n=== Pengujian Observer Pattern ===")
    
    temp_dir, test_config = setup_test_environment()
    try:
        # Buat instance config manager
        config_manager = SimpleConfigManager(base_dir=temp_dir)
        
        # Variabel untuk menyimpan notifikasi
        notification_received = False
        notified_config = None
        
        # Observer function
        def config_observer(config):
            nonlocal notification_received, notified_config
            notification_received = True
            notified_config = config
        
        # Register observer
        config_manager.register_observer('test_config', config_observer)
        
        # Update konfigurasi untuk memicu notifikasi
        update_dict = {'version': '1.0.2'}
        config_manager.update_config(update_dict, 'test_config')
        
        # Cek apakah notifikasi diterima
        assert notification_received, "Notifikasi observer tidak diterima"
        assert notified_config['version'] == '1.0.2', "Konfigurasi yang diterima tidak sesuai"
        
        # Unregister observer
        config_manager.unregister_observer('test_config', config_observer)
        
        # Reset flag
        notification_received = False
        
        # Update lagi
        update_dict = {'version': '1.0.3'}
        config_manager.update_config(update_dict, 'test_config')
        
        # Cek apakah notifikasi TIDAK diterima
        assert not notification_received, "Notifikasi masih diterima setelah unregister"
        
        print("✅ Observer pattern bekerja dengan baik")
    finally:
        cleanup_test_environment(temp_dir)

def test_singleton_pattern():
    """Pengujian singleton pattern"""
    print("\n=== Pengujian Singleton Pattern ===")
    
    temp_dir, _ = setup_test_environment()
    try:
        # Buat instance dengan get_config_manager
        config_manager1 = get_config_manager(base_dir=temp_dir)
        config_manager2 = get_config_manager()
        
        # Cek apakah kedua instance adalah objek yang sama
        assert config_manager1 is config_manager2, "Singleton pattern tidak bekerja"
        
        print("✅ Singleton pattern bekerja dengan baik")
    finally:
        cleanup_test_environment(temp_dir)

def run_tests():
    """Jalankan semua pengujian"""
    print("Menjalankan pengujian SimpleConfigManager...")
    
    test_config_manager_init()
    test_config_loading()
    test_config_saving()
    test_config_update()
    test_observer_pattern()
    test_singleton_pattern()
    
    print("\n🎉 Semua pengujian berhasil!")

if __name__ == "__main__":
    run_tests() 