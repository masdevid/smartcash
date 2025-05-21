"""
File: smartcash/common/config/tests/test_config_manager.py
Deskripsi: Test untuk implementasi SimpleConfigManager dan penggunaan dari project root
"""

import os
import yaml
import shutil
from pathlib import Path
import tempfile

from smartcash.common.config import get_config_manager, SimpleConfigManager
from smartcash.common.config.compat import get_instance, get_module_config

def test_singleton_pattern():
    """Pengujian singleton pattern"""
    print("=== Pengujian Singleton Pattern ===")
    
    # Buat instance dengan get_config_manager
    config_manager1 = get_config_manager()
    config_manager2 = get_config_manager()
    
    # Cek apakah kedua instance adalah objek yang sama
    assert config_manager1 is config_manager2, "Singleton pattern tidak bekerja"
    
    # Buat instance dengan get_instance
    config_manager3 = get_instance()
    
    # Cek apakah semua instance adalah objek yang sama
    assert config_manager1 is config_manager3, "Singleton pattern tidak bekerja dengan get_instance"
    
    print("✅ Singleton pattern bekerja dengan baik")

def test_default_config():
    """Pengujian default config file dan base_dir"""
    print("\n=== Pengujian Default Config ===")
    
    # Buat instance dengan get_config_manager
    config_manager = get_config_manager()
    
    # Verifikasi default config file
    assert config_manager.config_file == "base_config.yaml", f"Default config file seharusnya base_config.yaml, bukan {config_manager.config_file}"
    
    # Verifikasi base_dir
    expected_base_dir = Path(__file__).resolve().parents[4]  # Adjusted for tests folder
    assert config_manager.base_dir == expected_base_dir, f"Base dir seharusnya {expected_base_dir}, bukan {config_manager.base_dir}"
    
    print(f"✅ Default config file: {config_manager.config_file}")
    print(f"✅ Base directory: {config_manager.base_dir}")

def test_compat_functions():
    """Pengujian fungsi kompatibilitas"""
    print("\n=== Pengujian Fungsi Kompatibilitas ===")
    
    # Buat config test
    test_config = {
        'test': {
            'value1': 123,
            'value2': 'abc'
        }
    }
    
    # Simpan config
    config_manager = get_config_manager()
    config_manager.save_config(test_config, "test")
    
    # Ambil dengan get_config
    config1 = config_manager.get_config("test")
    
    # Ambil dengan get_module_config (kompatibilitas)
    config2 = config_manager.get_module_config("test")
    
    # Ambil dengan fungsi global get_module_config
    config3 = get_module_config("test")
    
    # Verifikasi
    assert config1 == test_config, "get_config tidak mengembalikan config yang benar"
    assert config2 == test_config, "get_module_config tidak mengembalikan config yang benar"
    assert config3 == test_config, "fungsi global get_module_config tidak mengembalikan config yang benar"
    
    print("✅ Fungsi kompatibilitas bekerja dengan baik")

def test_drive_sync():
    """Pengujian sinkronisasi dengan Google Drive"""
    print("\n=== Pengujian Sinkronisasi Google Drive ===")
    
    try:
        # Cek apakah kita di Colab
        import google.colab
        is_colab = True
    except ImportError:
        is_colab = False
    
    if is_colab:
        # Verifikasi bahwa config_dir adalah symlink ke drive
        config_manager = get_config_manager()
        config_dir = config_manager.config_dir
        drive_config_dir = Path("/content/drive/MyDrive/SmartCash/configs")
        
        is_symlink = config_dir.is_symlink()
        if is_symlink:
            resolved_path = os.path.realpath(config_dir)
            if resolved_path == str(drive_config_dir):
                print(f"✅ {config_dir} adalah symlink ke {drive_config_dir}")
            else:
                print(f"⚠️ {config_dir} adalah symlink tetapi mengarah ke {resolved_path} (seharusnya {drive_config_dir})")
        else:
            print(f"⚠️ {config_dir} bukan symlink dari {drive_config_dir}")
        
        # Uji sinkronisasi dengan menyimpan config
        test_config = {'test_sync': {'timestamp': 'now'}}
        success = config_manager.save_config(test_config, "test_sync")
        
        if success:
            # Verifikasi file ada di drive
            drive_file = drive_config_dir / "test_sync_config.yaml"
            if drive_file.exists():
                print(f"✅ File berhasil disimpan ke {drive_file}")
            else:
                print(f"⚠️ File tidak ditemukan di {drive_file}")
        else:
            print("⚠️ Gagal menyimpan config")
    else:
        print("ℹ️ Tidak berjalan di Google Colab, melewati test sinkronisasi Google Drive")

def run_tests():
    """Jalankan semua pengujian"""
    print("🧪 Menjalankan pengujian SimpleConfigManager...\n")
    
    test_singleton_pattern()
    test_default_config()
    test_compat_functions()
    test_drive_sync()
    
    print("\n🎉 Semua pengujian berhasil!")

if __name__ == "__main__":
    run_tests() 