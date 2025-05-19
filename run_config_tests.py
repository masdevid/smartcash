"""
File: run_config_tests.py
Deskripsi: Script untuk menjalankan test sinkronisasi konfigurasi dan environment config
"""

import os
import sys
from pathlib import Path

def setup_paths():
    """Setup path untuk import smartcash modules dan pastikan direktori konfigurasi ada"""
    # Tambahkan direktori root ke sys.path
    root_dir = Path(__file__).parent.absolute()
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))
    
    # Pastikan direktori konfigurasi ada
    configs_dir = root_dir / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    
    # Buat beberapa file konfigurasi contoh jika belum ada
    example_configs = {
        "environment_config.yaml": {
            "base_dir": str(root_dir),
            "platform": "test",
            "python_version": "3.x",
            "drive_mounted": False
        },
        "model_config.yaml": {
            "model_name": "yolov5",
            "backbone": "efficientnet-b4",
            "input_size": 640,
            "batch_size": 16
        },
        "dataset_config.yaml": {
            "name": "currency_dataset",
            "classes": ["1000", "2000", "5000", "10000", "20000", "50000", "100000"],
            "train_split": 0.7,
            "val_split": 0.15,
            "test_split": 0.15
        }
    }
    
    # Import fungsi untuk menyimpan konfigurasi
    try:
        from smartcash.common.io import save_config
        
        # Buat file konfigurasi contoh
        for config_name, config_data in example_configs.items():
            config_path = configs_dir / config_name
            if not config_path.exists():
                print(f"📝 Membuat file konfigurasi contoh: {config_path}")
                save_config(config_data, config_path)
    except ImportError:
        print("⚠️ Tidak dapat mengimpor save_config, file konfigurasi contoh tidak dibuat")

def run_config_sync_test():
    """Menjalankan test sinkronisasi konfigurasi"""
    print("\n" + "="*50)
    print("🧪 MENJALANKAN TEST SINKRONISASI KONFIGURASI")
    print("="*50)
    
    try:
        from smartcash.common.config.tests.test_config_sync import TestConfigSync
        TestConfigSync.test_config_sync()
    except Exception as e:
        print(f"⚠️ Error saat menjalankan test sinkronisasi konfigurasi: {str(e)}")

def run_env_config_test():
    """Menjalankan test environment config"""
    print("\n" + "="*50)
    print("🧪 MENJALANKAN TEST ENVIRONMENT CONFIG")
    print("="*50)
    
    try:
        # Import environment manager
        from smartcash.common.environment import get_environment_manager
        env_manager = get_environment_manager()
        
        # Dapatkan informasi environment
        env_info = env_manager.get_system_info()
        
        # Tampilkan informasi environment
        print("📊 Informasi Environment:")
        for key, value in env_info.items():
            print(f"  • {key}: {value}")
        
        # Cek apakah drive terhubung
        print(f"\n🔍 Status Google Drive: {'Terhubung' if env_manager.is_drive_mounted else 'Tidak terhubung'}")
        if env_manager.is_drive_mounted:
            print(f"  • Drive path: {env_manager.drive_path}")
        
        # Cek file konfigurasi environment
        env_config_path = Path(env_manager.base_dir) / "configs" / "environment_config.yaml"
        if env_config_path.exists():
            print(f"\n✅ File konfigurasi environment ditemukan: {env_config_path}")
            
            # Load konfigurasi environment
            from smartcash.common.io import load_config
            env_config = load_config(env_config_path, {})
            
            # Tampilkan isi konfigurasi environment
            print("📄 Isi konfigurasi environment:")
            for key, value in env_config.items():
                print(f"  • {key}: {value}")
        else:
            print(f"\n⚠️ File konfigurasi environment tidak ditemukan: {env_config_path}")
        
        # Cek direktori konfigurasi
        configs_dir = Path(env_manager.base_dir) / "configs"
        if configs_dir.exists():
            print(f"\n✅ Direktori konfigurasi ditemukan: {configs_dir}")
            
            # Hitung jumlah file konfigurasi
            config_files = list(configs_dir.glob("*.yaml")) + list(configs_dir.glob("*.yml")) + list(configs_dir.glob("*.json"))
            print(f"  • Jumlah file konfigurasi: {len(config_files)}")
            
            # Tampilkan daftar file konfigurasi
            print("  • Daftar file konfigurasi:")
            for config_file in config_files:
                print(f"    - {config_file.name}")
        else:
            print(f"\n⚠️ Direktori konfigurasi tidak ditemukan: {configs_dir}")
        
        # Cek sinkronisasi konfigurasi
        print("\n🔄 Menjalankan sinkronisasi konfigurasi...")
        if hasattr(env_manager, 'sync_config'):
            sync_success, sync_message = env_manager.sync_config()
            print(f"  • Hasil sinkronisasi: {sync_message}")
        else:
            print("  • Metode sync_config tidak ditemukan")
        
        # Simpan konfigurasi environment
        print("\n💾 Menyimpan konfigurasi environment...")
        if hasattr(env_manager, 'save_environment_config'):
            save_success, save_message = env_manager.save_environment_config()
            print(f"  • Hasil penyimpanan: {save_message}")
        else:
            print("  • Metode save_environment_config tidak ditemukan")
        
        print("\n✅ Test environment config selesai")
    except Exception as e:
        print(f"\n⚠️ Error saat menjalankan test environment config: {str(e)}")

def main():
    """Fungsi utama"""
    # Setup paths
    setup_paths()
    
    # Jalankan test
    run_config_sync_test()
    run_env_config_test()
    
    print("\n" + "="*50)
    print("✅ SEMUA TEST SELESAI")
    print("="*50)

if __name__ == "__main__":
    main()
