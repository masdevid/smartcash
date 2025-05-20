#!/usr/bin/env python
"""
File: test_config_simple.py
Deskripsi: Test sederhana untuk verifikasi perbaikan sinkronisasi konfigurasi
"""

import os
import sys
import tempfile
import shutil
import yaml
from pathlib import Path

def get_default_split_config():
    """
    Dapatkan konfigurasi default untuk split dataset.
    
    Returns:
        Dictionary konfigurasi default
    """
    return {
        "split": {
            "enabled": True,
            "train_ratio": 0.7,
            "val_ratio": 0.15,
            "test_ratio": 0.15,
            "random_seed": 42,
            "stratify": True
        }
    }

def save_config(config, config_path):
    """
    Simpan konfigurasi ke file.
    
    Args:
        config: Konfigurasi yang akan disimpan
        config_path: Path file konfigurasi
        
    Returns:
        Boolean yang menunjukkan keberhasilan penyimpanan
    """
    try:
        # Pastikan direktori ada
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        # Simpan konfigurasi
        with open(config_path, 'w') as f:
            yaml.safe_dump(config, f)
        
        return True
    except Exception as e:
        print(f"❌ Error saat menyimpan konfigurasi: {str(e)}")
        return False

def load_config(config_path, default_config=None):
    """
    Load konfigurasi dari file.
    
    Args:
        config_path: Path file konfigurasi
        default_config: Konfigurasi default jika file tidak ditemukan
        
    Returns:
        Dictionary konfigurasi
    """
    try:
        # Periksa apakah file ada
        if not os.path.exists(config_path):
            return default_config or {}
        
        # Load konfigurasi
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    except Exception as e:
        print(f"❌ Error saat load konfigurasi: {str(e)}")
        return default_config or {}

def test_save_and_verify():
    """Test save dan verifikasi konfigurasi."""
    # Buat direktori temp untuk test
    temp_dir = tempfile.mkdtemp()
    config_path = os.path.join(temp_dir, 'configs', 'split_config.yaml')
    
    try:
        # Dapatkan konfigurasi default
        config = get_default_split_config()
        
        # Simpan konfigurasi
        save_success = save_config(config, config_path)
        
        # Verifikasi penyimpanan berhasil
        assert save_success, "Penyimpanan konfigurasi gagal"
        
        # Load konfigurasi
        loaded_config = load_config(config_path)
        
        # Verifikasi konfigurasi berhasil dimuat
        assert loaded_config, "Konfigurasi tidak berhasil dimuat"
        
        # Verifikasi konfigurasi sama dengan yang disimpan
        assert loaded_config == config, "Konfigurasi dimuat tidak sama dengan yang disimpan"
        
        # Modifikasi konfigurasi
        config['split']['train_ratio'] = 0.8
        config['split']['val_ratio'] = 0.1
        config['split']['test_ratio'] = 0.1
        
        # Simpan konfigurasi yang dimodifikasi
        save_config(config, config_path)
        
        # Load konfigurasi
        loaded_config = load_config(config_path)
        
        # Verifikasi konfigurasi berhasil dimodifikasi
        assert loaded_config['split']['train_ratio'] == 0.8, "train_ratio tidak diupdate dengan benar"
        assert loaded_config['split']['val_ratio'] == 0.1, "val_ratio tidak diupdate dengan benar"
        assert loaded_config['split']['test_ratio'] == 0.1, "test_ratio tidak diupdate dengan benar"
        
        print("✅ Test save dan verifikasi berhasil")
        return True
    except AssertionError as e:
        print(f"❌ Test gagal: {str(e)}")
        return False
    finally:
        # Hapus direktori temp
        shutil.rmtree(temp_dir)

def simulate_colab_sync():
    """Simulasi sinkronisasi dengan Google Drive."""
    # Buat direktori temp untuk test
    temp_dir = tempfile.mkdtemp()
    config_path = os.path.join(temp_dir, 'configs', 'split_config.yaml')
    drive_dir = os.path.join(temp_dir, 'drive', 'MyDrive')
    drive_config_path = os.path.join(drive_dir, 'configs', 'split_config.yaml')
    
    try:
        # Pastikan direktori ada
        os.makedirs(os.path.dirname(drive_config_path), exist_ok=True)
        
        # Dapatkan konfigurasi default
        config = get_default_split_config()
        config['split']['train_ratio'] = 0.75
        
        # Simpan konfigurasi lokal
        save_config(config, config_path)
        
        # Simulasi sync ke drive
        save_config(config, drive_config_path)
        
        # Load konfigurasi dari drive
        loaded_drive_config = load_config(drive_config_path)
        
        # Verifikasi konfigurasi di drive sama dengan lokal
        assert loaded_drive_config['split']['train_ratio'] == 0.75, "train_ratio tidak disinkronkan dengan benar"
        
        # Simulasi update di drive
        drive_config = loaded_drive_config.copy()
        drive_config['split']['train_ratio'] = 0.8
        save_config(drive_config, drive_config_path)
        
        # Simulasi sync dari drive
        loaded_drive_config = load_config(drive_config_path)
        save_config(loaded_drive_config, config_path)
        
        # Load konfigurasi lokal
        loaded_local_config = load_config(config_path)
        
        # Verifikasi konfigurasi lokal terupdate dari drive
        assert loaded_local_config['split']['train_ratio'] == 0.8, "train_ratio tidak disinkronkan dari drive dengan benar"
        
        print("✅ Test simulasi sinkronisasi berhasil")
        return True
    except AssertionError as e:
        print(f"❌ Test gagal: {str(e)}")
        return False
    finally:
        # Hapus direktori temp
        shutil.rmtree(temp_dir)

def test_sequential_updates():
    """Test update konfigurasi secara berurutan."""
    # Buat direktori temp untuk test
    temp_dir = tempfile.mkdtemp()
    config_path = os.path.join(temp_dir, 'configs', 'split_config.yaml')
    
    try:
        # Update 1: Ubah train_ratio
        config1 = get_default_split_config()
        config1['split']['train_ratio'] = 0.8
        save_config(config1, config_path)
        loaded1 = load_config(config_path)
        assert loaded1['split']['train_ratio'] == 0.8, "train_ratio tidak diupdate dengan benar"
        
        # Update 2: Ubah val_ratio
        config2 = loaded1.copy()
        config2['split']['val_ratio'] = 0.1
        save_config(config2, config_path)
        loaded2 = load_config(config_path)
        assert loaded2['split']['train_ratio'] == 0.8, "train_ratio berubah setelah update val_ratio"
        assert loaded2['split']['val_ratio'] == 0.1, "val_ratio tidak diupdate dengan benar"
        
        # Update 3: Ubah random_seed
        config3 = loaded2.copy()
        config3['split']['random_seed'] = 123
        save_config(config3, config_path)
        loaded3 = load_config(config_path)
        assert loaded3['split']['train_ratio'] == 0.8, "train_ratio berubah setelah update random_seed"
        assert loaded3['split']['val_ratio'] == 0.1, "val_ratio berubah setelah update random_seed"
        assert loaded3['split']['random_seed'] == 123, "random_seed tidak diupdate dengan benar"
        
        print("✅ Test sequential updates berhasil")
        return True
    except AssertionError as e:
        print(f"❌ Test gagal: {str(e)}")
        return False
    finally:
        # Hapus direktori temp
        shutil.rmtree(temp_dir)

def run_all_tests():
    """Jalankan semua test."""
    tests = [
        test_save_and_verify,
        test_sequential_updates,
        simulate_colab_sync
    ]
    
    results = []
    for test in tests:
        print(f"\n🔄 Menjalankan {test.__name__}...")
        result = test()
        results.append(result)
    
    # Print summary
    print("\n" + "="*50)
    print("📊 HASIL TEST")
    print("="*50)
    for i, test in enumerate(tests):
        status = "✅ BERHASIL" if results[i] else "❌ GAGAL"
        print(f"{status}: {test.__name__}")
    
    # Overall result
    if all(results):
        print("\n✅ Semua test berhasil")
        return True
    else:
        print("\n❌ Beberapa test gagal")
        return False

if __name__ == "__main__":
    run_all_tests() 