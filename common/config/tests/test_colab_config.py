"""
File: smartcash/common/config/tests/test_colab_config.py
Deskripsi: Test untuk penggunaan colab_config.yaml dalam sistem konfigurasi
"""

import os
from pathlib import Path
import yaml
import tempfile
import shutil

from smartcash.common.config import get_config_manager, SimpleConfigManager
from smartcash.common.utils import is_colab

def setup_test_env():
    """Setup lingkungan test"""
    # Buat temporary directory
    temp_dir = Path(tempfile.mkdtemp())
    
    # Buat struktur direktori
    config_dir = temp_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    
    # Salin colab_config.yaml ke temp directory
    src_config = Path(__file__).resolve().parents[3] / "configs/colab_config.yaml"
    if src_config.exists():
        shutil.copy2(src_config, config_dir / "colab_config.yaml")
    
    return temp_dir, config_dir

def cleanup_test_env(temp_dir):
    """Bersihkan lingkungan test"""
    if temp_dir.exists():
        shutil.rmtree(temp_dir)

def test_colab_config_loading():
    """Test loading colab_config.yaml"""
    print("=== Test Loading Colab Config ===")
    
    # Setup environment
    temp_dir, config_dir = setup_test_env()
    
    try:
        # Load config langsung dari file untuk pengujian
        colab_config_path = Path(__file__).resolve().parents[3] / "configs/colab_config.yaml"
        if not colab_config_path.exists():
            print(f"❌ File {colab_config_path} tidak ditemukan")
            return
            
        with open(colab_config_path, 'r') as f:
            colab_config = yaml.safe_load(f)
        
        # Cek apakah config berhasil dimuat
        if colab_config:
            print("✅ Colab config berhasil dimuat langsung dari file")
            
            # Cek struktur config
            if '_base_' in colab_config:
                print(f"✅ Base config: {colab_config['_base_']}")
            
            if 'drive' in colab_config:
                drive_config = colab_config['drive']
                print("✅ Drive configuration:")
                print(f"   - use_drive: {drive_config.get('use_drive')}")
                print(f"   - sync_strategy: {drive_config.get('sync_strategy')}")
                
                if 'paths' in drive_config:
                    paths = drive_config['paths']
                    print("   - Paths:")
                    for key, value in paths.items():
                        print(f"     - {key}: {value}")
            
            if 'environment' in colab_config:
                env_config = colab_config['environment']
                print("✅ Environment configuration:")
                print(f"   - colab: {env_config.get('colab')}")
                
                if 'required_packages' in env_config:
                    print(f"   - required packages: {len(env_config['required_packages'])}")
            
            if 'model' in colab_config:
                model_config = colab_config['model']
                print("✅ Model configuration:")
                print(f"   - use_gpu: {model_config.get('use_gpu')}")
                print(f"   - use_tpu: {model_config.get('use_tpu')}")
                print(f"   - precision: {model_config.get('precision')}")
            
            if 'performance' in colab_config:
                perf_config = colab_config['performance']
                print("✅ Performance configuration:")
                print(f"   - auto_garbage_collect: {perf_config.get('auto_garbage_collect')}")
                print(f"   - checkpoint_to_drive: {perf_config.get('checkpoint_to_drive')}")
        else:
            print("❌ Gagal memuat colab config dari file")
    
    finally:
        cleanup_test_env(temp_dir)

def test_colab_environment_detection():
    """Test deteksi lingkungan Colab"""
    print("\n=== Test Deteksi Lingkungan Colab ===")
    
    # Cek apakah berjalan di Colab
    if is_colab():
        print("✅ Berjalan di Google Colab")
        
        # Cek keberadaan /content/drive
        if Path("/content/drive").exists():
            print("✅ Google Drive sudah dimount")
        else:
            print("❌ Google Drive belum dimount")
        
        # Cek direktori SmartCash di Drive
        drive_dir = Path("/content/drive/MyDrive/SmartCash")
        if drive_dir.exists():
            print(f"✅ Direktori {drive_dir} ditemukan")
        else:
            print(f"❌ Direktori {drive_dir} tidak ditemukan")
    else:
        print("ℹ️ Tidak berjalan di Google Colab, test drive tidak dilakukan")
        print("   Simulasi konfigurasi Colab diverifikasi dari file")

def test_symlink_creation():
    """Test pembuatan symlink"""
    print("\n=== Test Pembuatan Symlink ===")
    
    if not is_colab():
        print("ℹ️ Test ini hanya berjalan di Google Colab")
        
        # Tampilkan langkah-langkah yang akan dilakukan di Colab
        print("Langkah-langkah yang akan dilakukan di Colab:")
        print("1. Cek apakah /content/configs adalah symlink")
        print("2. Verifikasi symlink mengarah ke /content/drive/MyDrive/SmartCash/configs")
        print("3. Pastikan konfigurasi tersimpan di Google Drive")
        return
    
    config_dir = Path("/content/configs")
    
    # Cek apakah config_dir adalah symlink
    if config_dir.is_symlink():
        target = os.path.realpath(config_dir)
        print(f"✅ {config_dir} adalah symlink ke {target}")
        
        # Cek apakah target adalah di Drive
        if "/content/drive/MyDrive/SmartCash/configs" in target:
            print("✅ Symlink mengarah ke direktori di Google Drive")
        else:
            print(f"❌ Symlink tidak mengarah ke direktori di Google Drive: {target}")
    else:
        print(f"❌ {config_dir} bukan symlink")

def test_simple_config_integration():
    """Test SimpleConfigManager dengan colab_config.yaml"""
    print("\n=== Test SimpleConfigManager dengan Colab Config ===")
    
    # Setup
    config_manager = get_config_manager()
    
    # Load colab_config
    config = config_manager.get_config('colab')
    
    if config:
        print("✅ Berhasil load colab_config dengan SimpleConfigManager")
        
        # Cek struktur penting
        if 'drive' in config and isinstance(config['drive'], dict):
            print("✅ Konfigurasi drive ditemukan")
            
            # Tampilkan beberapa setting
            drive_config = config['drive']
            print(f"   - use_drive: {drive_config.get('use_drive')}")
            
        if 'environment' in config and isinstance(config['environment'], dict):
            print("✅ Konfigurasi environment ditemukan")
            
            # Tampilkan beberapa setting
            env_config = config['environment']
            print(f"   - colab: {env_config.get('colab')}")
            
        # Test save dan reload
        test_update = {'test_field': 'test_value'}
        config.update(test_update)
        
        # Save ke file
        save_success = config_manager.save_config(config, 'colab')
        if save_success:
            print("✅ Berhasil menyimpan colab_config")
            
            # Reload untuk verifikasi
            updated_config = config_manager.get_config('colab', reload=True)
            if 'test_field' in updated_config and updated_config['test_field'] == 'test_value':
                print("✅ Field yang diupdate berhasil disimpan dan di-reload")
            else:
                print("❌ Field yang diupdate tidak berhasil disimpan atau di-reload")
        else:
            print("❌ Gagal menyimpan colab_config")
    else:
        print("❌ Gagal load colab_config dengan SimpleConfigManager")

def run_tests():
    """Jalankan semua pengujian"""
    print("🧪 Menjalankan pengujian Colab Config...\n")
    
    test_colab_config_loading()
    test_colab_environment_detection()
    test_symlink_creation()
    test_simple_config_integration()
    
    print("\n🎉 Pengujian selesai!")

if __name__ == "__main__":
    run_tests() 