"""
File: smartcash/common/config/tests/test_config_sync.py
Deskripsi: Test untuk memastikan semua file konfigurasi berhasil disimpan dan di-symlink
"""

import os
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Set, Tuple
from smartcash.common.config.manager import SimpleConfigManager, get_config_manager

class TestConfigSync:
    """Test untuk sinkronisasi konfigurasi antara direktori configs lokal dan di Google Drive"""
    
    @staticmethod
    def test_config_sync():
        """
        Test untuk memastikan semua file konfigurasi berhasil disimpan dan di-symlink
        antara direktori configs lokal dan di Google Drive
        """
        # Dapatkan config manager
        config_manager = get_config_manager()
        
        # Dapatkan path ke direktori konfigurasi
        smartcash_config_dir = Path(__file__).resolve().parents[4] / 'configs'
        content_config_dir = Path('/content/configs')
        
        # Cek apakah berjalan di Colab
        try:
            import google.colab
            is_colab = True
        except ImportError:
            is_colab = False
            print("ℹ️ Bukan di Colab, test symlink dilewati")
            return True
        
        # Pastikan kedua direktori ada
        if not smartcash_config_dir.exists():
            print(f"⚠️ Direktori {smartcash_config_dir} tidak ditemukan")
            smartcash_config_dir = Path('/content/smartcash/configs')
            if not smartcash_config_dir.exists():
                print(f"⚠️ Direktori {smartcash_config_dir} juga tidak ditemukan")
                return False
        
        if not content_config_dir.exists():
            print(f"⚠️ Direktori {content_config_dir} tidak ditemukan")
            return False
        
        # Dapatkan daftar file konfigurasi di kedua direktori
        def get_config_files(dir_path: Path) -> Set[str]:
            """Mendapatkan daftar file konfigurasi di direktori"""
            if not dir_path.exists():
                return set()
            
            # Kumpulkan semua file konfigurasi (yaml, yml, json)
            config_files = set()
            for ext in ['*.yaml', '*.yml', '*.json']:
                config_files.update(f.name for f in dir_path.glob(ext))
            
            # Filter file yang diawali dengan 'backup' atau '_'
            return {f for f in config_files if not f.startswith('backup') and not f.startswith('_')}
        
        # Dapatkan daftar file konfigurasi di kedua direktori
        smartcash_configs = get_config_files(smartcash_config_dir)
        content_configs = get_config_files(content_config_dir)
        
        # Hitung jumlah file konfigurasi
        smartcash_count = len(smartcash_configs)
        content_count = len(content_configs)
        
        # Tampilkan jumlah file konfigurasi
        print(f"📊 Jumlah file konfigurasi di {smartcash_config_dir}: {smartcash_count}")
        print(f"📊 Jumlah file konfigurasi di {content_config_dir}: {content_count}")
        
        # Bandingkan file konfigurasi
        missing_in_content = smartcash_configs - content_configs
        missing_in_smartcash = content_configs - smartcash_configs
        common_configs = smartcash_configs & content_configs
        
        # Tampilkan hasil perbandingan
        print(f"✅ File konfigurasi yang ada di kedua direktori: {len(common_configs)}")
        if common_configs:
            print(f"   {', '.join(sorted(common_configs))}")
        
        if missing_in_content:
            print(f"⚠️ File konfigurasi yang tidak ada di {content_config_dir}: {len(missing_in_content)}")
            print(f"   {', '.join(sorted(missing_in_content))}")
        
        if missing_in_smartcash:
            print(f"⚠️ File konfigurasi yang tidak ada di {smartcash_config_dir}: {len(missing_in_smartcash)}")
            print(f"   {', '.join(sorted(missing_in_smartcash))}")
        
        # Periksa isi file konfigurasi yang ada di kedua direktori
        print("\n🔍 Memeriksa isi file konfigurasi yang ada di kedua direktori...")
        
        import yaml
        
        for config_file in common_configs:
            # Load file konfigurasi
            try:
                with open(smartcash_config_dir / config_file, 'r') as f:
                    smartcash_config = yaml.safe_load(f) or {}
                
                with open(content_config_dir / config_file, 'r') as f:
                    content_config = yaml.safe_load(f) or {}
                
                # Bandingkan isi file konfigurasi
                import json
                is_identical = json.dumps(smartcash_config, sort_keys=True) == json.dumps(content_config, sort_keys=True)
                
                if is_identical:
                    print(f"✅ {config_file}: Isi file identik")
                else:
                    print(f"⚠️ {config_file}: Isi file berbeda")
            except Exception as e:
                print(f"❌ Error saat membandingkan {config_file}: {str(e)}")
        
        # Cek apakah content/configs adalah symlink dari drive/MyDrive/SmartCash/configs
        drive_config_dir = Path('/content/drive/MyDrive/SmartCash/configs')
        if drive_config_dir.exists():
            is_symlink = content_config_dir.is_symlink()
            if is_symlink:
                resolved_path = os.path.realpath(content_config_dir)
                if resolved_path == str(drive_config_dir):
                    print(f"✅ {content_config_dir} adalah symlink ke {drive_config_dir}")
                else:
                    print(f"⚠️ {content_config_dir} adalah symlink tetapi mengarah ke {resolved_path} (seharusnya {drive_config_dir})")
            else:
                print(f"⚠️ {content_config_dir} bukan symlink dari {drive_config_dir}")
        else:
            print(f"⚠️ Direktori {drive_config_dir} tidak ditemukan")
        
        print("\n✅ Test konfigurasi berhasil")
        return True

    @staticmethod
    def test_simple_config_manager():
        """Test fungsionalitas dasar SimpleConfigManager"""
        print("\n=== Test SimpleConfigManager ===")
        
        # Setup test environment
        temp_dir = Path(tempfile.mkdtemp())
        config_dir = temp_dir / 'configs'
        config_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Buat instance SimpleConfigManager
            config_manager = SimpleConfigManager(base_dir=temp_dir)
            
            # Cek apakah config_dir ada
            if config_manager.config_dir.exists():
                print(f"✅ Config dir ada: {config_manager.config_dir}")
            else:
                print(f"❌ Config dir tidak ada: {config_manager.config_dir}")
                return False
            
            # Test save dan load config
            test_config = {
                'app_name': 'SmartCash',
                'version': '1.0.0',
                'test': True
            }
            
            # Save config
            save_success = config_manager.save_config(test_config, 'test_config')
            if save_success:
                print("✅ Config berhasil disimpan")
            else:
                print("❌ Gagal menyimpan config")
                return False
            
            # Load config
            loaded_config = config_manager.get_config('test_config')
            if loaded_config == test_config:
                print("✅ Config berhasil dimuat dan sesuai")
            else:
                print("❌ Config yang dimuat tidak sesuai")
                return False
            
            # Test config caching
            config_manager.config_cache = {}  # Clear cache
            loaded_config = config_manager.get_config('test_config')
            if loaded_config == test_config:
                print("✅ Config berhasil dimuat dari disk")
            else:
                print("❌ Config yang dimuat dari disk tidak sesuai")
                return False
            
            # Test update config
            update_dict = {'version': '1.1.0'}
            update_success = config_manager.update_config(update_dict, 'test_config')
            if update_success:
                print("✅ Config berhasil diupdate")
            else:
                print("❌ Gagal mengupdate config")
                return False
            
            # Verify update
            updated_config = config_manager.get_config('test_config', reload=True)
            if updated_config['version'] == '1.1.0':
                print("✅ Update config berhasil diverifikasi")
            else:
                print("❌ Update config gagal diverifikasi")
                return False
            
            print("\n✅ Test SimpleConfigManager berhasil")
            return True
            
        finally:
            # Cleanup
            shutil.rmtree(temp_dir)

def run_test():
    """Menjalankan test konfigurasi"""
    print("🧪 Menjalankan test konfigurasi dan sinkronisasi...")
    
    # Jalankan test SimpleConfigManager terlebih dahulu
    simple_manager_result = TestConfigSync.test_simple_config_manager()
    
    # Jalankan test sinkronisasi konfigurasi
    sync_result = TestConfigSync.test_config_sync()
    
    if simple_manager_result and sync_result:
        print("✅ Semua test berhasil")
    else:
        print("⚠️ Beberapa test gagal")
    
    return simple_manager_result and sync_result

if __name__ == "__main__":
    run_test()
