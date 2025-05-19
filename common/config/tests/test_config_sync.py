"""
File: smartcash/common/config/tests/test_config_sync.py
Deskripsi: Test untuk memastikan semua file konfigurasi berhasil disinkronkan dan error base_dir None dicegah
"""

import os
import pytest
from pathlib import Path
from typing import Dict, List, Set, Tuple
from smartcash.common.config.manager import ConfigManager, get_config_manager

class TestConfigSync:
    """Test untuk sinkronisasi konfigurasi antara smartcash/configs dan /content/configs"""
    
    @staticmethod
    def test_config_sync():
        """
        Test untuk memastikan semua file konfigurasi berhasil disinkronkan
        antara smartcash/configs dan /content/configs
        """
        # Dapatkan environment manager
        from smartcash.common.environment import get_environment_manager
        env_manager = get_environment_manager()
        
        # Dapatkan path ke direktori konfigurasi
        smartcash_config_dir = Path(env_manager.base_dir) / 'configs'
        content_config_dir = Path('/content/configs')
        
        # Pastikan kedua direktori ada
        if not smartcash_config_dir.exists():
            print(f"⚠️ Direktori {smartcash_config_dir} tidak ditemukan")
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
        
        from smartcash.common.io import load_config
        
        for config_file in common_configs:
            smartcash_config = load_config(smartcash_config_dir / config_file, {})
            content_config = load_config(content_config_dir / config_file, {})
            
            # Bandingkan isi file konfigurasi
            import json
            is_identical = json.dumps(smartcash_config, sort_keys=True) == json.dumps(content_config, sort_keys=True)
            
            if is_identical:
                print(f"✅ {config_file}: Isi file identik")
            else:
                print(f"⚠️ {config_file}: Isi file berbeda")
        
        # Tampilkan kesimpulan
        if smartcash_count == content_count and not missing_in_content and not missing_in_smartcash:
            print("\n✅ Semua file konfigurasi berhasil disinkronkan")
            return True
        else:
            print("\n⚠️ Beberapa file konfigurasi tidak berhasil disinkronkan")
            return False

    @staticmethod
    def test_config_manager_base_dir_none():
        """Test ConfigManager raises ValueError if base_dir is None"""
        with pytest.raises(ValueError):
            ConfigManager(base_dir=None, config_file='test.yaml')

    @staticmethod
    def test_get_config_manager_base_dir_none():
        """Test get_config_manager raises ValueError if base_dir is None"""
        from importlib import reload
        import smartcash.common.config.manager as manager_mod
        reload(manager_mod)  # Reset singleton
        with pytest.raises(ValueError):
            manager_mod.get_config_manager(base_dir=None, config_file='test.yaml')

def run_test():
    """Menjalankan test sinkronisasi konfigurasi"""
    print("🧪 Menjalankan test sinkronisasi konfigurasi...")
    result = TestConfigSync.test_config_sync()
    
    if result:
        print("✅ Test sinkronisasi konfigurasi berhasil")
    else:
        print("⚠️ Test sinkronisasi konfigurasi gagal")
    
    return result

if __name__ == "__main__":
    run_test()
