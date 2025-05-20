"""
File: smartcash/common/config/tests/sync_all_configs.py
Deskripsi: Script untuk menyalin semua file konfigurasi dari satu direktori ke direktori lainnya
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List, Set, Tuple

def copy_all_configs(source_dir: str, target_dir: str, logger=None) -> Dict[str, int]:
    """
    Menyalin semua file konfigurasi dari direktori sumber ke direktori target.
    
    Args:
        source_dir: Direktori sumber
        target_dir: Direktori target
        logger: Logger untuk mencatat aktivitas
    
    Returns:
        Dictionary berisi hasil sinkronisasi
    """
    # Pastikan direktori ada
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    if not source_path.exists():
        if logger:
            logger.warning(f"⚠️ Direktori sumber tidak ditemukan: {source_path}")
        return {"copied": 0, "skipped": 0, "error": 0}
    
    # Buat direktori target jika belum ada
    target_path.mkdir(parents=True, exist_ok=True)
    
    # Dapatkan daftar file konfigurasi di direktori sumber
    def get_config_files(dir_path: Path) -> List[Path]:
        """Mendapatkan daftar file konfigurasi di direktori"""
        if not dir_path.exists():
            return []
        
        # Kumpulkan semua file konfigurasi (yaml, yml, json)
        config_files = []
        for ext in ['*.yaml', '*.yml', '*.json']:
            config_files.extend(dir_path.glob(ext))
        
        # Filter file yang diawali dengan 'backup' atau '_'
        return [f for f in config_files if not f.name.startswith('backup') and not f.name.startswith('_')]
    
    # Dapatkan daftar file konfigurasi
    source_configs = get_config_files(source_path)
    
    # Hasil sinkronisasi
    results = {"copied": 0, "skipped": 0, "error": 0}
    
    # Salin setiap file konfigurasi
    for source_file in source_configs:
        target_file = target_path / source_file.name
        
        try:
            # Salin file
            shutil.copy2(source_file, target_file)
            results["copied"] += 1
            
            if logger:
                logger.info(f"✅ Berhasil menyalin: {source_file.name}")
        except Exception as e:
            results["error"] += 1
            
            if logger:
                logger.warning(f"⚠️ Gagal menyalin: {source_file.name} - {str(e)}")
    
    # Tampilkan ringkasan
    if logger:
        logger.info(f"📊 Total file: {len(source_configs)}, Berhasil: {results['copied']}, Gagal: {results['error']}")
    
    return results

def sync_smartcash_configs():
    """
    Menyinkronkan semua file konfigurasi antara direktori smartcash/configs dan /content/configs.
    Catatan: Hanya menggunakan /content/configs sebagai direktori konfigurasi di lingkungan Colab.
    """
    try:
        # Import logger
        from smartcash.common.logger import get_logger
        logger = get_logger()
        
        # Dapatkan environment manager
        from smartcash.common.environment import get_environment_manager
        env_manager = get_environment_manager()
        
        # Dapatkan path ke direktori konfigurasi
        smartcash_config_dir = Path(env_manager.base_dir) / 'configs'
        content_config_dir = Path('/content/configs')
        
        # Deteksi lingkungan Colab
        is_colab = Path('/content').exists()
        
        # Jika di Colab, gunakan /content/configs sebagai direktori konfigurasi utama
        if is_colab:
            # Pastikan direktori ada
            content_config_dir.mkdir(parents=True, exist_ok=True)
            
            # Salin file dari smartcash/configs ke /content/configs
            logger.info(f"🔄 Menyalin file dari {smartcash_config_dir} ke {content_config_dir}...")
            results_to_content = copy_all_configs(smartcash_config_dir, content_config_dir, logger)
            
            # Tampilkan ringkasan
            logger.info(f"✅ Sinkronisasi selesai: {results_to_content['copied']} file disinkronkan ke {content_config_dir}")
        else:
            # Di lingkungan lokal, gunakan smartcash/configs sebagai direktori konfigurasi utama
            logger.info(f"ℹ️ Tidak di lingkungan Colab, menggunakan {smartcash_config_dir} sebagai direktori konfigurasi utama")
            results_to_content = {"copied": 0, "skipped": 0, "error": 0}
        
        # Jalankan test untuk memastikan semua file berhasil disinkronkan
        from smartcash.common.config.tests.test_config_sync import TestConfigSync
        TestConfigSync.test_config_sync()
        
        return True
    except Exception as e:
        print(f"⚠️ Error saat sinkronisasi konfigurasi: {str(e)}")
        return False

if __name__ == "__main__":
    print("🔄 Menjalankan sinkronisasi konfigurasi...")
    success = sync_smartcash_configs()
    
    if success:
        print("✅ Sinkronisasi konfigurasi berhasil")
    else:
        print("⚠️ Sinkronisasi konfigurasi gagal")
