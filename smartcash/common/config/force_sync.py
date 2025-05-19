"""
File: smartcash/common/config/force_sync.py
Deskripsi: Utilitas untuk memastikan semua file konfigurasi berhasil disinkronkan
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any

def force_sync_all_configs(logger=None) -> Dict[str, List[str]]:
    """
    Memastikan semua file konfigurasi berhasil disinkronkan antara direktori smartcash/configs dan /content/configs.
    Menggunakan pendekatan langsung dengan menyalin file secara eksplisit.
    
    Args:
        logger: Logger untuk mencatat aktivitas
    
    Returns:
        Dictionary berisi hasil sinkronisasi
    """
    # Deteksi lingkungan Colab
    is_colab = Path('/content').exists()
    if not is_colab:
        if logger:
            logger.info("ℹ️ Tidak di lingkungan Colab, sinkronisasi dilewati")
        return {"synced": [], "skipped": []}
    
    try:
        # Dapatkan environment manager
        from smartcash.common.environment import get_environment_manager
        env_manager = get_environment_manager()
        
        # Dapatkan path ke direktori konfigurasi
        smartcash_config_dir = Path(env_manager.base_dir) / 'configs'
        content_config_dir = Path('/content/configs')
        
        # Pastikan kedua direktori ada
        smartcash_config_dir.mkdir(parents=True, exist_ok=True)
        content_config_dir.mkdir(parents=True, exist_ok=True)
        
        # Hasil sinkronisasi
        results = {"synced": [], "skipped": []}
        
        # Dapatkan daftar file konfigurasi di kedua direktori
        smartcash_configs = list(smartcash_config_dir.glob("*_config.yaml"))
        content_configs = list(content_config_dir.glob("*_config.yaml"))
        
        # Tampilkan informasi jumlah file
        if logger:
            logger.info(f"📊 Jumlah file konfigurasi di {smartcash_config_dir}: {len(smartcash_configs)}")
            logger.info(f"📊 Jumlah file konfigurasi di {content_config_dir}: {len(content_configs)}")
        
        # Salin file dari smartcash/configs ke /content/configs
        for config_file in smartcash_configs:
            target_file = content_config_dir / config_file.name
            try:
                # Salin file
                shutil.copy2(config_file, target_file)
                results["synced"].append(config_file.name)
                
                if logger:
                    logger.info(f"✅ Berhasil menyalin: {config_file.name} ke {target_file}")
            except Exception as e:
                results["skipped"].append(config_file.name)
                
                if logger:
                    logger.warning(f"⚠️ Gagal menyalin: {config_file.name} - {str(e)}")
        
        # Salin file dari /content/configs ke smartcash/configs
        for config_file in content_configs:
            if config_file.name not in [f.name for f in smartcash_configs]:
                target_file = smartcash_config_dir / config_file.name
                try:
                    # Salin file
                    shutil.copy2(config_file, target_file)
                    results["synced"].append(config_file.name)
                    
                    if logger:
                        logger.info(f"✅ Berhasil menyalin: {config_file.name} ke {target_file}")
                except Exception as e:
                    results["skipped"].append(config_file.name)
                    
                    if logger:
                        logger.warning(f"⚠️ Gagal menyalin: {config_file.name} - {str(e)}")
        
        # Tampilkan ringkasan
        if logger:
            logger.info(f"✅ Sinkronisasi selesai: {len(results['synced'])} file disinkronkan, {len(results['skipped'])} dilewati")
        
        return results
    except Exception as e:
        if logger:
            logger.error(f"❌ Error saat sinkronisasi: {str(e)}")
        return {"synced": [], "skipped": []}

def sync_specific_config(config_name: str, logger=None) -> bool:
    """
    Menyinkronkan file konfigurasi tertentu antara direktori smartcash/configs dan /content/configs.
    
    Args:
        config_name: Nama file konfigurasi
        logger: Logger untuk mencatat aktivitas
    
    Returns:
        Boolean yang menunjukkan keberhasilan sinkronisasi
    """
    # Deteksi lingkungan Colab
    is_colab = Path('/content').exists()
    if not is_colab:
        if logger:
            logger.info(f"ℹ️ Tidak di lingkungan Colab, sinkronisasi {config_name} dilewati")
        return False
    
    try:
        # Dapatkan environment manager
        from smartcash.common.environment import get_environment_manager
        env_manager = get_environment_manager()
        
        # Dapatkan path ke direktori konfigurasi
        smartcash_config_dir = Path(env_manager.base_dir) / 'configs'
        content_config_dir = Path('/content/configs')
        
        # Pastikan kedua direktori ada
        smartcash_config_dir.mkdir(parents=True, exist_ok=True)
        content_config_dir.mkdir(parents=True, exist_ok=True)
        
        # Dapatkan path ke file konfigurasi
        smartcash_config_file = smartcash_config_dir / config_name
        content_config_file = content_config_dir / config_name
        
        # Salin file dari smartcash/configs ke /content/configs
        if smartcash_config_file.exists():
            try:
                # Salin file
                shutil.copy2(smartcash_config_file, content_config_file)
                
                if logger:
                    logger.info(f"✅ Berhasil menyalin: {config_name} ke {content_config_file}")
                
                return True
            except Exception as e:
                if logger:
                    logger.warning(f"⚠️ Gagal menyalin: {config_name} - {str(e)}")
                
                return False
        # Salin file dari /content/configs ke smartcash/configs
        elif content_config_file.exists():
            try:
                # Salin file
                shutil.copy2(content_config_file, smartcash_config_file)
                
                if logger:
                    logger.info(f"✅ Berhasil menyalin: {config_name} ke {smartcash_config_file}")
                
                return True
            except Exception as e:
                if logger:
                    logger.warning(f"⚠️ Gagal menyalin: {config_name} - {str(e)}")
                
                return False
        else:
            if logger:
                logger.warning(f"⚠️ File konfigurasi tidak ditemukan: {config_name}")
            
            return False
    except Exception as e:
        if logger:
            logger.error(f"❌ Error saat sinkronisasi {config_name}: {str(e)}")
        
        return False
