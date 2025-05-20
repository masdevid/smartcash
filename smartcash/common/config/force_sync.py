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

def detect_colab_environment() -> bool:
    """
    Deteksi apakah kode berjalan di Google Colab dengan metode yang lebih handal.
    
    Returns:
        bool: True jika di Google Colab, False jika tidak
    """
    # Metode 1: Periksa variabel lingkungan
    if "COLAB_GPU" in os.environ or "COLAB_TPU_ADDR" in os.environ:
        return True
    
    # Metode 2: Periksa apakah 'google.colab' tersedia
    try:
        import google.colab
        return True
    except ImportError:
        pass
    
    # Metode 3: Periksa path /content yang tipikal di Colab
    if os.path.exists('/content') and os.path.isdir('/content'):
        # Verifikasi tambahan untuk memastikan ini benar-benar Colab
        if os.path.exists('/content/drive'):
            return True
    
    return False

def verify_drive_mounted() -> bool:
    """
    Verifikasi Google Drive sudah ter-mount dengan benar.
    
    Returns:
        bool: True jika Google Drive ter-mount, False jika tidak
    """
    # Periksa mounting point standar di Colab
    drive_paths = ['/content/drive', '/content/gdrive']
    
    for path in drive_paths:
        if os.path.exists(path) and os.path.isdir(path):
            # Verifikasi lebih lanjut dengan memeriksa adanya 'My Drive'
            my_drive_path = os.path.join(path, 'My Drive')
            if os.path.exists(my_drive_path) and os.path.isdir(my_drive_path):
                return True
    
    return False

def mount_drive_if_needed() -> bool:
    """
    Mount Google Drive jika belum ter-mount dan kita di dalam Colab.
    
    Returns:
        bool: True jika berhasil, False jika gagal
    """
    # Jika bukan di Colab, langsung return False
    if not detect_colab_environment():
        return False
    
    # Jika sudah ter-mount, langsung return True
    if verify_drive_mounted():
        return True
    
    # Coba mount drive
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        
        # Verifikasi mounting berhasil
        return verify_drive_mounted()
    except Exception as e:
        logger.error(f"❌ Gagal mount Google Drive: {str(e)}")
        return False

def sync_with_drive(config: Dict[str, Any], module_name: str, 
                    ui_components: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Sinkronisasi konfigurasi dengan Google Drive yang ditingkatkan dengan verifikasi.
    
    Args:
        config: Konfigurasi yang akan disinkronkan
        module_name: Nama modul konfigurasi
        ui_components: Dictionary komponen UI (opsional)
        
    Returns:
        Konfigurasi yang telah disinkronkan
    """
    try:
        # Import sync_logger jika ada ui_components
        sync_logger = None
        if ui_components:
            try:
                from smartcash.ui.dataset.split.handlers.sync_logger import (
                    log_sync_info, log_sync_success, log_sync_error, update_sync_status_only
                )
                sync_logger = True
            except ImportError:
                sync_logger = None
        
        # Log status sinkronisasi dimulai
        if ui_components and sync_logger:
            update_sync_status_only(ui_components, "Menyinkronkan konfigurasi dengan Google Drive...", 'info')
        
        # Periksa apakah kita berada di Colab
        is_colab = detect_colab_environment()
        if not is_colab:
            # Tidak perlu sinkronisasi jika bukan di Colab
            logger.info(f"🔄 Tidak perlu sinkronisasi (bukan di Google Colab)")
            if ui_components and sync_logger:
                log_sync_info(ui_components, "Tidak perlu sinkronisasi (bukan di Google Colab)")
            return config
        
        # Pastikan Google Drive ter-mount
        if not verify_drive_mounted():
            # Coba mount Google Drive
            success = mount_drive_if_needed()
            if not success:
                logger.warning(f"⚠️ Google Drive tidak ter-mount, sinkronisasi tidak dilakukan")
                if ui_components and sync_logger:
                    log_sync_warning(ui_components, "Google Drive tidak ter-mount, sinkronisasi tidak dilakukan")
                return config
        
        # Dapatkan config manager
        from smartcash.common.config.manager import get_config_manager
        from smartcash.common.environment import get_environment_manager
        
        env = get_environment_manager()
        base_dir = getattr(env, 'base_dir', '/content')
        
        config_manager = get_config_manager(base_dir=base_dir)
        
        # Sinkronisasi konfigurasi
        # Simpan konfigurasi lokal terlebih dahulu
        local_save_success = config_manager.save_module_config(module_name, config)
        
        if not local_save_success:
            logger.error(f"❌ Gagal menyimpan konfigurasi {module_name} secara lokal")
            if ui_components and sync_logger:
                log_sync_error(ui_components, f"Gagal menyimpan konfigurasi {module_name} secara lokal")
            return config
        
        # Upload ke Google Drive
        from smartcash.common.config.sync import upload_config_to_drive
        
        config_path = config_manager._get_module_config_path(module_name)
        drive_sync_success, drive_message = config_manager.sync_to_drive(module_name)
        
        if not drive_sync_success:
            logger.error(f"❌ Gagal sinkronisasi dengan Google Drive: {drive_message}")
            if ui_components and sync_logger:
                log_sync_error(ui_components, f"Gagal sinkronisasi dengan Google Drive: {drive_message}")
            return config
        
        # Verifikasi konfigurasi dengan membaca ulang dari Drive
        success, message, synced_config = config_manager.sync_with_drive(
            config_manager._get_module_config_filename(module_name),
            sync_strategy='drive_priority'
        )
        
        # Verifikasi konsistensi data
        if success:
            # Muat config dari lokal untuk verifikasi
            local_config = config_manager.get_module_config(module_name, {})
            
            # Bandingkan key utama dalam konfigurasi
            if module_name in local_config and module_name in synced_config:
                main_config_key = module_name
            else:
                # Cari key pertama sebagai key utama
                main_config_key = list(config.keys())[0] if config else None
            
            # Verifikasi konsistensi
            if main_config_key and main_config_key in local_config and main_config_key in synced_config:
                is_consistent = True
                for key, value in local_config[main_config_key].items():
                    if key not in synced_config[main_config_key] or synced_config[main_config_key][key] != value:
                        is_consistent = False
                        logger.warning(f"⚠️ Inkonsistensi data pada key '{key}': {value} vs {synced_config[main_config_key].get(key, 'tidak ada')}")
                        break
                
                if is_consistent:
                    logger.info(f"✅ Konfigurasi {module_name} berhasil disinkronkan dengan Google Drive")
                    if ui_components and sync_logger:
                        log_sync_success(ui_components, f"Konfigurasi {module_name} berhasil disinkronkan dengan Google Drive")
                    return synced_config
                else:
                    logger.warning(f"⚠️ Konsistensi data tidak terjamin setelah sinkronisasi")
                    if ui_components and sync_logger:
                        log_sync_warning(ui_components, "Konsistensi data tidak terjamin setelah sinkronisasi. Mencoba lagi...")
                    
                    # Coba sekali lagi dengan upload langsung
                    direct_success, _ = upload_config_to_drive(config_path, config)
                    if direct_success:
                        logger.info(f"✅ Konfigurasi {module_name} berhasil disinkronkan dengan Google Drive (direct upload)")
                        if ui_components and sync_logger:
                            log_sync_success(ui_components, f"Konfigurasi {module_name} berhasil disinkronkan dengan Google Drive")
                        return config
            
            logger.info(f"✅ Konfigurasi {module_name} berhasil disinkronkan dengan Google Drive")
            if ui_components and sync_logger:
                log_sync_success(ui_components, f"Konfigurasi {module_name} berhasil disinkronkan dengan Google Drive")
            return synced_config
        else:
            logger.error(f"❌ Gagal sinkronisasi dengan Google Drive: {message}")
            if ui_components and sync_logger:
                log_sync_error(ui_components, f"Gagal sinkronisasi dengan Google Drive: {message}")
            return config
    
    except Exception as e:
        logger.error(f"❌ Error saat sinkronisasi dengan Google Drive: {str(e)}")
        import traceback
        logger.debug(f"Traceback: {traceback.format_exc()}")
        
        # Log error ke UI
        if ui_components and sync_logger:
            log_sync_error(ui_components, f"Error saat sinkronisasi dengan Google Drive: {str(e)}")
            
        return config

def force_sync_with_drive(module_name: str, max_retries: int = 3) -> Tuple[bool, Dict[str, Any]]:
    """
    Memaksa sinkronisasi konfigurasi dengan Google Drive dengan percobaan ulang.
    
    Args:
        module_name: Nama modul konfigurasi
        max_retries: Jumlah maksimum percobaan ulang
        
    Returns:
        Tuple (success, config)
    """
    try:
        from smartcash.common.config.manager import get_config_manager
        from smartcash.common.environment import get_environment_manager
        
        env = get_environment_manager()
        base_dir = getattr(env, 'base_dir', '/content')
        
        config_manager = get_config_manager(base_dir=base_dir)
        
        # Dapatkan konfigurasi saat ini
        config = config_manager.get_module_config(module_name, {})
        
        # Sinkronisasi dengan Google Drive
        retry_count = 0
        success = False
        
        while retry_count < max_retries and not success:
            # Sinkronisasi dengan Google Drive
            success, message, _ = config_manager.sync_with_drive(
                config_manager._get_module_config_filename(module_name),
                sync_strategy='drive_priority'
            )
            
            if success:
                # Verifikasi sinkronisasi berhasil dengan memuat ulang konfigurasi
                synced_config = config_manager.get_module_config(module_name, {})
                return True, synced_config
            else:
                logger.warning(f"⚠️ Percobaan ke-{retry_count+1} gagal: {message}")
                retry_count += 1
                time.sleep(1)  # Tunggu sebentar sebelum mencoba lagi
        
        if not success:
            logger.error(f"❌ Gagal memaksa sinkronisasi setelah {max_retries} percobaan")
            return False, config
        
        # Muat ulang konfigurasi setelah sinkronisasi
        synced_config = config_manager.get_module_config(module_name, {})
        return True, synced_config
    
    except Exception as e:
        logger.error(f"❌ Error saat memaksa sinkronisasi: {str(e)}")
        return False, {}
