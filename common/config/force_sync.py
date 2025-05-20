"""
File: smartcash/common/config/force_sync.py
Deskripsi: Utilitas untuk memastikan semua file konfigurasi berhasil disinkronkan
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any
import yaml

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

def sync_with_drive(config: Dict[str, Any], module_name: str, ui_components: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Sinkronisasi konfigurasi dengan Google Drive.
    
    Args:
        config: Konfigurasi yang akan disinkronkan
        module_name: Nama modul konfigurasi
        ui_components: Dictionary komponen UI (opsional)
        
    Returns:
        Konfigurasi yang telah disinkronkan
    """
    try:
        # Jika bukan di Colab, tidak perlu sinkronisasi
        if not detect_colab_environment():
            return config
        
        # Import logger jika diperlukan
        from smartcash.common.logger import get_logger
        logger = get_logger(__name__)
        
        # Pastikan config memiliki struktur yang benar
        if module_name not in config and isinstance(config, dict):
            config = {module_name: config}
            
        # Dapatkan config manager
        from smartcash.common.config.manager import get_config_manager
        base_dir = get_default_base_dir()
        config_manager = get_config_manager(base_dir=base_dir)
        
        # Cek apakah ada konfigurasi yang sudah ada sebelumnya
        existing_config = config_manager.get_module_config(module_name, {})
        
        # Gabungkan konfigurasi yang ada dengan yang baru untuk mempertahankan struktur lengkap
        if existing_config and module_name in existing_config:
            # Jika konfigurasi baru hanya berisi subset dari konfigurasi lengkap,
            # pastikan untuk mempertahankan struktur lengkap
            if module_name in config:
                # Update konfigurasi yang ada dengan nilai baru
                for key, value in config[module_name].items():
                    existing_config[module_name][key] = value
                # Gunakan konfigurasi gabungan
                config = existing_config
        
        # Simpan konfigurasi lokal terlebih dahulu
        config_path = os.path.join(base_dir, 'configs', f'{module_name}_config.yaml')
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        # Simpan dengan format YAML untuk memastikan struktur tetap utuh
        with open(config_path, 'w') as f:
            yaml.safe_dump(config, f, default_flow_style=False)
        
        # Pastikan Google Drive sudah ter-mount
        if not verify_drive_mounted():
            success = mount_drive_if_needed()
            if not success:
                logger.error("❌ Gagal mount Google Drive untuk sinkronisasi")
                return config
        
        # Sinkronisasi dengan Google Drive
        drive_path = os.path.join('/content/drive/MyDrive/SmartCash/configs')
        os.makedirs(drive_path, exist_ok=True)
        drive_config_path = os.path.join(drive_path, f'{module_name}_config.yaml')
        
        # Salin file ke Google Drive
        shutil.copy2(config_path, drive_config_path)
        
        # Verifikasi file berhasil disalin
        if os.path.exists(drive_config_path):
            # Baca kembali konfigurasi dari Drive untuk verifikasi
            try:
                with open(drive_config_path, 'r') as f:
                    drive_config = yaml.safe_load(f)
                    
                # Log informasi verifikasi
                logger.info(f"✅ Konfigurasi {module_name} berhasil disinkronkan ke Google Drive")
                
                # Verifikasi konfigurasi sama dengan yang disimpan
                if drive_config and module_name in drive_config:
                    # Periksa apakah semua kunci ada dan nilainya sama
                    is_consistent = True
                    for key, value in config[module_name].items():
                        if key not in drive_config[module_name] or drive_config[module_name][key] != value:
                            is_consistent = False
                            logger.warning(f"⚠️ Inkonsistensi pada key '{key}' setelah sinkronisasi")
                    
                    if is_consistent:
                        logger.info(f"✅ Verifikasi sinkronisasi berhasil: semua nilai konsisten")
                    
                    return drive_config
            except Exception as e:
                logger.error(f"❌ Error saat membaca konfigurasi dari Drive: {str(e)}")
        
        return config
    except Exception as e:
        from smartcash.common.logger import get_logger
        logger = get_logger(__name__)
        logger.error(f"❌ Error saat sinkronisasi dengan Drive: {str(e)}")
        return config

def get_default_base_dir():
    """Dapatkan direktori base default."""
    if "COLAB_GPU" in os.environ or "COLAB_TPU_ADDR" in os.environ:
        return "/content"
    return str(Path.home() / "SmartCash")

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
