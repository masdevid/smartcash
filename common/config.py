"""
File: smartcash/common/config.py
Deskripsi: Patch untuk ConfigManager dengan sinkronisasi Drive yang lebih kuat
"""

# Tambahkan fungsi ini ke class ConfigManager

def sync_with_drive_enhanced(self, config_file: str, 
                            sync_strategy: str = 'drive_priority',
                            backup: bool = True) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Sinkronisasi file konfigurasi dengan Google Drive menggunakan strategi yang ditingkatkan.
    
    Args:
        config_file: Nama file konfigurasi (tanpa path)
        sync_strategy: Strategi sinkronisasi:
            - 'drive_priority': Config Drive menang (default)
            - 'local_priority': Config lokal menang
            - 'newest': Config terbaru menang
            - 'merge': Gabungkan config dengan strategi smart
        backup: Buat backup sebelum update
        
    Returns:
        Tuple (success, message, merged_config)
    """
    try:
        # Import sinkronisasi yang ditingkatkan
        from smartcash.common.config_sync import sync_config_with_drive
        
        # Mendapatkan logger
        logger = None
        try:
            from smartcash.common.logger import get_logger
            logger = get_logger("config_manager")
        except ImportError:
            pass
        
        # Panggil fungsi sinkronisasi yang ditingkatkan
        success, message, merged_config = sync_config_with_drive(
            config_file=config_file,
            sync_strategy=sync_strategy,
            create_backup=backup,
            logger=logger
        )
        
        # Update config saat ini jika sukses
        if success and merged_config:
            self.config = merged_config
            
        return success, message, merged_config
        
    except Exception as e:
        error_msg = f"❌ Error saat sinkronisasi konfigurasi: {str(e)}"
        if hasattr(self, '_logger') and self._logger:
            self._logger.error(error_msg)
        return False, error_msg, {}

def use_drive_as_source_of_truth(self) -> bool:
    """
    Sinkronisasi semua konfigurasi dengan Drive sebagai sumber kebenaran.
    
    Returns:
        Boolean menunjukkan keberhasilan operasi
    """
    try:
        # Import fungsi sinkronisasi yang ditingkatkan
        from smartcash.common.config_sync import sync_all_configs
        
        # Mendapatkan logger
        logger = None
        try:
            from smartcash.common.logger import get_logger
            logger = get_logger("config_manager")
        except ImportError:
            pass
        
        # Sinkronisasi semua konfigurasi dengan Drive sebagai prioritas
        results = sync_all_configs(
            sync_strategy='drive_priority',
            create_backup=True,
            logger=logger
        )
        
        # Muat ulang konfigurasi utama setelah sinkronisasi
        if "success" in results and results["success"]:
            # Reload config dari file utama jika ada
            for result in results["success"]:
                if result["file"] == "base_config.yaml":
                    self.load_config("base_config.yaml")
                    break
        
        # Hitung jumlah sukses dan gagal
        success_count = len(results.get("success", []))
        failure_count = len(results.get("failure", []))
        
        # Log hasil operasi
        if logger:
            logger.info(f"🔄 Sinkronisasi selesai: {success_count} berhasil, {failure_count} gagal")
            
            # Log detail jika ada kegagalan
            if failure_count > 0:
                for failure in results.get("failure", []):
                    logger.warning(f"⚠️ Gagal sinkronisasi {failure.get('file', 'unknown')}: {failure.get('message', 'unknown error')}")
        
        # Operasi dianggap sukses jika tidak ada kegagalan
        return failure_count == 0
        
    except Exception as e:
        error_msg = f"❌ Error saat menggunakan Drive sebagai sumber kebenaran: {str(e)}"
        if hasattr(self, '_logger') and self._logger:
            self._logger.error(error_msg)
        return False

# Tambahkan juga fungsi ini untuk mempermudah akses

def get_drive_config_path(config_file: str = None) -> Optional[str]:
    """
    Dapatkan path konfigurasi di Google Drive.
    
    Args:
        config_file: Nama file konfigurasi (opsional)
        
    Returns:
        Path konfigurasi di Drive atau None jika Drive tidak terpasang
    """
    try:
        from smartcash.common.environment import get_environment_manager
        env_manager = get_environment_manager()
        
        if not env_manager.is_drive_mounted:
            return None
            
        drive_configs_dir = env_manager.drive_path / 'configs'
        
        if config_file:
            return str(drive_configs_dir / config_file)
        else:
            return str(drive_configs_dir)
            
    except Exception:
        return None