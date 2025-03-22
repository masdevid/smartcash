"""
File: smartcash/ui/setup/drive_sync_initializer.py
Deskripsi: Modul untuk inisialisasi sinkronisasi Drive dengan output logging yang ditingkatkan dan perbaikan progress tracking
"""

from typing import Dict, Any, Optional
from IPython.display import display, HTML

def initialize_drive_sync(ui_components: Dict[str, Any]):
    """
    Inisialisasi dan sinkronisasi Google Drive untuk proyek dengan integrasi logging UI yang ditingkatkan.
    
    Args:
        ui_components: Dictionary berisi komponen UI
    """
    logger = ui_components.get('logger')
    
    # Update progress bar dan pesan jika tersedia
    if 'progress_bar' in ui_components and 'progress_message' in ui_components:
        ui_components['progress_bar'].value = 0
        ui_components['progress_message'].value = "Memeriksa Google Drive..."
        ui_components['progress_bar'].layout.visibility = 'visible'
        ui_components['progress_message'].layout.visibility = 'visible'
    
    # Cek apakah environment sudah terdeteksi
    try:
        from smartcash.ui.utils.drive_utils import detect_drive_mount
        is_mounted, drive_path = detect_drive_mount()
        
        if logger:
            logger.info(f"🔍 Status Google Drive: {'terhubung' if is_mounted else 'tidak terhubung'}")
        
        # Update progress components
        if 'progress_bar' in ui_components and 'progress_message' in ui_components:
            ui_components['progress_bar'].value = 1
            ui_components['progress_message'].value = "Status Drive: " + ('terhubung' if is_mounted else 'tidak terhubung')
        
        # Jika Drive terhubung, coba sinkronisasi konfigurasi
        if is_mounted and drive_path:
            # Update progress components
            if 'progress_bar' in ui_components and 'progress_message' in ui_components:
                ui_components['progress_bar'].value = 1
                ui_components['progress_message'].value = "Menyinkronkan konfigurasi dengan Google Drive..."
            
            # Coba sinkronisasi konfigurasi
            try:
                from smartcash.common.config import get_config_manager
                config_manager = get_config_manager()
                
                # Langkah 1: Coba sync_with_drive_enhanced
                try:
                    success, message, _ = config_manager.sync_with_drive_enhanced(
                        "configs/base_config.yaml", 
                        sync_strategy='merge',
                        backup=True
                    )
                    
                    if logger:
                        if success:
                            logger.info(f"✅ Sinkronisasi konfigurasi berhasil: {message}")
                        else:
                            logger.warning(f"⚠️ Sinkronisasi konfigurasi: {message}")
                            
                    # Update progress
                    if 'progress_bar' in ui_components and 'progress_message' in ui_components:
                        ui_components['progress_bar'].value = 2
                        ui_components['progress_message'].value = f"Sinkronisasi base_config: {'berhasil' if success else 'sebagian'}"
                except Exception as e:
                    if logger:
                        logger.warning(f"⚠️ Error saat sync_with_drive_enhanced: {str(e)}")
                    # Update progress
                    if 'progress_bar' in ui_components and 'progress_message' in ui_components:
                        ui_components['progress_bar'].value = 2
                        ui_components['progress_message'].value = "Sinkronisasi dasar: gagal"
                    
                # Langkah 2: Coba use_drive_as_source_of_truth
                try:
                    # Sinkronisasi semua konfigurasi
                    success = config_manager.use_drive_as_source_of_truth()
                    
                    if logger:
                        if success:
                            logger.info("✅ Sinkronisasi semua konfigurasi berhasil")
                        else:
                            logger.warning("⚠️ Sinkronisasi semua konfigurasi berhasil sebagian")
                    
                    # Update progress
                    if 'progress_bar' in ui_components and 'progress_message' in ui_components:
                        ui_components['progress_bar'].value = 3
                        ui_components['progress_message'].value = f"Sinkronisasi semua konfigurasi: {'berhasil' if success else 'sebagian'}"
                except Exception as e:
                    if logger:
                        logger.warning(f"⚠️ Error saat use_drive_as_source_of_truth: {str(e)}")
                    # Update progress
                    if 'progress_bar' in ui_components and 'progress_message' in ui_components:
                        ui_components['progress_bar'].value = 3
                        ui_components['progress_message'].value = "Sinkronisasi semua: gagal"
            
            except ImportError as e:
                if logger:
                    logger.warning(f"⚠️ ConfigManager tidak tersedia: {str(e)}")
                
                # Update progress
                if 'progress_bar' in ui_components and 'progress_message' in ui_components:
                    ui_components['progress_bar'].value = 3
                    ui_components['progress_message'].value = "Sinkronisasi tidak tersedia"
            
            # Kembalikan UI ke status normal
            if 'progress_bar' in ui_components and 'progress_message' in ui_components:
                ui_components['progress_bar'].value = 3
                ui_components['progress_message'].value = "Sinkronisasi selesai"
        else:
            # Drive tidak terhubung, perbarui UI
            if 'progress_bar' in ui_components and 'progress_message' in ui_components:
                ui_components['progress_bar'].value = 3
                ui_components['progress_message'].value = "Google Drive tidak terhubung"
        
        # Notifikasi observer jika ada
        notify_drive_sync_status(ui_components, is_mounted, drive_path)
        
    except ImportError as e:
        if logger:
            logger.warning(f"⚠️ Error saat initialize_drive_sync: {str(e)}")
        
        # Update progress jika terjadi error
        if 'progress_bar' in ui_components and 'progress_message' in ui_components:
            ui_components['progress_bar'].value = 3
            ui_components['progress_message'].value = "Error: Drive utils tidak tersedia"
        
        # Notifikasi observer jika ada
        notify_drive_sync_status(ui_components, False, None, str(e))

def notify_drive_sync_status(ui_components: Dict[str, Any], is_mounted: bool, drive_path: Optional[str], error: str = None):
    """
    Notifikasi observer tentang status Drive.
    
    Args:
        ui_components: Dictionary berisi komponen UI
        is_mounted: Apakah Drive terhubung
        drive_path: Path Google Drive jika terhubung
        error: Pesan error jika ada
    """
    try:
        if 'observer_manager' in ui_components:
            from smartcash.components.observer import notify
            from smartcash.components.observer.event_topics_observer import EventTopics
            
            if is_mounted:
                notify(
                    event_type=EventTopics.DRIVE_MOUNTED,
                    sender="drive_sync_initializer",
                    message=f"Google Drive terhubung di {drive_path}",
                    drive_path=drive_path
                )
            else:
                event_type = EventTopics.DRIVE_ERROR if error else EventTopics.DRIVE_NOT_MOUNTED
                notify(
                    event_type=event_type,
                    sender="drive_sync_initializer",
                    message=f"Google Drive tidak terhubung{': ' + error if error else ''}",
                    error=error
                )
    except (ImportError, AttributeError, Exception) as e:
        logger = ui_components.get('logger')
        if logger:
            logger.debug(f"⚠️ Error saat notify_drive_sync_status: {str(e)}")