"""
File: smartcash/ui/dataset/download/handlers/cleanup_action.py
Deskripsi: Updated cleanup action dengan observer progress
"""

import shutil
from pathlib import Path
from typing import Dict, Any
from smartcash.ui.dataset.download.utils.confirmation_dialog import show_cleanup_confirmation
from smartcash.ui.dataset.download.utils.button_state import disable_download_buttons
from smartcash.components.observer import notify, EventTopics, EventDispatcher

def execute_cleanup_action(ui_components: Dict[str, Any], button: Any = None) -> None:
    """Eksekusi cleanup dataset dengan observer progress."""
    logger = ui_components.get('logger')
    if logger:
        logger.info("🧹 Memulai cleanup dataset")
    
    disable_download_buttons(ui_components, True)
    
    try:
        output_dir = ui_components.get('output_dir', {}).value or 'data'
        output_path = Path(output_dir)
        
        if not output_path.exists():
            if logger:
                logger.warning(f"⚠️ Direktori tidak ditemukan: {output_dir}")
            disable_download_buttons(ui_components, False)
            return
        
        # Hitung file untuk konfirmasi
        total_files = sum(1 for _ in output_path.rglob('*') if _.is_file())
        
        if total_files == 0:
            if logger:
                logger.info("ℹ️ Tidak ada file untuk dihapus")
            disable_download_buttons(ui_components, False)
            return
        
        # Tampilkan konfirmasi
        show_cleanup_confirmation(ui_components, output_dir, total_files)
        
    except Exception as e:
        if logger:
            logger.error(f"❌ Error persiapan cleanup: {str(e)}")
        disable_download_buttons(ui_components, False)

def execute_cleanup_confirmed(ui_components: Dict[str, Any], output_dir: str) -> None:
    """Eksekusi cleanup setelah konfirmasi dengan observer progress."""
    logger = ui_components.get('logger')
    output_path = Path(output_dir)
    
    try:
        # Notify start via observer
        notify('DOWNLOAD_START', ui_components, 
               message="Memulai cleanup dataset", namespace="cleanup")
        
        if logger:
            logger.info(f"🗑️ Menghapus dataset: {output_dir}")
        
        # Progress updates via observer
        notify('DOWNLOAD_PROGRESS', ui_components,
               progress=25, message="Menghapus file...", namespace="cleanup")
        
        if output_path.exists():
            shutil.rmtree(output_path, ignore_errors=True)
        
        notify('DOWNLOAD_PROGRESS', ui_components,
               progress=75, message="Membersihkan direktori...", namespace="cleanup")
            
        # Verify deletion
        if not output_path.exists():
            notify('DOWNLOAD_COMPLETE', ui_components,
                   message="Cleanup berhasil", namespace="cleanup")
            
            if logger:
                logger.success("✅ Dataset berhasil dihapus")
        else:
            raise Exception("Gagal menghapus direktori")
        
    except Exception as e:
        notify('DOWNLOAD_ERROR', ui_components,
               message=f"Cleanup gagal: {str(e)}", namespace="cleanup")
        if logger:
            logger.error(f"❌ Error cleanup: {str(e)}")
    finally:
        disable_download_buttons(ui_components, False)