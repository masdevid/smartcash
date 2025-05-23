"""
File: smartcash/ui/dataset/download/handlers/reset_action.py
Deskripsi: Handler aksi reset form download
"""

from typing import Dict, Any
from smartcash.ui.dataset.download.utils.form_resetter import reset_form_fields
from smartcash.ui.dataset.download.utils.progress_updater import reset_progress

def execute_reset_action(ui_components: Dict[str, Any], button: Any = None) -> None:
    """Eksekusi reset form."""
    logger = ui_components.get('logger')
    if logger:
        logger.info("🔄 Mereset form download")
    
    try:
        # Reset form fields
        reset_form_fields(ui_components)
        
        # Reset progress
        reset_progress(ui_components)
        
        # Clear confirmation area
        if 'confirmation_area' in ui_components and hasattr(ui_components['confirmation_area'], 'clear_output'):
            ui_components['confirmation_area'].clear_output()
        
        if logger:
            logger.success("✅ Form berhasil direset")
            
    except Exception as e:
        if logger:
            logger.error(f"❌ Error saat reset: {str(e)}")

"""
File: smartcash/ui/dataset/download/handlers/cleanup_action.py  
Deskripsi: Handler aksi cleanup dataset
"""

import shutil
from pathlib import Path
from smartcash.ui.dataset.download.utils.confirmation_dialog import show_cleanup_confirmation
from smartcash.ui.dataset.download.utils.button_state import disable_download_buttons

def execute_cleanup_action(ui_components: Dict[str, Any], button: Any = None) -> None:
    """Eksekusi cleanup dataset."""
    logger = ui_components.get('logger')
    if logger:
        logger.info("🧹 Memulai cleanup dataset")
    
    disable_download_buttons(ui_components, True)
    
    try:
        # Dapatkan output directory
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
    """Eksekusi cleanup setelah konfirmasi."""
    from smartcash.ui.dataset.download.utils.progress_updater import show_progress, update_progress
    
    logger = ui_components.get('logger')
    output_path = Path(output_dir)
    
    try:
        show_progress(ui_components, "Menghapus dataset...")
        
        if logger:
            logger.info(f"🗑️ Menghapus dataset: {output_dir}")
        
        # Hapus direktori
        if output_path.exists():
            update_progress(ui_components, 50, "Menghapus file...")
            shutil.rmtree(output_path, ignore_errors=True)
            update_progress(ui_components, 100, "Cleanup selesai")
            
            if logger:
                logger.success("✅ Dataset berhasil dihapus")
        
    except Exception as e:
        if logger:
            logger.error(f"❌ Error cleanup: {str(e)}")
    finally:
        disable_download_buttons(ui_components, False)