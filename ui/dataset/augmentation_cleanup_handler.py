"""
File: smartcash/ui/dataset/augmentation_cleanup_handler.py
Deskripsi: Handler untuk membersihkan data augmentasi dengan strategi kosongkan folder
"""

from typing import Dict, Any
from IPython.display import display, clear_output
import shutil, os, glob
from pathlib import Path
from tqdm.auto import tqdm

from smartcash.ui.utils.constants import ICONS
from smartcash.ui.utils.alert_utils import create_status_indicator, create_info_alert

def setup_cleanup_handler(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """Setup handler untuk tombol cleanup augmentasi data."""
    logger = ui_components.get('logger')
    
    def perform_cleanup():
        """Bersihkan folder augmentasi dengan progress tracking."""
        augmented_dir = ui_components.get('augmented_dir', 'data/augmented')
        
        try:
            # Update status dan tampilkan progress bar
            with ui_components['status']:
                clear_output(wait=True)
                progress_bar = tqdm(total=3, desc="🧹 Membersihkan Augmentasi", ncols=100)
                display(create_status_indicator("info", f"{ICONS['trash']} Mempersiapkan pembersihan..."))
                
            # Pastikan direktori ada dan berisi subdirektori images/labels
            path = Path(augmented_dir)
            if not path.exists():
                with ui_components['status']:
                    display(create_status_indicator("warning", f"{ICONS['warning']} Direktori tidak ditemukan: {augmented_dir}"))
                return
            
            # Update progress
            progress_bar.update(1)
            with ui_components['status']:
                display(create_status_indicator("info", f"{ICONS['processing']} Menghapus gambar augmentasi..."))
            
            # Hapus isi subdirektori images dan labels
            subdirs = ['images', 'labels']
            total_files_deleted = 0
            
            for subdir in subdirs:
                dir_path = path / subdir
                if dir_path.exists():
                    # Hapus file dengan pola augmentasi
                    aug_files = list(glob.glob(str(dir_path / 'aug_*')) + 
                                     glob.glob(str(dir_path / '*_augmented*')))
                    
                    for file_path in aug_files:
                        try:
                            os.remove(file_path)
                            total_files_deleted += 1
                        except Exception as e:
                            logger.warning(f"⚠️ Gagal menghapus {file_path}: {e}")
            
            # Update progress
            progress_bar.update(1)
            with ui_components['status']:
                display(create_status_indicator("info", f"{ICONS['processing']} Membersihkan metadata..."))
            
            # Hapus file-file metadata tambahan jika ada
            metadata_files = list(path.glob('*.json')) + list(path.glob('*.yaml'))
            for metadata_file in metadata_files:
                try:
                    metadata_file.unlink()
                except Exception as e:
                    logger.warning(f"⚠️ Gagal menghapus metadata: {e}")
            
            # Update progress akhir
            progress_bar.update(1)
            
            # Tutup progress bar
            progress_bar.close()
            
            # Tampilkan status sukses
            with ui_components['status']:
                clear_output(wait=True)
                display(create_status_indicator("success", 
                    f"{ICONS['success']} Folder augmentasi dibersihkan. {total_files_deleted} file dihapus"))
            
            # Reset UI
            if 'visualization_buttons' in ui_components:
                ui_components['visualization_buttons'].layout.display = 'none'
            if 'visualization_container' in ui_components:
                ui_components['visualization_container'].layout.display = 'none'
            if 'summary_container' in ui_components:
                ui_components['summary_container'].layout.display = 'none'
            if 'cleanup_button' in ui_components:
                ui_components['cleanup_button'].layout.display = 'none'
            
        except Exception as e:
            with ui_components['status']:
                clear_output(wait=True)
                display(create_status_indicator("error", f"{ICONS['error']} Gagal membersihkan: {str(e)}"))
            
            if logger:
                logger.error(f"❌ Error saat membersihkan data: {str(e)}")
    
    # Daftarkan handler untuk tombol cleanup
    ui_components['cleanup_button'].on_click(lambda b: perform_cleanup())
    
    return ui_components