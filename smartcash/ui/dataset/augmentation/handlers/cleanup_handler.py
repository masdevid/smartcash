"""
File: smartcash/ui/dataset/augmentation/handlers/cleanup_handler.py
Deskripsi: Handler untuk pembersihan hasil augmentasi dengan integrasi shared components terbaru
"""

import os
import shutil
from pathlib import Path
from typing import Dict, Any, List
from smartcash.ui.utils.logger_bridge import create_ui_logger_bridge
from smartcash.ui.utils.button_state_manager import get_button_state_manager

def handle_cleanup_button_click(ui_components: Dict[str, Any], button: Any = None) -> None:
    """
    Handler utama untuk tombol cleanup augmentasi dengan shared components.
    
    Args:
        ui_components: Dictionary komponen UI
        button: Button widget yang diklik
    """
    ui_logger = create_ui_logger_bridge(ui_components, "cleanup_handler")
    button_state_manager = get_button_state_manager(ui_components)
    
    # Cek apakah operation bisa dimulai
    can_start, reason = button_state_manager.can_start_operation("cleanup")
    if not can_start:
        ui_logger.warning(f"⚠️ {reason}")
        return
    
    try:
        ui_logger.info("🧹 Mempersiapkan pembersihan hasil augmentasi...")
        
        # Dapatkan direktori yang akan dibersihkan
        cleanup_paths = _get_cleanup_paths(ui_components)
        
        if not cleanup_paths:
            ui_logger.warning("📁 Tidak ada direktori augmentasi yang ditemukan")
            return
        
        # Analisis file yang akan dihapus
        analysis = _analyze_cleanup_files(cleanup_paths, ui_logger)
        
        if analysis['total_files'] == 0:
            ui_logger.info("✨ Direktori sudah bersih, tidak ada file augmentasi")
            return
        
        # Tampilkan konfirmasi dengan shared confirmation dialog
        _show_cleanup_confirmation(ui_components, analysis, ui_logger)
        
    except Exception as e:
        ui_logger.error(f"❌ Error persiapan cleanup: {str(e)}")

def _show_cleanup_confirmation(ui_components: Dict[str, Any], analysis: Dict[str, Any], ui_logger) -> None:
    """Tampilkan dialog konfirmasi menggunakan shared confirmation dialog."""
    from smartcash.ui.components.confirmation_dialog import create_confirmation_dialog
    from IPython.display import display
    
    # Buat pesan konfirmasi dengan detail
    message = _build_confirmation_message(analysis)
    
    def on_confirm(b):
        ui_components['confirmation_area'].clear_output()
        ui_logger.info("✅ Konfirmasi cleanup diterima")
        _execute_cleanup(ui_components, analysis, ui_logger)
    
    def on_cancel(b):
        ui_components['confirmation_area'].clear_output()
        ui_logger.info("❌ Cleanup dibatalkan oleh pengguna")
    
    # Gunakan shared confirmation dialog
    dialog = create_confirmation_dialog(
        title="🧹 Konfirmasi Pembersihan Augmentasi",
        message=message,
        on_confirm=on_confirm,
        on_cancel=on_cancel,
        danger_mode=True
    )
    
    ui_components['confirmation_area'].clear_output()
    with ui_components['confirmation_area']:
        display(dialog)

def _execute_cleanup(ui_components: Dict[str, Any], analysis: Dict[str, Any], ui_logger) -> None:
    """Eksekusi proses cleanup dengan shared button state manager dan progress tracking."""
    button_state_manager = get_button_state_manager(ui_components)
    
    # Gunakan context manager untuk disable semua buttons
    with button_state_manager.operation_context("cleanup"):
        try:
            from tqdm.auto import tqdm
            
            total_files = analysis['total_files']
            paths_to_clean = list(analysis['paths_detail'].keys())
            
            ui_logger.info(f"🚀 Memulai pembersihan {total_files:,} file dari {len(paths_to_clean)} direktori")
            
            # Start progress tracking dengan shared component
            _start_progress(ui_components, f"🗑️ Membersihkan {total_files:,} file...")
            
            # Cleanup dengan progress
            deleted_count = 0
            error_count = 0
            aug_patterns = ['aug_', '_augmented', '_modified', '_processed']
            
            with tqdm(total=total_files, desc="🗑️ Cleanup", unit="file", colour="red") as pbar:
                for i, path in enumerate(paths_to_clean):
                    try:
                        # Update progress
                        progress_percent = int((i / len(paths_to_clean)) * 80) + 10  # 10-90%
                        _update_progress(ui_components, progress_percent, f"Membersihkan {Path(path).name}...")
                        
                        result = _cleanup_single_directory(path, aug_patterns, pbar)
                        deleted_count += result['deleted']
                        error_count += result['errors']
                        
                    except Exception as e:
                        ui_logger.warning(f"⚠️ Error cleanup {path}: {str(e)}")
                        error_count += 1
            
            # Cleanup empty directories
            _update_progress(ui_components, 95, "🧹 Membersihkan direktori kosong...")
            _cleanup_empty_directories(paths_to_clean, ui_logger)
            
            # Report hasil dan complete progress
            _report_cleanup_results(ui_components, deleted_count, error_count, analysis, ui_logger)
            _complete_progress(ui_components, f"🎉 Cleanup selesai: {deleted_count:,} file dihapus!")
            
        except Exception as e:
            ui_logger.error(f"❌ Error saat cleanup: {str(e)}")
            _error_progress(ui_components, f"❌ Error cleanup: {str(e)}")

def _get_cleanup_paths(ui_components: Dict[str, Any]) -> List[str]:
    """Dapatkan daftar path yang akan dibersihkan."""
    paths = []
    
    # Path dari UI components
    if 'output_dir' in ui_components and hasattr(ui_components['output_dir'], 'value'):
        paths.append(ui_components['output_dir'].value)
    
    # Path dari config
    config = ui_components.get('config', {})
    augmentation_config = config.get('augmentation', {})
    
    if 'output_dir' in augmentation_config:
        paths.append(augmentation_config['output_dir'])
    
    # Default paths
    default_paths = ['data/augmented', '/content/data/augmented']
    paths.extend(default_paths)
    
    # Filter path yang ada dan unique
    existing_paths = []
    for path in paths:
        if path and os.path.exists(path) and path not in existing_paths:
            existing_paths.append(path)
    
    return existing_paths

def _analyze_cleanup_files(paths: List[str], ui_logger) -> Dict[str, Any]:
    """Analisis file yang akan dihapus dengan detail."""
    analysis = {'total_files': 0, 'total_size_mb': 0, 'paths_detail': {}, 'file_types': {}}
    aug_patterns = ['aug_', '_augmented', '_modified', '_processed']
    
    for path in paths:
        path_detail = {'files': 0, 'size_mb': 0}
        
        try:
            for root, dirs, files in os.walk(path):
                for file in files:
                    if any(pattern in file.lower() for pattern in aug_patterns):
                        file_path = os.path.join(root, file)
                        try:
                            file_size = os.path.getsize(file_path)
                            path_detail['files'] += 1
                            path_detail['size_mb'] += file_size / (1024 * 1024)
                            analysis['total_files'] += 1
                            analysis['total_size_mb'] += file_size / (1024 * 1024)
                            
                            ext = Path(file).suffix.lower()
                            analysis['file_types'][ext] = analysis['file_types'].get(ext, 0) + 1
                        except OSError:
                            continue
        
        except Exception as e:
            ui_logger.warning(f"⚠️ Error analisis {path}: {str(e)}")
            continue
        
        if path_detail['files'] > 0:
            analysis['paths_detail'][path] = path_detail
    
    return analysis

def _build_confirmation_message(analysis: Dict[str, Any]) -> str:
    """Build pesan konfirmasi dari analisis."""
    total_files = analysis['total_files']
    total_size = analysis['total_size_mb']
    paths_count = len(analysis['paths_detail'])
    
    message = f"📊 **Detail Pembersihan:**\n"
    message += f"• **{total_files:,} file** augmentasi akan dihapus\n"
    message += f"• **{total_size:.1f} MB** ruang disk akan dibebaskan\n"
    message += f"• **{paths_count} direktori** akan dibersihkan\n\n"
    
    # Detail per direktori (max 3)
    shown_paths = list(analysis['paths_detail'].keys())[:3]
    for path in shown_paths:
        detail = analysis['paths_detail'][path]
        message += f"📁 `{path}`: {detail['files']} files ({detail['size_mb']:.1f} MB)\n"
    
    if len(analysis['paths_detail']) > 3:
        remaining = len(analysis['paths_detail']) - 3
        message += f"... dan {remaining} direktori lainnya\n"
    
    message += f"\n⚠️ **Tindakan ini tidak dapat dibatalkan!**\n"
    message += f"Lanjutkan pembersihan?"
    
    return message

def _cleanup_single_directory(path: str, patterns: List[str], pbar) -> Dict[str, int]:
    """Cleanup file dalam satu direktori."""
    result = {'deleted': 0, 'errors': 0}
    
    for root, dirs, files in os.walk(path, topdown=False):
        for file in files:
            if any(pattern in file.lower() for pattern in patterns):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    result['deleted'] += 1
                    pbar.update(1)
                except OSError:
                    result['errors'] += 1
                    pbar.update(1)
    
    return result

def _cleanup_empty_directories(paths: List[str], ui_logger) -> None:
    """Hapus direktori kosong setelah cleanup."""
    for path in paths:
        try:
            for root, dirs, files in os.walk(path, topdown=False):
                for dir_name in dirs:
                    dir_path = os.path.join(root, dir_name)
                    try:
                        if not os.listdir(dir_path):
                            os.rmdir(dir_path)
                    except OSError:
                        pass
        except Exception:
            pass

def _report_cleanup_results(ui_components: Dict[str, Any], deleted: int, errors: int, 
                           analysis: Dict[str, Any], ui_logger) -> None:
    """Report hasil cleanup dengan detail."""
    total_expected = analysis['total_files']
    freed_space = analysis['total_size_mb']
    
    if deleted == total_expected and errors == 0:
        ui_logger.success(f"🎉 Cleanup berhasil sempurna!")
        ui_logger.info(f"✅ {deleted:,} file dihapus, {freed_space:.1f} MB dibebaskan")
    elif deleted > 0:
        ui_logger.success(f"✅ Cleanup selesai dengan peringatan")
        ui_logger.info(f"📊 {deleted:,}/{total_expected:,} file dihapus")
        if errors > 0:
            ui_logger.warning(f"⚠️ {errors} file gagal dihapus")
    else:
        ui_logger.error(f"❌ Cleanup gagal - tidak ada file yang dihapus")

# Shared progress tracking integration functions
def _start_progress(ui_components: Dict[str, Any], message: str):
    """Start progress menggunakan shared progress tracking."""
    if 'show_for_operation' in ui_components and callable(ui_components['show_for_operation']):
        ui_components['show_for_operation']('cleanup')
    elif 'tracker' in ui_components:
        ui_components['tracker'].show('cleanup')
    
    _update_progress(ui_components, 0, message)

def _update_progress(ui_components: Dict[str, Any], value: int, message: str):
    """Update progress menggunakan shared progress tracking."""
    if 'update_progress' in ui_components and callable(ui_components['update_progress']):
        ui_components['update_progress']('overall', value, message)
    elif 'tracker' in ui_components:
        ui_components['tracker'].update('overall', value, message)

def _complete_progress(ui_components: Dict[str, Any], message: str):
    """Complete progress menggunakan shared progress tracking."""
    if 'complete_operation' in ui_components and callable(ui_components['complete_operation']):
        ui_components['complete_operation'](message)
    elif 'tracker' in ui_components:
        ui_components['tracker'].complete(message)

def _error_progress(ui_components: Dict[str, Any], message: str):
    """Error progress menggunakan shared progress tracking."""
    if 'error_operation' in ui_components and callable(ui_components['error_operation']):
        ui_components['error_operation'](message)
    elif 'tracker' in ui_components:
        ui_components['tracker'].error(message)