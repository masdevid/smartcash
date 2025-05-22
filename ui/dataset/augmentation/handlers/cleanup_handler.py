"""
File: smartcash/ui/dataset/augmentation/handlers/cleanup_handler.py
Deskripsi: Handler untuk pembersihan hasil augmentasi dengan konfirmasi dan progress tracking
"""

import os
import shutil
from pathlib import Path
from typing import Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor

from smartcash.ui.utils.logger_bridge import create_ui_logger_bridge
from smartcash.ui.dataset.augmentation.handlers.state_handler import StateHandler
from smartcash.common.threadpools import process_with_stats

def handle_cleanup_button_click(ui_components: Dict[str, Any], button: Any = None) -> None:
    """
    Handler utama untuk tombol cleanup augmentasi.
    
    Args:
        ui_components: Dictionary komponen UI
        button: Button widget yang diklik
    """
    ui_logger = create_ui_logger_bridge(ui_components, "cleanup_handler")
    state_handler = StateHandler(ui_components, ui_logger)
    
    # Cek apakah sedang berjalan proses lain
    if state_handler.is_running():
        ui_logger.warning("⚠️ Proses augmentasi sedang berjalan, cleanup tidak dapat dilakukan")
        return
    
    # Disable button selama proses
    if button and hasattr(button, 'disabled'):
        button.disabled = True
    
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
        
        # Tampilkan konfirmasi dengan detail
        _show_cleanup_confirmation(ui_components, analysis, ui_logger)
        
    except Exception as e:
        ui_logger.error(f"❌ Error persiapan cleanup: {str(e)}")
    finally:
        if button and hasattr(button, 'disabled'):
            button.disabled = False

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
    default_paths = [
        'data/augmented',
        '/content/data/augmented'  # Colab path
    ]
    paths.extend(default_paths)
    
    # Filter path yang ada dan unique
    existing_paths = []
    for path in paths:
        if path and os.path.exists(path) and path not in existing_paths:
            existing_paths.append(path)
    
    return existing_paths

def _analyze_cleanup_files(paths: List[str], ui_logger) -> Dict[str, Any]:
    """Analisis file yang akan dihapus dengan detail."""
    analysis = {
        'total_files': 0,
        'total_size_mb': 0,
        'paths_detail': {},
        'file_types': {},
        'augmented_patterns': []
    }
    
    # Pattern file augmentasi
    aug_patterns = ['aug_', '_augmented', '_modified', '_processed']
    
    for path in paths:
        path_obj = Path(path)
        path_detail = {
            'files': 0,
            'size_mb': 0,
            'subdirs': []
        }
        
        try:
            for root, dirs, files in os.walk(path):
                for file in files:
                    # Filter hanya file augmentasi
                    if any(pattern in file.lower() for pattern in aug_patterns):
                        file_path = os.path.join(root, file)
                        try:
                            file_size = os.path.getsize(file_path)
                            path_detail['files'] += 1
                            path_detail['size_mb'] += file_size / (1024 * 1024)
                            analysis['total_files'] += 1
                            analysis['total_size_mb'] += file_size / (1024 * 1024)
                            
                            # Analisis tipe file
                            ext = Path(file).suffix.lower()
                            analysis['file_types'][ext] = analysis['file_types'].get(ext, 0) + 1
                            
                            # Analisis pattern
                            for pattern in aug_patterns:
                                if pattern in file.lower():
                                    if pattern not in analysis['augmented_patterns']:
                                        analysis['augmented_patterns'].append(pattern)
                        except OSError:
                            continue
                
                # Track subdirectories
                path_detail['subdirs'] = dirs
        
        except Exception as e:
            ui_logger.warning(f"⚠️ Error analisis {path}: {str(e)}")
            continue
        
        if path_detail['files'] > 0:
            analysis['paths_detail'][path] = path_detail
    
    return analysis

def _show_cleanup_confirmation(ui_components: Dict[str, Any], analysis: Dict[str, Any], ui_logger) -> None:
    """Tampilkan dialog konfirmasi dengan detail analisis."""
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
    
    # Ensure confirmation area exists
    _ensure_confirmation_area(ui_components)
    
    dialog = create_confirmation_dialog(
        title="🧹 Konfirmasi Pembersihan Augmentasi",
        message=message,
        on_confirm=on_confirm,
        on_cancel=on_cancel
    )
    
    ui_components['confirmation_area'].clear_output()
    with ui_components['confirmation_area']:
        display(dialog)

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
    
    # Pattern yang akan dihapus
    if analysis['augmented_patterns']:
        patterns_str = ', '.join(f"`{p}`" for p in analysis['augmented_patterns'])
        message += f"\n🎯 **Pattern:** {patterns_str}\n"
    
    # File types
    if analysis['file_types']:
        types_list = [f"{ext}({count})" for ext, count in list(analysis['file_types'].items())[:5]]
        message += f"📄 **Tipe file:** {', '.join(types_list)}\n"
    
    message += f"\n⚠️ **Tindakan ini tidak dapat dibatalkan!**\n"
    message += f"Lanjutkan pembersihan?"
    
    return message

def _execute_cleanup(ui_components: Dict[str, Any], analysis: Dict[str, Any], ui_logger) -> None:
    """Eksekusi proses cleanup dengan progress tracking."""
    from tqdm.auto import tqdm
    
    total_files = analysis['total_files']
    paths_to_clean = list(analysis['paths_detail'].keys())
    
    ui_logger.info(f"🚀 Memulai pembersihan {total_files:,} file dari {len(paths_to_clean)} direktori")
    
    try:
        # Setup progress tracking
        _setup_cleanup_progress(ui_components, total_files)
        
        # Cleanup dengan progress
        deleted_count = 0
        error_count = 0
        
        # Pattern file augmentasi
        aug_patterns = ['aug_', '_augmented', '_modified', '_processed']
        
        with tqdm(total=total_files, desc="🗑️ Cleanup", unit="file", colour="red") as pbar:
            for path in paths_to_clean:
                try:
                    result = _cleanup_single_directory(path, aug_patterns, pbar)
                    deleted_count += result['deleted']
                    error_count += result['errors']
                    
                    # Update progress UI
                    _update_cleanup_progress(ui_components, deleted_count, total_files)
                    
                except Exception as e:
                    ui_logger.warning(f"⚠️ Error cleanup {path}: {str(e)}")
                    error_count += 1
        
        # Cleanup empty directories
        _cleanup_empty_directories(paths_to_clean, ui_logger)
        
        # Report hasil
        _report_cleanup_results(ui_components, deleted_count, error_count, analysis, ui_logger)
        
    except Exception as e:
        ui_logger.error(f"❌ Error saat cleanup: {str(e)}")
    finally:
        _hide_cleanup_progress(ui_components)

def _cleanup_single_directory(path: str, patterns: List[str], pbar) -> Dict[str, int]:
    """Cleanup file dalam satu direktori."""
    result = {'deleted': 0, 'errors': 0}
    
    for root, dirs, files in os.walk(path, topdown=False):
        for file in files:
            # Filter hanya file augmentasi
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
                        if not os.listdir(dir_path):  # Directory kosong
                            os.rmdir(dir_path)
                    except OSError:
                        pass  # Ignore errors untuk empty directory cleanup
        except Exception:
            pass

def _setup_cleanup_progress(ui_components: Dict[str, Any], total_files: int) -> None:
    """Setup progress tracking untuk cleanup."""
    if 'progress_container' in ui_components:
        container = ui_components['progress_container']
        if hasattr(container, 'layout'):
            container.layout.display = 'block'
    
    if 'progress_bar' in ui_components:
        progress_bar = ui_components['progress_bar']
        if hasattr(progress_bar, 'max'):
            progress_bar.max = total_files
            progress_bar.value = 0
            progress_bar.description = "Cleanup: 0%"

def _update_cleanup_progress(ui_components: Dict[str, Any], current: int, total: int) -> None:
    """Update progress UI cleanup."""
    if total == 0:
        return
    
    percentage = int((current / total) * 100)
    
    if 'progress_bar' in ui_components:
        progress_bar = ui_components['progress_bar']
        if hasattr(progress_bar, 'value'):
            progress_bar.value = current
            progress_bar.description = f"Cleanup: {percentage}%"
    
    # Update progress message
    message = f"Menghapus file: {current:,}/{total:,} ({percentage}%)"
    for label_key in ['progress_message', 'step_label']:
        if label_key in ui_components:
            label = ui_components[label_key]
            if hasattr(label, 'value'):
                label.value = message

def _hide_cleanup_progress(ui_components: Dict[str, Any]) -> None:
    """Sembunyikan progress UI setelah cleanup."""
    if 'progress_container' in ui_components:
        container = ui_components['progress_container']
        if hasattr(container, 'layout'):
            container.layout.display = 'none'

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
    
    # Update status panel
    if deleted > 0:
        _update_status_panel(ui_components, 
                           f"✅ Cleanup selesai: {deleted:,} file dihapus", 
                           "success")
    else:
        _update_status_panel(ui_components, 
                           "❌ Cleanup gagal", 
                           "error")

def _ensure_confirmation_area(ui_components: Dict[str, Any]) -> None:
    """Pastikan confirmation area tersedia."""
    if 'confirmation_area' not in ui_components:
        from ipywidgets import Output
        ui_components['confirmation_area'] = Output()

def _update_status_panel(ui_components: Dict[str, Any], message: str, status: str) -> None:
    """Update status panel jika tersedia."""
    if 'status_panel' in ui_components:
        try:
            from smartcash.ui.utils.alert_utils import update_status_panel
            update_status_panel(ui_components['status_panel'], message, status)
        except ImportError:
            pass  # Status panel tidak tersedia

# Utility functions untuk testing dan debugging
def get_cleanup_analysis(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Dapatkan analisis cleanup tanpa eksekusi (untuk testing).
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary analisis cleanup
    """
    cleanup_paths = _get_cleanup_paths(ui_components)
    ui_logger = create_ui_logger_bridge(ui_components, "cleanup_analysis")
    return _analyze_cleanup_files(cleanup_paths, ui_logger)

def dry_run_cleanup(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simulasi cleanup tanpa eksekusi (dry run).
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary hasil simulasi
    """
    ui_logger = create_ui_logger_bridge(ui_components, "cleanup_dry_run")
    
    cleanup_paths = _get_cleanup_paths(ui_components)
    analysis = _analyze_cleanup_files(cleanup_paths, ui_logger)
    
    ui_logger.info(f"🔍 **Dry Run Cleanup Results:**")
    ui_logger.info(f"📊 {analysis['total_files']:,} file akan dihapus")
    ui_logger.info(f"💾 {analysis['total_size_mb']:.1f} MB akan dibebaskan")
    ui_logger.info(f"📁 {len(analysis['paths_detail'])} direktori akan dibersihkan")
    
    return analysis