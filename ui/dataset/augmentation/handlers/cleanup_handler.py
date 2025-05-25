"""
File: smartcash/ui/dataset/augmentation/handlers/cleanup_handler.py
Deskripsi: Handler untuk pembersihan hasil augmentasi dengan null safety
"""

from typing import Dict, Any
from smartcash.ui.utils.logger_bridge import create_ui_logger_bridge
from smartcash.ui.utils.button_state_manager import get_button_state_manager, ensure_button_state_manager

def handle_cleanup_button_click(ui_components: Dict[str, Any], button: Any = None) -> None:
    """Handler cleanup dengan robust null safety."""
    ui_logger = create_ui_logger_bridge(ui_components, "cleanup_handler")
    
    if not ensure_button_state_manager(ui_components):
        ui_logger.error("❌ Gagal inisialisasi button state manager")
        return
    
    button_state_manager = get_button_state_manager(ui_components)
    
    try:
        can_start, reason = button_state_manager.can_start_operation("cleanup")
        if not can_start:
            ui_logger.warning(f"⚠️ {reason}")
            return
    except Exception as e:
        ui_logger.error(f"❌ Error cek operation state: {str(e)}")
        return
    
    try:
        ui_logger.info("🧹 Mempersiapkan pembersihan hasil augmentasi...")
        
        cleanup_paths = _get_cleanup_paths_safe(ui_components)
        
        if not cleanup_paths:
            ui_logger.warning("📁 Tidak ada direktori augmentasi yang ditemukan")
            return
        
        analysis = _analyze_cleanup_files_safe(cleanup_paths, ui_logger)
        
        if analysis['total_files'] == 0:
            ui_logger.info("✨ Direktori sudah bersih, tidak ada file augmentasi")
            return
        
        _show_cleanup_confirmation_safe(ui_components, analysis, ui_logger)
        
    except Exception as e:
        ui_logger.error(f"❌ Error persiapan cleanup: {str(e)}")

def _get_cleanup_paths_safe(ui_components: Dict[str, Any]) -> list:
    """Get cleanup paths dengan error handling."""
    try:
        import os
        paths = []
        
        # Path dari UI components dengan null safety
        if ui_components.get('output_dir') and hasattr(ui_components['output_dir'], 'value'):
            paths.append(ui_components['output_dir'].value)
        
        # Default paths
        default_paths = ['data/augmented', '/content/data/augmented']
        paths.extend(default_paths)
        
        return [path for path in paths if path and os.path.exists(path)]
    except Exception:
        return []

def _analyze_cleanup_files_safe(paths: list, ui_logger) -> dict:
    """Analisis files dengan safe error handling."""
    try:
        import os
        from pathlib import Path
        
        analysis = {'total_files': 0, 'total_size_mb': 0, 'paths_detail': {}, 'file_types': {}}
        aug_patterns = ['aug_', '_augmented', '_modified', '_processed']
        
        for path in paths:
            path_detail = {'files': 0, 'size_mb': 0}
            
            try:
                for root, dirs, files in os.walk(path):
                    for file in files:
                        if any(pattern in file.lower() for pattern in aug_patterns):
                            try:
                                file_path = os.path.join(root, file)
                                file_size = os.path.getsize(file_path)
                                path_detail['files'] += 1
                                path_detail['size_mb'] += file_size / (1024 * 1024)
                                analysis['total_files'] += 1
                                analysis['total_size_mb'] += file_size / (1024 * 1024)
                                
                                ext = Path(file).suffix.lower()
                                analysis['file_types'][ext] = analysis['file_types'].get(ext, 0) + 1
                            except OSError:
                                continue
            except Exception:
                continue
            
            if path_detail['files'] > 0:
                analysis['paths_detail'][path] = path_detail
        
        return analysis
    except Exception:
        return {'total_files': 0, 'total_size_mb': 0, 'paths_detail': {}, 'file_types': {}}

def _show_cleanup_confirmation_safe(ui_components: Dict[str, Any], analysis: Dict[str, Any], ui_logger) -> None:
    """Show confirmation dengan safe error handling."""
    try:
        from smartcash.ui.components.confirmation_dialog import create_confirmation_dialog
        from IPython.display import display
        
        # Ensure confirmation area
        if 'confirmation_area' not in ui_components:
            from ipywidgets import Output
            ui_components['confirmation_area'] = Output()
        
        message = _build_confirmation_message_safe(analysis)
        
        def on_confirm(b):
            ui_components['confirmation_area'].clear_output()
            ui_logger.info("✅ Konfirmasi cleanup diterima")
            _execute_cleanup_safe(ui_components, analysis, ui_logger)
        
        def on_cancel(b):
            ui_components['confirmation_area'].clear_output()
            ui_logger.info("❌ Cleanup dibatalkan oleh pengguna")
        
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
            
    except Exception as e:
        ui_logger.error(f"❌ Error show confirmation: {str(e)}")

def _build_confirmation_message_safe(analysis: Dict[str, Any]) -> str:
    """Build confirmation message dengan safe access."""
    try:
        total_files = analysis.get('total_files', 0)
        total_size = analysis.get('total_size_mb', 0)
        paths_count = len(analysis.get('paths_detail', {}))
        
        message = f"📊 **Detail Pembersihan:**\n"
        message += f"• **{total_files:,} file** augmentasi akan dihapus\n"
        message += f"• **{total_size:.1f} MB** ruang disk akan dibebaskan\n"
        message += f"• **{paths_count} direktori** akan dibersihkan\n\n"
        message += f"\n⚠️ **Tindakan ini tidak dapat dibatalkan!**\n"
        message += f"Lanjutkan pembersihan?"
        
        return message
    except Exception:
        return "Konfirmasi pembersihan file augmentasi. Lanjutkan?"

def _execute_cleanup_safe(ui_components: Dict[str, Any], analysis: Dict[str, Any], ui_logger) -> None:
    """Execute cleanup dengan safe error handling."""
    try:
        button_state_manager = get_button_state_manager(ui_components)
        
        with button_state_manager.operation_context("cleanup"):
            _run_cleanup_process(ui_components, analysis, ui_logger)
            
    except Exception as e:
        ui_logger.error(f"❌ Error saat cleanup: {str(e)}")

def _run_cleanup_process(ui_components: Dict[str, Any], analysis: Dict[str, Any], ui_logger) -> None:
    """Run cleanup process dengan progress tracking."""
    try:
        total_files = analysis.get('total_files', 0)
        paths_to_clean = list(analysis.get('paths_detail', {}).keys())
        
        ui_logger.info(f"🚀 Memulai pembersihan {total_files:,} file dari {len(paths_to_clean)} direktori")
        
        # Safe progress start
        _safe_start_progress(ui_components, f"🗑️ Membersihkan {total_files:,} file...")
        
        deleted_count = 0
        error_count = 0
        aug_patterns = ['aug_', '_augmented', '_modified', '_processed']
        
        for i, path in enumerate(paths_to_clean):
            try:
                progress_percent = int((i / len(paths_to_clean)) * 80) + 10
                _safe_update_progress(ui_components, progress_percent, f"Membersihkan {path.split('/')[-1]}...")
                
                result = _cleanup_single_directory_safe(path, aug_patterns)
                deleted_count += result['deleted']
                error_count += result['errors']
                
            except Exception as e:
                ui_logger.warning(f"⚠️ Error cleanup {path}: {str(e)}")
                error_count += 1
        
        _safe_update_progress(ui_components, 95, "🧹 Membersihkan direktori kosong...")
        _cleanup_empty_directories_safe(paths_to_clean, ui_logger)
        
        _report_cleanup_results_safe(deleted_count, error_count, analysis, ui_logger)
        _safe_complete_progress(ui_components, f"🎉 Cleanup selesai: {deleted_count:,} file dihapus!")
        
    except Exception as e:
        ui_logger.error(f"❌ Error dalam run cleanup: {str(e)}")
        _safe_error_progress(ui_components, f"❌ Error cleanup: {str(e)}")

def _cleanup_single_directory_safe(path: str, patterns: list) -> dict:
    """Cleanup single directory dengan error handling."""
    import os
    result = {'deleted': 0, 'errors': 0}
    
    try:
        for root, dirs, files in os.walk(path, topdown=False):
            for file in files:
                if any(pattern in file.lower() for pattern in patterns):
                    try:
                        file_path = os.path.join(root, file)
                        os.remove(file_path)
                        result['deleted'] += 1
                    except OSError:
                        result['errors'] += 1
    except Exception:
        result['errors'] += 1
    
    return result

def _cleanup_empty_directories_safe(paths: list, ui_logger) -> None:
    """Cleanup empty directories dengan safe error handling."""
    import os
    try:
        for path in paths:
            try:
                for root, dirs, files in os.walk(path, topdown=False):
                    for dir_name in dirs:
                        try:
                            dir_path = os.path.join(root, dir_name)
                            if not os.listdir(dir_path):
                                os.rmdir(dir_path)
                        except OSError:
                            pass
            except Exception:
                pass
    except Exception:
        pass

def _report_cleanup_results_safe(deleted: int, errors: int, analysis: dict, ui_logger) -> None:
    """Report cleanup results dengan safe access."""
    try:
        total_expected = analysis.get('total_files', 0)
        freed_space = analysis.get('total_size_mb', 0)
        
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
    except Exception:
        ui_logger.info(f"📊 Cleanup selesai: {deleted} file dihapus")

# Safe progress functions
def _safe_start_progress(ui_components: Dict[str, Any], message: str):
    try:
        if 'show_for_operation' in ui_components and callable(ui_components['show_for_operation']):
            ui_components['show_for_operation']('cleanup')
        elif 'tracker' in ui_components and hasattr(ui_components['tracker'], 'show'):
            ui_components['tracker'].show('cleanup')
    except Exception:
        pass

def _safe_update_progress(ui_components: Dict[str, Any], value: int, message: str):
    try:
        if 'update_progress' in ui_components and callable(ui_components['update_progress']):
            ui_components['update_progress']('overall', value, message)
        elif 'tracker' in ui_components and hasattr(ui_components['tracker'], 'update'):
            ui_components['tracker'].update('overall', value, message)
    except Exception:
        pass

def _safe_complete_progress(ui_components: Dict[str, Any], message: str):
    try:
        if 'complete_operation' in ui_components and callable(ui_components['complete_operation']):
            ui_components['complete_operation'](message)
        elif 'tracker' in ui_components and hasattr(ui_components['tracker'], 'complete'):
            ui_components['tracker'].complete(message)
    except Exception:
        pass

def _safe_error_progress(ui_components: Dict[str, Any], message: str):
    try:
        if 'error_operation' in ui_components and callable(ui_components['error_operation']):
            ui_components['error_operation'](message)
        elif 'tracker' in ui_components and hasattr(ui_components['tracker'], 'error'):
            ui_components['tracker'].error(message)
    except Exception:
        pass


"""
File: smartcash/ui/dataset/augmentation/handlers/save_handler.py
Deskripsi: Handler untuk menyimpan konfigurasi augmentasi dengan null safety
"""

def handle_save_button_click(ui_components: Dict[str, Any], button: Any = None) -> None:
    """Handler save dengan robust null safety."""
    ui_logger = create_ui_logger_bridge(ui_components, "save_handler")
    
    if not ensure_button_state_manager(ui_components):
        ui_logger.error("❌ Gagal inisialisasi button state manager")
        return
    
    button_state_manager = get_button_state_manager(ui_components)
    
    try:
        can_start, reason = button_state_manager.can_start_operation("save_config")
        if not can_start:
            ui_logger.warning(f"⚠️ {reason}")
            return
    except Exception as e:
        ui_logger.error(f"❌ Error cek operation state: {str(e)}")
        return
    
    try:
        with button_state_manager.operation_context("save_config"):
            ui_logger.info("💾 Menyimpan konfigurasi augmentasi...")
            
            result = _save_config_safe(ui_components, ui_logger)
            
            if result:
                ui_logger.success("✅ Konfigurasi berhasil disimpan dan disinkronkan ke Google Drive")
                _update_status_panel_safe(ui_components, "✅ Konfigurasi tersimpan di Google Drive", "success")
            else:
                ui_logger.error("❌ Gagal menyimpan konfigurasi")
                _update_status_panel_safe(ui_components, "❌ Gagal menyimpan konfigurasi", "error")
                
    except Exception as e:
        ui_logger.error(f"❌ Error saat menyimpan: {str(e)}")
        _update_status_panel_safe(ui_components, f"❌ Error: {str(e)}", "error")

def _save_config_safe(ui_components: Dict[str, Any], ui_logger) -> bool:
    """Save config dengan error handling."""
    try:
        from smartcash.ui.dataset.augmentation.handlers.config_handler import save_augmentation_config
        return save_augmentation_config(ui_components)
    except Exception as e:
        ui_logger.error(f"❌ Error save config: {str(e)}")
        return False


"""
File: smartcash/ui/dataset/augmentation/handlers/reset_handler.py  
Deskripsi: Handler untuk reset konfigurasi augmentasi dengan null safety
"""

def handle_reset_button_click(ui_components: Dict[str, Any], button: Any = None) -> None:
    """Handler reset dengan robust null safety."""
    ui_logger = create_ui_logger_bridge(ui_components, "reset_handler")
    
    if not ensure_button_state_manager(ui_components):
        ui_logger.error("❌ Gagal inisialisasi button state manager")
        return
    
    button_state_manager = get_button_state_manager(ui_components)
    
    try:
        can_start, reason = button_state_manager.can_start_operation("reset_config")
        if not can_start:
            ui_logger.warning(f"⚠️ {reason}")
            return
    except Exception as e:
        ui_logger.error(f"❌ Error cek operation state: {str(e)}")
        return
    
    try:
        with button_state_manager.operation_context("reset_config"):
            ui_logger.info("🔄 Mereset konfigurasi ke default...")
            
            result = _reset_config_safe(ui_components, ui_logger)
            
            if result:
                _reset_ui_states_safe(ui_components, ui_logger)
                _reset_progress_components_safe(ui_components)
                
                ui_logger.success("✅ Konfigurasi berhasil direset dan disimpan ke Google Drive")
                _update_status_panel_safe(ui_components, "✅ Konfigurasi direset dan tersinkronisasi", "success")
            else:
                ui_logger.error("❌ Gagal mereset konfigurasi")
                _update_status_panel_safe(ui_components, "❌ Gagal mereset konfigurasi", "error")
            
    except Exception as e:
        ui_logger.error(f"❌ Error saat mereset: {str(e)}")
        _update_status_panel_safe(ui_components, f"❌ Gagal mereset: {str(e)}", "error")

def _reset_config_safe(ui_components: Dict[str, Any], ui_logger) -> bool:
    """Reset config dengan error handling."""
    try:
        from smartcash.ui.dataset.augmentation.handlers.config_handler import reset_augmentation_config
        return reset_augmentation_config(ui_components)
    except Exception as e:
        ui_logger.error(f"❌ Error reset config: {str(e)}")
        return False

def _reset_ui_states_safe(ui_components: Dict[str, Any], ui_logger) -> None:
    """Reset UI states dengan safe error handling."""
    try:
        if 'confirmation_result' in ui_components:
            ui_components['confirmation_result'] = False
        
        if 'confirmation_area' in ui_components:
            ui_components['confirmation_area'].clear_output()
        
        ui_components['augmentation_running'] = False
        ui_components['stop_requested'] = False
        
        ui_logger.debug("🔄 UI states berhasil direset")
    except Exception:
        pass

def _reset_progress_components_safe(ui_components: Dict[str, Any]) -> None:
    """Reset progress components dengan safe error handling."""
    try:
        if 'reset_all' in ui_components and callable(ui_components['reset_all']):
            ui_components['reset_all']()
        elif 'tracker' in ui_components and hasattr(ui_components['tracker'], 'reset'):
            ui_components['tracker'].reset()
    except Exception:
        pass

def _update_status_panel_safe(ui_components: Dict[str, Any], message: str, status: str) -> None:
    """Update status panel dengan safe error handling."""
    try:
        if 'status_panel' in ui_components:
            from smartcash.ui.components.status_panel import update_status_panel
            update_status_panel(ui_components['status_panel'], message, status)
    except Exception:
        pass