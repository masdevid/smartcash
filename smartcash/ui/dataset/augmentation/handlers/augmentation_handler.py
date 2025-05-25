"""
File: smartcash/ui/dataset/augmentation/handlers/augmentation_handler.py
Deskripsi: Refactored augmentation handler menggunakan safe utils pattern
"""

import time
from typing import Dict, Any
from smartcash.ui.utils.safe_handler_utils import (
    safe_handler_wrapper, safe_progress_update, safe_progress_start, 
    safe_progress_complete, safe_progress_error, safe_update_status_panel,
    safe_import_and_call, create_safe_callback
)

@safe_handler_wrapper("augmentation")
def handle_augmentation_button_click(ui_components: Dict[str, Any], button: Any, ui_logger) -> None:
    """Handler augmentation dengan safe utils pattern."""
    
    # Reset state dan validasi parameter
    ui_components['stop_requested'] = False
    ui_components['augmentation_running'] = False
    
    # Ekstrak dan validasi parameter
    validation_result = safe_import_and_call(
        'smartcash.ui.dataset.augmentation.handlers.parameter_handler',
        'extract_and_validate_parameters',
        ui_components, ui_logger,
        fallback_result=(False, "Error validasi parameter", {})
    )
    
    is_valid, error_message, validated_params = validation_result
    if not is_valid:
        ui_logger.error(f"❌ Validasi parameter gagal: {error_message}")
        return
    
    # Show confirmation
    safe_import_and_call(
        'smartcash.ui.dataset.augmentation.handlers.confirmation_handler',
        'show_augmentation_confirmation',
        ui_components, validated_params, ui_logger,
        lambda: execute_augmentation_process(ui_components, validated_params, ui_logger)
    )

def execute_augmentation_process(ui_components: Dict[str, Any], params: Dict[str, Any], ui_logger) -> None:
    """Execute augmentation dengan comprehensive error handling."""
    start_time = time.time()
    
    try:
        ui_logger.info("🚀 Memulai proses augmentasi...")
        safe_progress_start(ui_components, "augmentation", "Memulai augmentasi...")
        
        # Setup symlink (5-15%)
        safe_progress_update(ui_components, 5, "🔗 Setup symlink untuk Google Drive...")
        
        symlink_result = safe_import_and_call(
            'smartcash.ui.dataset.augmentation.handlers.symlink_handler',
            'setup_augmentation_symlinks',
            ui_components, params, ui_logger,
            fallback_result=(True, "Local fallback", {'uses_symlink': False, 'storage_type': 'Local'})
        )
        
        symlink_success, symlink_message, symlink_info = symlink_result
        if not symlink_success:
            result = {'status': 'error', 'message': f"Symlink setup gagal: {symlink_message}"}
            _handle_result_safe(ui_components, result, time.time() - start_time, ui_logger)
            return
        
        params.update(symlink_info)
        safe_progress_update(ui_components, 15, "✅ Symlink setup berhasil")
        
        # Run augmentation (15-95%)
        safe_progress_update(ui_components, 20, "🚀 Memulai proses augmentasi...")
        
        service_callback = create_safe_service_callback(ui_components)
        result = run_augmentation_safe(ui_components, params, service_callback, ui_logger)
        
        # Finalize (95-100%)
        safe_progress_update(ui_components, 98, "🔄 Memproses hasil augmentasi...")
        
        duration = time.time() - start_time
        _handle_result_safe(ui_components, result, duration, ui_logger)
        safe_progress_complete(ui_components, "🎉 Augmentasi selesai!")
        
    except Exception as e:
        duration = time.time() - start_time
        error_result = {'status': 'error', 'message': f"Error augmentasi: {str(e)}"}
        
        safe_progress_error(ui_components, f"❌ Error: {str(e)}")
        _handle_result_safe(ui_components, error_result, duration, ui_logger)
        ui_logger.error(f"🔥 Critical error augmentasi: {str(e)}")

def run_augmentation_safe(ui_components: Dict[str, Any], params: Dict[str, Any], 
                         progress_callback, ui_logger) -> Dict[str, Any]:
    """Run augmentation dengan safe error handling."""
    try:
        from smartcash.dataset.services.augmentor.augmentation_service import AugmentationService
        
        service_config = {
            'data_dir': params.get('data_path', 'data'),
            'augmented_dir': f"{params.get('data_path', 'data')}/augmented",
            'num_workers': 1
        }
        
        ui_components['progress_callback'] = progress_callback
        
        service = AugmentationService(
            config=service_config,
            data_dir=params.get('data_path', 'data'),
            num_workers=1,
            ui_components=ui_components
        )
        
        return service.augment_dataset(
            data_dir=params.get('data_path', 'data'),
            split=params.get('split', 'train'),
            types=params.get('types', ['combined']),
            num_variations=params.get('num_variations', 2),
            target_count=params.get('target_count', 500),
            output_prefix=params.get('output_prefix', 'aug_'),
            balance_classes=params.get('balance_classes', False),
            validate_results=params.get('validate_results', True),
            progress_callback=progress_callback,
            create_symlinks=True
        )
        
    except ImportError:
        return {'status': 'error', 'message': 'AugmentationService tidak tersedia', 'generated_images': 0, 'processed': 0}
    except Exception as e:
        return {'status': 'error', 'message': f'Error saat augmentasi: {str(e)}', 'generated_images': 0, 'processed': 0}

def create_safe_service_callback(ui_components: Dict[str, Any]):
    """Create safe service callback dengan progress mapping."""
    def service_callback(current: int, total: int, message: str = "", **kwargs) -> bool:
        try:
            if ui_components.get('stop_requested', False):
                return False
            
            if total > 0:
                service_progress = min(100, (current / total) * 100)
                ui_progress = 15 + int(service_progress * 0.8)  # Map ke 15-95%
                
                display_message = f"{message} ({current}/{total})" if message else f"Memproses: {current}/{total}"
                safe_progress_update(ui_components, ui_progress, display_message)
            
            return True
        except Exception:
            return True  # Silent fail, continue processing
    
    return service_callback

def _handle_result_safe(ui_components: Dict[str, Any], result: Dict[str, Any], 
                       duration: float, ui_logger) -> None:
    """Handle result dengan safe error handling."""
    result_handled = safe_import_and_call(
        'smartcash.ui.dataset.augmentation.handlers.result_handler',
        'handle_augmentation_result',
        ui_components, result, duration, ui_logger,
        fallback_result=True
    )
    
    if not result_handled:
        # Fallback result handling
        status = result.get('status', 'error')
        message = result.get('message', 'Proses selesai')
        
        if status == 'success':
            ui_logger.success(f"✅ {message}")
            safe_update_status_panel(ui_components, message, 'success')
        elif status == 'error':
            ui_logger.error(f"❌ {message}")
            safe_update_status_panel(ui_components, message, 'error')
        else:
            ui_logger.info(f"ℹ️ {message}")
            safe_update_status_panel(ui_components, message, 'info')


"""
File: smartcash/ui/dataset/augmentation/handlers/cleanup_handler.py
Deskripsi: Refactored cleanup handler menggunakan safe utils pattern
"""

@safe_handler_wrapper("cleanup")  
def handle_cleanup_button_click(ui_components: Dict[str, Any], button: Any, ui_logger) -> None:
    """Handler cleanup dengan safe utils pattern."""
    
    ui_logger.info("🧹 Mempersiapkan pembersihan hasil augmentasi...")
    
    # Get cleanup paths
    cleanup_paths = _get_cleanup_paths_safe(ui_components)
    if not cleanup_paths:
        ui_logger.warning("📁 Tidak ada direktori augmentasi yang ditemukan")
        return
    
    # Analyze files
    analysis = _analyze_cleanup_files_safe(cleanup_paths, ui_logger)
    if analysis['total_files'] == 0:
        ui_logger.info("✨ Direktori sudah bersih, tidak ada file augmentasi")
        return
    
    # Show confirmation
    _show_cleanup_confirmation_safe(ui_components, analysis, ui_logger)

def _get_cleanup_paths_safe(ui_components: Dict[str, Any]) -> list:
    """Get cleanup paths dengan safe error handling."""
    try:
        import os
        from smartcash.ui.utils.safe_handler_utils import safe_get_widget_value
        
        paths = []
        
        # Path dari UI components
        output_dir = safe_get_widget_value(ui_components, 'output_dir')
        if output_dir:
            paths.append(output_dir)
        
        # Default paths
        default_paths = ['data/augmented', '/content/data/augmented']
        paths.extend(default_paths)
        
        return [path for path in paths if path and os.path.exists(path)]
    except Exception:
        return []

def _analyze_cleanup_files_safe(paths: list, ui_logger) -> dict:
    """Analyze files dengan comprehensive error handling."""
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
    """Show cleanup confirmation dengan safe error handling."""
    try:
        from smartcash.ui.components.confirmation_dialog import create_confirmation_dialog
        from IPython.display import display
        from smartcash.ui.utils.safe_handler_utils import safe_clear_confirmation_area
        
        # Ensure confirmation area
        if 'confirmation_area' not in ui_components:
            from ipywidgets import Output
            ui_components['confirmation_area'] = Output()
        
        message = _build_cleanup_message_safe(analysis)
        
        def on_confirm(b):
            safe_clear_confirmation_area(ui_components)
            ui_logger.info("✅ Konfirmasi cleanup diterima")
            _execute_cleanup_process(ui_components, analysis, ui_logger)
        
        def on_cancel(b):
            safe_clear_confirmation_area(ui_components)
            ui_logger.info("❌ Cleanup dibatalkan oleh pengguna")
        
        dialog = create_confirmation_dialog(
            title="🧹 Konfirmasi Pembersihan Augmentasi",
            message=message,
            on_confirm=on_confirm,
            on_cancel=on_cancel,
            danger_mode=True
        )
        
        safe_clear_confirmation_area(ui_components)
        with ui_components['confirmation_area']:
            display(dialog)
            
    except Exception as e:
        ui_logger.error(f"❌ Error show confirmation: {str(e)}")

def _build_cleanup_message_safe(analysis: Dict[str, Any]) -> str:
    """Build cleanup message dengan safe access."""
    try:
        total_files = analysis.get('total_files', 0)
        total_size = analysis.get('total_size_mb', 0)
        paths_count = len(analysis.get('paths_detail', {}))
        
        return f"""📊 **Detail Pembersihan:**
• **{total_files:,} file** augmentasi akan dihapus
• **{total_size:.1f} MB** ruang disk akan dibebaskan  
• **{paths_count} direktori** akan dibersihkan

⚠️ **Tindakan ini tidak dapat dibatalkan!**
Lanjutkan pembersihan?"""
    except Exception:
        return "Konfirmasi pembersihan file augmentasi. Lanjutkan?"

def _execute_cleanup_process(ui_components: Dict[str, Any], analysis: Dict[str, Any], ui_logger) -> None:
    """Execute cleanup process dengan comprehensive progress tracking."""
    try:
        total_files = analysis.get('total_files', 0)
        paths_to_clean = list(analysis.get('paths_detail', {}).keys())
        
        ui_logger.info(f"🚀 Memulai pembersihan {total_files:,} file dari {len(paths_to_clean)} direktori")
        
        safe_progress_start(ui_components, "cleanup", f"🗑️ Membersihkan {total_files:,} file...")
        
        deleted_count = 0
        error_count = 0
        aug_patterns = ['aug_', '_augmented', '_modified', '_processed']
        
        # Process cleanup dengan progress
        for i, path in enumerate(paths_to_clean):
            try:
                progress_percent = int((i / len(paths_to_clean)) * 80) + 10  # 10-90%
                path_name = path.split('/')[-1] if '/' in path else path
                safe_progress_update(ui_components, progress_percent, f"Membersihkan {path_name}...")
                
                result = _cleanup_single_directory_safe(path, aug_patterns)
                deleted_count += result['deleted']
                error_count += result['errors']
                
            except Exception as e:
                ui_logger.warning(f"⚠️ Error cleanup {path}: {str(e)}")
                error_count += 1
        
        # Final cleanup
        safe_progress_update(ui_components, 95, "🧹 Membersihkan direktori kosong...")
        _cleanup_empty_directories_safe(paths_to_clean)
        
        # Report results
        _report_cleanup_results_safe(deleted_count, error_count, analysis, ui_logger)
        safe_progress_complete(ui_components, f"🎉 Cleanup selesai: {deleted_count:,} file dihapus!")
        
    except Exception as e:
        ui_logger.error(f"❌ Error dalam cleanup process: {str(e)}")
        safe_progress_error(ui_components, f"❌ Error cleanup: {str(e)}")

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

def _cleanup_empty_directories_safe(paths: list) -> None:
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
    """Report cleanup results dengan safe formatting."""
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


"""
File: smartcash/ui/dataset/augmentation/handlers/save_handler.py
Deskripsi: Refactored save handler dengan safe utils pattern
"""

@safe_handler_wrapper("save_config")
def handle_save_button_click(ui_components: Dict[str, Any], button: Any, ui_logger) -> None:
    """Handler save config dengan safe utils pattern."""
    
    ui_logger.info("💾 Menyimpan konfigurasi augmentasi...")
    
    result = safe_import_and_call(
        'smartcash.ui.dataset.augmentation.handlers.config_handler',
        'save_augmentation_config',
        ui_components,
        fallback_result=False
    )
    
    if result:
        ui_logger.success("✅ Konfigurasi berhasil disimpan dan disinkronkan ke Google Drive")
        safe_update_status_panel(ui_components, "✅ Konfigurasi tersimpan di Google Drive", "success")
    else:
        ui_logger.error("❌ Gagal menyimpan konfigurasi")
        safe_update_status_panel(ui_components, "❌ Gagal menyimpan konfigurasi", "error")


"""  
File: smartcash/ui/dataset/augmentation/handlers/reset_handler.py
Deskripsi: Refactored reset handler dengan safe utils pattern
"""

@safe_handler_wrapper("reset_config")
def handle_reset_button_click(ui_components: Dict[str, Any], button: Any, ui_logger) -> None:
    """Handler reset config dengan safe utils pattern."""
    
    ui_logger.info("🔄 Mereset konfigurasi ke default...")
    
    result = safe_import_and_call(
        'smartcash.ui.dataset.augmentation.handlers.config_handler',
        'reset_augmentation_config',
        ui_components,
        fallback_result=False
    )
    
    if result:
        # Reset UI states dan progress components dengan safe methods
        from smartcash.ui.utils.safe_handler_utils import safe_reset_ui_state
        safe_reset_ui_state(ui_components)
        
        # Reset progress components
        try:
            if 'reset_all' in ui_components and callable(ui_components['reset_all']):
                ui_components['reset_all']()
            elif 'tracker' in ui_components and hasattr(ui_components['tracker'], 'reset'):
                ui_components['tracker'].reset()
        except Exception:
            pass
        
        ui_logger.success("✅ Konfigurasi berhasil direset dan disimpan ke Google Drive")
        safe_update_status_panel(ui_components, "✅ Konfigurasi direset dan tersinkronisasi", "success")
    else:
        ui_logger.error("❌ Gagal mereset konfigurasi")
        safe_update_status_panel(ui_components, "❌ Gagal mereset konfigurasi", "error")