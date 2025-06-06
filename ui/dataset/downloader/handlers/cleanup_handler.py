"""
File: smartcash/ui/dataset/downloader/handlers/cleanup_handler.py
Deskripsi: Fixed cleanup handler dengan safe progress callbacks dan proper error handling
"""

from typing import Dict, Any, Callable
from smartcash.ui.utils.fallback_utils import show_status_safe
from smartcash.ui.components.confirmation_dialog import create_destructive_confirmation
from smartcash.dataset.services.organizer.dataset_organizer import DatasetOrganizer
from smartcash.dataset.utils.path_validator import get_path_validator
from smartcash.common.utils.one_liner_fixes import safe_operation_or_none, fix_directory_operation

def setup_cleanup_handler(ui_components: Dict[str, Any], config: Dict[str, Any], logger) -> Callable:
    """Setup optimized cleanup handler dengan safe progress tracking"""
    
    def handle_cleanup(button):
        """Handle cleanup dengan safe validation dan confirmation flow"""
        try:
            # Check existing data dengan safe info gathering
            cleanup_info = _get_streamlined_cleanup_info()
            
            # Early return jika tidak ada file
            if cleanup_info.get('total_files', 0) == 0:
                show_status_safe("ℹ️ Tidak ada file untuk dibersihkan", "info", ui_components)
                return
            
            # Show confirmation dialog
            _show_streamlined_cleanup_confirmation(ui_components, cleanup_info, logger)
            
        except Exception as e:
            logger.error(f"❌ Error cleanup handler: {str(e)}")
            show_status_safe(f"❌ Error: {str(e)}", "error", ui_components)
    
    return handle_cleanup

def _get_streamlined_cleanup_info() -> Dict[str, Any]:
    """Get streamlined cleanup info dengan safe file counting"""
    def info_operation():
        path_validator = get_path_validator()
        dataset_paths = path_validator.get_dataset_paths()
        
        # Streamlined cleanup targets
        cleanup_targets = {
            'train': {'path': dataset_paths['train'], 'description': 'Training dataset', 'icon': '🏋️'},
            'valid': {'path': dataset_paths['valid'], 'description': 'Validation dataset', 'icon': '✅'},
            'test': {'path': dataset_paths['test'], 'description': 'Test dataset', 'icon': '🧪'},
            'downloads': {'path': dataset_paths.get('downloads', ''), 'description': 'Downloaded files', 'icon': '📥'},
            'backup': {'path': dataset_paths.get('backup', ''), 'description': 'Backup files', 'icon': '💾'}
        }
        
        target_info = {}
        total_files = 0
        total_size_mb = 0
        
        # Safe file analysis
        from pathlib import Path
        
        for target, target_config in cleanup_targets.items():
            if not target_config['path']:
                target_info[target] = {
                    'path': '', 'files': 0, 'size_mb': 0, 'exists': False, **target_config
                }
                continue
                
            path = Path(target_config['path'])
            if path.exists():
                # Safe file counting
                try:
                    files = [f for f in path.rglob('*') if f.is_file()]
                    file_count = len(files)
                    size_bytes = sum(f.stat().st_size for f in files if f.exists())
                    size_mb = size_bytes / (1024 * 1024)
                    
                    target_info[target] = {
                        'path': str(path), 'files': file_count, 'size_mb': size_mb, 
                        'exists': True, **target_config
                    }
                    total_files += file_count
                    total_size_mb += size_mb
                except Exception:
                    target_info[target] = {
                        'path': str(path), 'files': 0, 'size_mb': 0, 
                        'exists': False, **target_config
                    }
            else:
                target_info[target] = {
                    'path': str(path), 'files': 0, 'size_mb': 0, 
                    'exists': False, **target_config
                }
        
        return {
            'total_files': total_files,
            'total_size_mb': total_size_mb,
            'targets': target_info,
            'has_data': total_files > 0,
            'targets_with_data': [name for name, info in target_info.items() if info['files'] > 0]
        }
    
    return safe_operation_or_none(info_operation) or {
        'total_files': 0, 'total_size_mb': 0, 'targets': {}, 
        'has_data': False, 'targets_with_data': []
    }

def _show_streamlined_cleanup_confirmation(ui_components: Dict[str, Any], cleanup_info: Dict[str, Any], logger) -> None:
    """Show streamlined destructive confirmation"""
    
    total_files = cleanup_info.get('total_files', 0)
    total_size_mb = cleanup_info.get('total_size_mb', 0)
    targets_with_data = cleanup_info.get('targets_with_data', [])
    targets = cleanup_info.get('targets', {})
    
    # Format file breakdown dengan safe access
    file_breakdown_lines = []
    for target in targets_with_data:
        target_info = targets.get(target, {})
        icon = target_info.get('icon', '📁')
        description = target_info.get('description', target)
        files = target_info.get('files', 0)
        size_mb = target_info.get('size_mb', 0)
        
        file_breakdown_lines.append(
            f"{icon} **{description}**: {files:,} files ({size_mb:.1f} MB)"
        )
    
    file_breakdown = '\n'.join(file_breakdown_lines)
    
    # Safe confirmation handlers
    def on_confirm_handler(button):
        _clear_confirmation_area(ui_components)
        _execute_streamlined_cleanup_sync(ui_components, cleanup_info, logger)
    
    def on_cancel_handler(button):
        _clear_confirmation_area(ui_components)
    
    confirmation_dialog = create_destructive_confirmation(
        title="🧹 Konfirmasi Cleanup Dataset",
        message=f"""⚠️ **OPERASI DESTRUKTIF - TIDAK DAPAT DIBATALKAN** ⚠️

📊 **File yang akan dihapus:**
{file_breakdown}

📈 **Total Summary:**
• **Files:** {total_files:,} file akan dihapus
• **Size:** {total_size_mb:.1f} MB akan dibebaskan
• **Locations:** {len(targets_with_data)} direktori akan dibersihkan

🚨 **PERINGATAN PENTING:**
❌ Operasi ini **PERMANENT** dan **TIDAK DAPAT DIBATALKAN**
❌ Semua data dataset akan **HILANG SELAMANYA**
💾 Pastikan sudah **BACKUP DATA PENTING** secara manual

⚡ **Yakin ingin melanjutkan cleanup destruktif ini?**""",
        on_confirm=on_confirm_handler,
        on_cancel=on_cancel_handler,
        item_name="Dataset Complete", 
        confirm_text="🗑️ Ya, Hapus SEMUA", 
        cancel_text="❌ Batal"
    )
    
    _show_in_confirmation_area(ui_components, confirmation_dialog)

def _execute_streamlined_cleanup_sync(ui_components: Dict[str, Any], cleanup_info: Dict[str, Any], logger) -> None:
    """Execute streamlined cleanup dengan safe progress tracking"""
    try:
        # Get progress tracker safely
        progress_tracker = ui_components.get('progress_tracker')
        if not progress_tracker:
            logger.error("❌ Progress tracker tidak ditemukan")
            show_status_safe("❌ Progress tracker tidak tersedia", "error", ui_components)
            return
        
        # Show progress dengan safe API calls
        cleanup_steps = ["scan", "cleanup", "verify"]
        step_weights = {"scan": 10, "cleanup": 80, "verify": 10}
        safe_operation_or_none(lambda: progress_tracker.show("Cleanup Dataset", cleanup_steps, step_weights))
        
        # Create organizer dengan safe initialization
        organizer = safe_operation_or_none(lambda: DatasetOrganizer(logger))
        if not organizer:
            error_msg = "❌ Gagal membuat dataset organizer"
            safe_operation_or_none(lambda: progress_tracker.error(error_msg))
            show_status_safe(error_msg, "error", ui_components)
            return
        
        # Setup organizer dengan safe progress callback
        progress_callback = _create_streamlined_cleanup_progress_callback(progress_tracker, logger, cleanup_info)
        safe_operation_or_none(lambda: organizer.set_progress_callback(progress_callback))
        
        # Execute cleanup safely
        result = safe_operation_or_none(lambda: organizer.cleanup_all_dataset_folders()) or {
            'status': 'error', 'message': 'Cleanup operation failed', 'stats': {}
        }
        
        # Handle result dengan safe status reporting
        _handle_cleanup_result(result, cleanup_info, progress_tracker, ui_components, logger)
        
    except Exception as e:
        error_msg = f"❌ Error saat cleanup: {str(e)}"
        progress_tracker = ui_components.get('progress_tracker')
        safe_operation_or_none(lambda: progress_tracker.error(error_msg) if progress_tracker else None)
        show_status_safe(error_msg, "error", ui_components)
        logger.error(error_msg)

def _handle_cleanup_result(result: Dict[str, Any], cleanup_info: Dict[str, Any], 
                          progress_tracker, ui_components: Dict[str, Any], logger) -> None:
    """Handle cleanup result dengan safe operations"""
    
    status = result.get('status', 'error')
    
    if status == 'success':
        stats = result.get('stats', {})
        files_removed = stats.get('total_files_removed', cleanup_info.get('total_files', 0))
        size_freed = cleanup_info.get('total_size_mb', 0)
        
        success_msg = f"✅ Berhasil membersihkan {files_removed} file ({size_freed:.1f} MB)"
        safe_operation_or_none(lambda: progress_tracker.complete(success_msg))
        show_status_safe(success_msg, "success", ui_components)
        logger.info(f"🧹 {success_msg}")
        
        # Additional success details dalam log
        _show_cleanup_summary(ui_components, result, cleanup_info)
        
    elif status == 'empty':
        info_msg = "ℹ️ Tidak ada file yang perlu dibersihkan"
        safe_operation_or_none(lambda: progress_tracker.complete(info_msg))
        show_status_safe(info_msg, "info", ui_components)
        logger.info(f"🧹 {info_msg}")
        
    else:
        error_msg = f"❌ Error saat cleanup: {result.get('message', 'Unknown error')}"
        safe_operation_or_none(lambda: progress_tracker.error(error_msg))
        show_status_safe(error_msg, "error", ui_components)
        logger.error(f"🧹 {error_msg}")

def _show_cleanup_summary(ui_components: Dict[str, Any], result: Dict[str, Any], cleanup_info: Dict[str, Any]) -> None:
    """Show cleanup summary dalam log output"""
    def show_operation():
        log_output = ui_components.get('log_output')
        if log_output and hasattr(log_output, 'clear_output'):
            with log_output:
                from IPython.display import display, HTML
                
                stats = result.get('stats', {})
                folders_cleaned = stats.get('folders_cleaned', [])
                duration = result.get('duration', 'N/A')
                size_freed = cleanup_info.get('total_size_mb', 0)
                
                display(HTML(f"""
                <div style="padding: 10px; background: #e8f5e8; border-radius: 4px; margin: 5px 0;">
                    <strong style="color: #2e7d32;">🎉 Cleanup Summary:</strong><br>
                    📁 Folders cleaned: {', '.join(folders_cleaned) if folders_cleaned else 'None'}<br>
                    💾 Space freed: ~{size_freed:.1f} MB<br>
                    ⏱️ Duration: {duration} seconds
                </div>
                """))
    
    safe_operation_or_none(show_operation)

def _create_streamlined_cleanup_progress_callback(progress_tracker, logger, cleanup_info: Dict[str, Any]) -> Callable:
    """Create streamlined progress callback untuk cleanup operations"""
    
    def cleanup_progress_callback(step: str, current: int, total: int, message: str):
        """Streamlined progress callback dengan safe API calls"""
        def callback_operation():
            percentage = min(100, max(0, int((current / total) * 100) if total > 0 else 0))
            
            # Map cleanup steps ke progress levels
            if step == 'scan':
                safe_operation_or_none(lambda: progress_tracker.update('overall', percentage, "🔍 Scanning files"))
                safe_operation_or_none(lambda: progress_tracker.update('current', percentage, message))
            elif step == 'cleanup':
                safe_operation_or_none(lambda: progress_tracker.update('current', percentage, message))
                overall_pct = 10 + int(percentage * 0.8)
                safe_operation_or_none(lambda: progress_tracker.update('overall', overall_pct, f"🧹 Cleanup: {percentage}%"))
            elif step == 'verify':
                overall_pct = 90 + int(percentage * 0.1)
                safe_operation_or_none(lambda: progress_tracker.update('overall', overall_pct, "✅ Verifikasi cleanup"))
                safe_operation_or_none(lambda: progress_tracker.update('current', percentage, message))
            else:
                safe_operation_or_none(lambda: progress_tracker.update('current', percentage, message))
        
        safe_operation_or_none(callback_operation)
    
    return cleanup_progress_callback

def _show_in_confirmation_area(ui_components: Dict[str, Any], dialog_widget) -> None:
    """Show dialog dalam confirmation area dengan safe display"""
    def show_operation():
        confirmation_area = ui_components.get('confirmation_area')
        if confirmation_area and hasattr(confirmation_area, 'layout'):
            confirmation_area.layout.display = 'block'
            confirmation_area.layout.visibility = 'visible'
            
            if hasattr(confirmation_area, 'clear_output'):
                confirmation_area.clear_output(wait=True)
                
            from IPython.display import display
            with confirmation_area:
                display(dialog_widget)
    
    safe_operation_or_none(show_operation)

def _clear_confirmation_area(ui_components: Dict[str, Any]) -> None:
    """Clear confirmation area dengan safe operations"""
    def clear_operation():
        confirmation_area = ui_components.get('confirmation_area')
        if confirmation_area and hasattr(confirmation_area, 'clear_output'):
            confirmation_area.clear_output(wait=True)
            
            if hasattr(confirmation_area, 'layout'):
                confirmation_area.layout.display = 'none'
                confirmation_area.layout.visibility = 'hidden'
    
    safe_operation_or_none(clear_operation)

# Safe utilities dengan improved error handling
def get_cleanup_status(ui: Dict[str, Any]) -> Dict[str, Any]:
    """Get cleanup status safely"""
    def status_operation():
        cleanup_info = _get_streamlined_cleanup_info()
        return {
            'ready_for_cleanup': cleanup_info.get('has_data', False),
            'confirmation_available': bool(ui.get('confirmation_area')),
            'progress_tracker_available': bool(ui.get('progress_tracker')),
            'total_files': cleanup_info.get('total_files', 0)
        }
    
    return safe_operation_or_none(status_operation) or {
        'ready_for_cleanup': False, 'confirmation_available': False, 
        'progress_tracker_available': False, 'total_files': 0
    }

def has_cleanup_data() -> bool:
    """Check if has cleanup data safely"""
    def check_operation():
        cleanup_info = _get_streamlined_cleanup_info()
        return cleanup_info.get('has_data', False)
    
    return bool(safe_operation_or_none(check_operation))

def get_cleanup_targets() -> list:
    """Get cleanup targets safely"""
    def targets_operation():
        cleanup_info = _get_streamlined_cleanup_info()
        return cleanup_info.get('targets_with_data', [])
    
    return safe_operation_or_none(targets_operation) or []

def estimate_cleanup_duration() -> str:
    """Estimate cleanup duration safely"""
    def estimate_operation():
        cleanup_info = _get_streamlined_cleanup_info()
        total_files = cleanup_info.get('total_files', 0)
        
        if total_files == 0:
            return "Instant"
        
        estimated_seconds = max(5, total_files // 1000)
        return f"{estimated_seconds} detik"
    
    return safe_operation_or_none(estimate_operation) or "Unknown"

def format_cleanup_summary(info: Dict[str, Any]) -> str:
    """Format cleanup summary safely"""
    def format_operation():
        total_files = info.get('total_files', 0)
        total_size_mb = info.get('total_size_mb', 0)
        targets_count = len(info.get('targets_with_data', []))
        
        return f"Files: {total_files:,} | Size: {total_size_mb:.1f} MB | Targets: {targets_count}"
    
    return safe_operation_or_none(format_operation) or "No cleanup info available"