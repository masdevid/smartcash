"""
File: smartcash/ui/dataset/downloader/handlers/cleanup_handler.py
Deskripsi: Optimized cleanup handler dengan progress tracker dual-level dan one-liner style
"""

from typing import Dict, Any, Callable
from smartcash.ui.utils.fallback_utils import show_status_safe
from smartcash.ui.components.confirmation_dialog import create_destructive_confirmation
from smartcash.dataset.services.organizer.dataset_organizer import DatasetOrganizer
from smartcash.dataset.utils.path_validator import get_path_validator

def setup_cleanup_handler(ui_components: Dict[str, Any], config: Dict[str, Any], logger) -> Callable:
    """Setup optimized cleanup handler dengan dual-level progress"""
    
    def handle_cleanup(button):
        """Handle cleanup dengan streamlined validation dan confirmation flow"""
        try:
            # Check existing data dengan one-liner info gathering
            cleanup_info = _get_streamlined_cleanup_info()
            
            # Early return jika tidak ada file
            if cleanup_info['total_files'] == 0:
                show_status_safe("ℹ️ Tidak ada file untuk dibersihkan", "info", ui_components)
                return
            
            # Show confirmation dialog
            _show_streamlined_cleanup_confirmation(ui_components, cleanup_info, logger)
            
        except Exception as e:
            logger.error(f"❌ Error cleanup handler: {str(e)}")
            show_status_safe(f"❌ Error: {str(e)}", "error", ui_components)
    
    return handle_cleanup

def _get_streamlined_cleanup_info() -> Dict[str, Any]:
    """Get streamlined cleanup info dengan optimized file counting"""
    try:
        path_validator = get_path_validator()
        dataset_paths = path_validator.get_dataset_paths()
        
        # Streamlined cleanup targets
        cleanup_targets = {
            'train': {'path': dataset_paths['train'], 'description': 'Training dataset', 'icon': '🏋️'},
            'valid': {'path': dataset_paths['valid'], 'description': 'Validation dataset', 'icon': '✅'},
            'test': {'path': dataset_paths['test'], 'description': 'Test dataset', 'icon': '🧪'},
            'downloads': {'path': dataset_paths['downloads'], 'description': 'Downloaded files', 'icon': '📥'},
            'backup': {'path': dataset_paths.get('backup', ''), 'description': 'Backup files', 'icon': '💾'}
        }
        
        target_info, total_files, total_size_mb = {}, 0, 0
        
        # Optimized file analysis dengan one-liner processing
        from pathlib import Path
        
        for target, target_config in cleanup_targets.items():
            if not target_config['path']:
                target_info[target] = {'path': '', 'files': 0, 'size_mb': 0, 'exists': False, **target_config}
                continue
                
            path = Path(target_config['path'])
            if path.exists():
                # One-liner file counting dan size calculation
                files = [f for f in path.rglob('*') if f.is_file()]
                file_count, size_bytes = len(files), sum(f.stat().st_size for f in files if f.exists())
                size_mb = size_bytes / (1024 * 1024)
                
                target_info[target] = {'path': str(path), 'files': file_count, 'size_mb': size_mb, 'exists': True, **target_config}
                total_files, total_size_mb = total_files + file_count, total_size_mb + size_mb
            else:
                target_info[target] = {'path': str(path), 'files': 0, 'size_mb': 0, 'exists': False, **target_config}
        
        return {
            'total_files': total_files, 'total_size_mb': total_size_mb, 'targets': target_info,
            'has_data': total_files > 0, 'targets_with_data': [name for name, info in target_info.items() if info['files'] > 0]
        }
        
    except Exception as e:
        return {'total_files': 0, 'total_size_mb': 0, 'targets': {}, 'has_data': False, 'error': str(e)}

def _show_streamlined_cleanup_confirmation(ui_components: Dict[str, Any], cleanup_info: Dict[str, Any], logger) -> None:
    """Show streamlined destructive confirmation"""
    
    total_files, total_size_mb = cleanup_info['total_files'], cleanup_info['total_size_mb']
    targets_with_data = cleanup_info['targets_with_data']
    
    # Format file breakdown dengan one-liner
    file_breakdown_lines = [
        f"{cleanup_info['targets'][target]['icon']} **{cleanup_info['targets'][target]['description']}**: {cleanup_info['targets'][target]['files']:,} files ({cleanup_info['targets'][target]['size_mb']:.1f} MB)"
        for target in targets_with_data
    ]
    file_breakdown = '\n'.join(file_breakdown_lines)
    
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
        on_confirm=lambda b: (_clear_confirmation_area(ui_components), _execute_streamlined_cleanup_sync(ui_components, cleanup_info, logger)),
        on_cancel=lambda b: _clear_confirmation_area(ui_components),
        item_name="Dataset Complete", confirm_text="🗑️ Ya, Hapus SEMUA", cancel_text="❌ Batal"
    )
    
    _show_in_confirmation_area(ui_components, confirmation_dialog)

def _execute_streamlined_cleanup_sync(ui_components: Dict[str, Any], cleanup_info: Dict[str, Any], logger) -> None:
    """Execute streamlined cleanup dengan optimized dual-level progress"""
    try:
        # Get progress tracker
        progress_tracker = ui_components.get('progress_tracker')
        if not progress_tracker:
            logger.error("❌ Progress tracker tidak ditemukan")
            show_status_safe("❌ Progress tracker tidak tersedia", "error", ui_components)
            return
        
        # Show progress
        progress_tracker.show("Cleanup Dataset")
        
        # Create organizer dengan progress integration
        organizer = DatasetOrganizer(logger)
        
        # Setup organizer dengan optimized progress callback
        organizer.set_progress_callback(_create_streamlined_cleanup_progress_callback(progress_tracker, logger, cleanup_info))
        
        # Execute cleanup
        result = organizer.cleanup_all_dataset_folders()
        
        # Handle result dengan streamlined status reporting
        if result['status'] == 'success':
            files_removed = result['stats']['total_files_removed']
            folders_cleaned = len(result['stats']['folders_cleaned'])
            
            success_msg = f"✅ Berhasil membersihkan {cleanup_info['total_files']} file ({cleanup_info['total_size_mb']:.1f} MB)"
            progress_tracker.complete(success_msg)
            show_status_safe(success_msg, "success", ui_components)
            logger.success(f"🧹 {success_msg}")
            
            # Additional success details dalam log
            with ui_components.get('log_output'):
                from IPython.display import display, HTML
                display(HTML(f"""
                <div style="padding: 10px; background: #e8f5e8; border-radius: 4px; margin: 5px 0;">
                    <strong style="color: #2e7d32;">🎉 Cleanup Summary:</strong><br>
                    📁 Folders cleaned: {', '.join(result['stats']['folders_cleaned'])}<br>
                    💾 Space freed: ~{cleanup_info['total_size_mb']:.1f} MB<br>
                    ⏱️ Duration: {result.get('duration', 'N/A')} seconds
                </div>
                """))
            
        elif result['status'] == 'empty':
            info_msg = "ℹ️ Tidak ada file yang perlu dibersihkan"
            progress_tracker.complete(info_msg)
            show_status_safe(info_msg, "info", ui_components)
            logger.info(f"🧹 {info_msg}")
            
        else:
            error_msg = f"❌ Error saat cleanup: {result.get('message', 'Unknown error')}"
            progress_tracker.error(error_msg)
            show_status_safe(error_msg, "error", ui_components)
            logger.error(f"🧹 {error_msg}")
            
    except Exception as e:
        error_msg = f"❌ Error saat cleanup: {str(e)}"
        progress_tracker = ui_components.get('progress_tracker')
        progress_tracker and progress_tracker.error(error_msg)
        show_status_safe(error_msg, "error", ui_components)
        logger.error(error_msg)

def _create_streamlined_cleanup_progress_callback(progress_tracker, logger, cleanup_info: Dict[str, Any]) -> Callable:
    """Create streamlined progress callback untuk cleanup operations"""
    
    def cleanup_progress_callback(step: str, current: int, total: int, message: str):
        """Streamlined progress callback dengan optimized level mapping"""
        try:
            percentage = min(100, max(0, int((current / total) * 100) if total > 0 else 0))
            
            # Map cleanup steps ke dual-level progress dengan optimized calculation
            if step == 'scan':
                # Scanning phase (0-20%)
                progress_tracker.update_overall(int(percentage * 0.2), "🔍 Scanning files")
                progress_tracker.update_current(percentage, message)
            elif step == 'cleanup':
                # Main cleanup phase (20-90%)
                overall_pct = 20 + int(percentage * 0.7)
                progress_tracker.update_overall(overall_pct, f"🧹 Cleanup: {percentage}%")
                progress_tracker.update_current(percentage, message)
            elif step == 'verify':
                # Verification phase (90-100%)
                overall_pct = 90 + int(percentage * 0.1)
                progress_tracker.update_overall(overall_pct, "✅ Verifikasi cleanup")
                progress_tracker.update_current(percentage, message)
            else:
                # Generic progress
                progress_tracker.update_current(percentage, message)
                
        except Exception as e:
            logger.debug(f"🔍 Cleanup progress callback error: {str(e)}")
    
    return cleanup_progress_callback

def _show_in_confirmation_area(ui_components: Dict[str, Any], dialog_widget) -> None:
    """Show dialog dalam confirmation area dengan optimized display"""
    confirmation_area = ui_components.get('confirmation_area')
    if confirmation_area:
        setattr(confirmation_area.layout, 'display', 'block')
        setattr(confirmation_area.layout, 'visibility', 'visible')
        with confirmation_area:
            confirmation_area.clear_output(wait=True)
            from IPython.display import display
            display(dialog_widget)

def _clear_confirmation_area(ui_components: Dict[str, Any]) -> None:
    """Clear confirmation area dengan one-liner cleanup"""
    confirmation_area = ui_components.get('confirmation_area')
    confirmation_area and (
        confirmation_area.clear_output(wait=True),
        setattr(confirmation_area.layout, 'display', 'none'),
        setattr(confirmation_area.layout, 'visibility', 'hidden')
    )

# Optimized utilities dengan one-liner style
get_cleanup_status = lambda ui: {'ready_for_cleanup': _get_streamlined_cleanup_info()['has_data'], 'confirmation_available': 'confirmation_area' in ui, 'progress_tracker_available': 'progress_tracker' in ui}
has_cleanup_data = lambda: _get_streamlined_cleanup_info()['has_data']
get_cleanup_targets = lambda: _get_streamlined_cleanup_info()['targets_with_data']
estimate_cleanup_duration = lambda: f"{max(5, _get_streamlined_cleanup_info()['total_files'] // 1000)} detik" if _get_streamlined_cleanup_info()['total_files'] > 0 else "Instant"
format_cleanup_summary = lambda info: f"Files: {info['total_files']:,} | Size: {info['total_size_mb']:.1f} MB | Targets: {len(info['targets_with_data'])}"