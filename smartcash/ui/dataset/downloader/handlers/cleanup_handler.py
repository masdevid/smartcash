"""
File: smartcash/ui/dataset/downloader/handlers/cleanup_handler.py
Deskripsi: Regenerated cleanup handler dengan complete implementation dan proper dialog confirmation
"""

from typing import Dict, Any, Callable
from smartcash.ui.utils.fallback_utils import show_status_safe
from smartcash.ui.components.confirmation_dialog import create_destructive_confirmation
from smartcash.dataset.services.organizer.dataset_organizer import DatasetOrganizer
from smartcash.dataset.utils.path_validator import get_path_validator

def setup_cleanup_handler(ui_components: Dict[str, Any], config: Dict[str, Any], logger) -> Callable:
    """Setup cleanup handler dengan comprehensive confirmation dan three-level progress"""
    
    def handle_cleanup(button):
        """Handle cleanup operation dengan proper validation dan confirmation flow"""
        try:
            # Check existing data untuk konfirmasi dengan one-liner info gathering
            cleanup_info = _get_comprehensive_cleanup_info()
            
            # Early return jika tidak ada file dengan one-liner check
            cleanup_info['total_files'] == 0 and show_status_safe("ℹ️ Tidak ada file untuk dibersihkan", "info", ui_components) and None
            
            if cleanup_info['total_files'] > 0:
                # Show comprehensive confirmation dialog
                _show_comprehensive_cleanup_confirmation(ui_components, cleanup_info, logger)
            
        except Exception as e:
            logger.error(f"❌ Error cleanup handler: {str(e)}")
            show_status_safe(f"❌ Error: {str(e)}", "error", ui_components)
    
    return handle_cleanup

def _get_comprehensive_cleanup_info() -> Dict[str, Any]:
    """Get comprehensive informasi tentang file yang akan dibersihkan dengan detailed counting"""
    try:
        path_validator = get_path_validator()
        dataset_paths = path_validator.get_dataset_paths()
        
        # Comprehensive cleanup targets dengan one-liner path mapping
        cleanup_targets = {
            'train': {'path': dataset_paths['train'], 'description': 'Training dataset', 'icon': '🏋️'},
            'valid': {'path': dataset_paths['valid'], 'description': 'Validation dataset', 'icon': '✅'},
            'test': {'path': dataset_paths['test'], 'description': 'Test dataset', 'icon': '🧪'},
            'downloads': {'path': dataset_paths['downloads'], 'description': 'Downloaded files', 'icon': '📥'},
            'preprocessed': {'path': dataset_paths.get('preprocessed', ''), 'description': 'Preprocessed data', 'icon': '⚙️'},
            'augmented': {'path': dataset_paths.get('augmented', ''), 'description': 'Augmented data', 'icon': '🔄'},
            'backup': {'path': dataset_paths.get('backup', ''), 'description': 'Backup files', 'icon': '💾'}
        }
        
        target_info = {}
        total_files = 0
        total_size_mb = 0
        
        # Comprehensive file analysis dengan one-liner processing
        from pathlib import Path
        
        for target, target_config in cleanup_targets.items():
            if not target_config['path']:
                target_info[target] = {'path': '', 'files': 0, 'size_mb': 0, 'exists': False, **target_config}
                continue
                
            path = Path(target_config['path'])
            if path.exists():
                # One-liner file counting dan size calculation
                files = [f for f in path.rglob('*') if f.is_file()]
                file_count = len(files)
                size_bytes = sum(f.stat().st_size for f in files if f.exists())
                size_mb = size_bytes / (1024 * 1024)
                
                target_info[target] = {
                    'path': str(path), 'files': file_count, 'size_mb': size_mb, 'exists': True, **target_config
                }
                total_files += file_count
                total_size_mb += size_mb
            else:
                target_info[target] = {'path': str(path), 'files': 0, 'size_mb': 0, 'exists': False, **target_config}
        
        return {
            'total_files': total_files,
            'total_size_mb': total_size_mb,
            'targets': target_info,
            'has_data': total_files > 0,
            'targets_with_data': [name for name, info in target_info.items() if info['files'] > 0]
        }
        
    except Exception as e:
        return {'total_files': 0, 'total_size_mb': 0, 'targets': {}, 'has_data': False, 'error': str(e)}

def _show_comprehensive_cleanup_confirmation(ui_components: Dict[str, Any], cleanup_info: Dict[str, Any], logger) -> None:
    """Show comprehensive destructive confirmation dengan detailed file breakdown"""
    
    total_files = cleanup_info['total_files']
    total_size_mb = cleanup_info['total_size_mb']
    targets_with_data = cleanup_info['targets_with_data']
    
    # Format comprehensive file breakdown dengan icons dan descriptions
    file_breakdown_lines = []
    for target in targets_with_data:
        info = cleanup_info['targets'][target]
        file_breakdown_lines.append(
            f"{info['icon']} **{info['description']}**: {info['files']:,} files ({info['size_mb']:.1f} MB)"
        )
    
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
❌ Backup otomatis **TIDAK TERSEDIA** untuk cleanup
💾 Pastikan sudah **BACKUP DATA PENTING** secara manual

🔄 **Yang akan terjadi:**
1. Scan semua file dalam direktori target
2. Hapus semua file dan folder secara permanent
3. Bersihkan direktori kosong
4. Verifikasi cleanup completed

⚡ **Yakin ingin melanjutkan cleanup destruktif ini?**""",
        on_confirm=lambda b: (_clear_confirmation_area(ui_components), _execute_comprehensive_cleanup_sync(ui_components, cleanup_info, logger)),
        on_cancel=lambda b: _clear_confirmation_area(ui_components),
        item_name="Dataset Complete",
        confirm_text="🗑️ Ya, Hapus SEMUA",
        cancel_text="❌ Batal"
    )
    
    _show_in_confirmation_area(ui_components, confirmation_dialog)

def _execute_comprehensive_cleanup_sync(ui_components: Dict[str, Any], cleanup_info: Dict[str, Any], logger) -> None:
    """Execute comprehensive cleanup operation dengan detailed dual-level progress tracking"""
    try:
        # Initialize dual-level progress tracking untuk cleanup
        progress_tracker = ui_components.get('progress_tracker')
        if progress_tracker:
            # Reset progress tracker terlebih dahulu
            progress_tracker.reset()
            # Tampilkan container progress
            if 'container' in ui_components:
                ui_components['container'].layout.display = 'block'
            # Update progress awal
            progress_tracker.update('level1', 0, "🧹 Memulai comprehensive cleanup dataset...")
            progress_tracker.update('level2', 0, "Menyiapkan proses cleanup...")
        
        # Create organizer dengan progress integration
        organizer = DatasetOrganizer(logger)
        
        # Setup organizer dengan progress callback
        organizer = DatasetOrganizer(logger=logger)
        if progress_tracker:
            organizer.set_progress_callback(
                _create_comprehensive_cleanup_progress_callback(progress_tracker, logger, cleanup_info)
            )
        
        # Execute comprehensive cleanup
        result = organizer.cleanup_all_dataset_folders()
        
        # Handle result dengan comprehensive status reporting
        if result['status'] == 'success':
            files_removed = result['stats']['total_files_removed']
            folders_cleaned = len(result['stats']['folders_cleaned'])
            
            # Comprehensive success message dengan detailed stats
            success_msg = f"✅ Berhasil membersihkan {cleanup_info['total_files']} file ({cleanup_info['total_size_mb']:.1f} MB)"
            if progress_tracker:
                progress_tracker.update('level1', 100, success_msg)
                progress_tracker.update('level2', 100, "Pembersihan selesai")
                # Sembunyikan progress setelah delay
                import time
                time.sleep(0.5)
                if 'container' in ui_components:
                    ui_components['container'].layout.display = 'none'
            show_status_safe(success_msg, "success", ui_components)
            logger.success(f"🧹 {success_msg}")
            
            # Additional success details dalam log
            with ui_components.get('log_output', type('MockContext', (), {'__enter__': lambda: None, '__exit__': lambda *args: None})())():
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
            if progress_tracker:
                progress_tracker.update('level1', 100, info_msg)
                progress_tracker.update('level2', 100, "Tidak ada yang dibersihkan")
                # Sembunyikan progress setelah delay
                import time
                time.sleep(0.5)
                if 'container' in ui_components:
                    ui_components['container'].layout.display = 'none'
            show_status_safe(info_msg, "info", ui_components)
            logger.info(f"🧹 {info_msg}")
            
        else:
            error_msg = f"❌ Error saat cleanup: {str(result['error'])}"
            if progress_tracker:
                progress_tracker.error('level1', error_msg)
                progress_tracker.error('level2', "Error saat cleanup")
                # Sembunyikan progress setelah delay
                import time
                time.sleep(1.0)
                if 'container' in ui_components:
                    ui_components['container'].layout.display = 'none'
            show_status_safe(error_msg, "error", ui_components)
            logger.error(f"🧹 {error_msg}")
            
    except Exception as e:
        error_msg = f"❌ Error saat comprehensive cleanup: {str(e)}"
        progress_tracker = ui_components.get('progress_tracker')
        if progress_tracker:
            progress_tracker.error('level1', error_msg)
            progress_tracker.error('level2', "Error saat cleanup")
            # Sembunyikan progress setelah delay
            import time
            time.sleep(1.0)
            if 'container' in ui_components:
                ui_components['container'].layout.display = 'none'
        show_status_safe(error_msg, "error", ui_components)
        logger.error(error_msg)


def _create_comprehensive_cleanup_progress_callback(progress_tracker, logger, cleanup_info: Dict[str, Any]) -> Callable:
    """Create comprehensive progress callback untuk cleanup operations dengan detailed dual-level mapping"""
    
    def comprehensive_progress_callback(step: str, current: int, total: int, message: str):
        """Comprehensive progress callback dengan detailed step mapping dan file counting"""
        try:
            # Map cleanup steps ke comprehensive dual-level progress dengan detailed tracking
            if step == 'scan':
                # Scan step updates dengan dual-level progress mapping
                level1_percentage = int(current * 0.3)  # 0-30% progress untuk scan
                progress_tracker.update('level1', level1_percentage, f"🔍 Scanning: {level1_percentage}%")
                progress_tracker.update('level2', current, f"🔍 {message}")
                
            elif step == 'delete':
                # Delete step updates dengan dual-level progress mapping
                level1_percentage = 30 + int(current * 0.6)  # 30-90% progress untuk delete
                progress_tracker.update('level1', level1_percentage, f"🗑️ Cleanup: {level1_percentage}%")
                progress_tracker.update('level2', current, f"🗑️ {message}")
                
            elif step == 'complete':
                # Complete step updates dengan dual-level progress mapping
                progress_tracker.update('level1', 100, f"✅ Cleanup completed")
                progress_tracker.update('level2', 100, f"✅ {message}")
                
            elif step == 'verify':
                # Verify step updates dengan dual-level progress mapping
                level1_percentage = 90 + int(current * 0.1)  # 90-100% progress untuk verify
                progress_tracker.update('level1', level1_percentage, f"🔍 Verifying: {level1_percentage}%")
                progress_tracker.update('level2', current, f"🔍 {message}")
                
            else:
                # Generic progress update untuk unknown steps
                progress_tracker.update('level1', min(99, current), message)
                progress_tracker.update('level2', current, message)
                
        except Exception as e:
            logger.debug(f"🔍 Comprehensive cleanup progress callback error: {str(e)}")
    
    return comprehensive_progress_callback

def _show_in_confirmation_area(ui_components: Dict[str, Any], dialog_widget) -> None:
    """Show dialog dalam confirmation area dengan proper display management dan visibility control"""
    confirmation_area = ui_components.get('confirmation_area')
    if confirmation_area:
        # Show confirmation area dan display dialog dengan one-liner state management
        setattr(confirmation_area.layout, 'display', 'block'), setattr(confirmation_area.layout, 'visibility', 'visible')
        with confirmation_area:
            confirmation_area.clear_output(wait=True)
            from IPython.display import display
            display(dialog_widget)

def _clear_confirmation_area(ui_components: Dict[str, Any]) -> None:
    """Clear confirmation area dengan comprehensive hiding - one-liner cleanup"""
    confirmation_area = ui_components.get('confirmation_area')
    confirmation_area and (
        confirmation_area.clear_output(wait=True),
        setattr(confirmation_area.layout, 'display', 'none'),
        setattr(confirmation_area.layout, 'visibility', 'hidden')
    )

def get_cleanup_status(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Get comprehensive cleanup status untuk debugging dan monitoring"""
    try:
        cleanup_info = _get_comprehensive_cleanup_info()
        return {
            'ready_for_cleanup': cleanup_info['has_data'],
            'total_files': cleanup_info['total_files'],
            'total_size_mb': cleanup_info['total_size_mb'],
            'target_count': len(cleanup_info['targets_with_data']),
            'targets_with_data': cleanup_info['targets_with_data'],
            'confirmation_available': 'confirmation_area' in ui_components,
            'progress_tracker_available': 'progress_tracker' in ui_components
        }
    except Exception as e:
        return {'ready_for_cleanup': False, 'error': str(e)}

def estimate_cleanup_time(cleanup_info: Dict[str, Any]) -> Dict[str, Any]:
    """Estimate cleanup time berdasarkan file count dan size dengan one-liner calculations"""
    try:
        total_files = cleanup_info['total_files']
        total_size_mb = cleanup_info['total_size_mb']
        
        # Rough estimates berdasarkan file operations
        # Assume: ~1000 files per second untuk deletion, ~50MB per second untuk I/O
        file_time_seconds = total_files / 1000
        io_time_seconds = total_size_mb / 50
        estimated_seconds = max(file_time_seconds, io_time_seconds) + 5  # Add 5s overhead
        
        # Format time dengan one-liner conditional formatting
        formatted_time = (f"{estimated_seconds:.0f} detik" if estimated_seconds < 60 else
                         f"{estimated_seconds/60:.1f} menit" if estimated_seconds < 3600 else
                         f"{estimated_seconds/3600:.1f} jam")
        
        return {
            'estimated_seconds': estimated_seconds,
            'formatted_time': formatted_time,
            'file_factor': file_time_seconds,
            'io_factor': io_time_seconds
        }
    except Exception as e:
        return {'estimated_seconds': 0, 'formatted_time': 'Unknown', 'error': str(e)}

# One-liner utilities untuk cleanup operations
get_cleanup_info = lambda: _get_comprehensive_cleanup_info()
has_cleanup_data = lambda: _get_comprehensive_cleanup_info()['has_data']
get_cleanup_targets = lambda: _get_comprehensive_cleanup_info()['targets_with_data']
estimate_cleanup_duration = lambda: estimate_cleanup_time(_get_comprehensive_cleanup_info())['formatted_time']