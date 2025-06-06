"""
File: smartcash/ui/dataset/downloader/handlers/cleanup_handler.py
Deskripsi: Cleanup handler untuk membersihkan /data/{train,valid,test} dan /data/downloads
"""

import shutil
from pathlib import Path
from typing import Dict, Any
from smartcash.ui.components.dialogs import confirm_cleanup

def setup_cleanup_handler(ui_components: Dict[str, Any], config: Dict[str, Any], logger) -> None:
    """Setup cleanup button handler dengan confirmation dialog"""
    cleanup_button = ui_components.get('cleanup_button')
    if not cleanup_button:
        return
    
    def on_cleanup_click(button):
        """Handle cleanup button click dengan confirmation"""
        # Scan directories untuk cleanup info
        cleanup_info = _scan_cleanup_directories(logger)
        
        if not cleanup_info['has_files']:
            logger.info("✨ Tidak ada file untuk dibersihkan")
            return
        
        # Show confirmation dengan cleanup details
        confirm_cleanup(
            on_confirm=lambda btn: _execute_cleanup(cleanup_info, logger),
            on_cancel=lambda btn: logger.info("🚫 Cleanup dibatalkan")
        )
    
    cleanup_button.on_click(on_cleanup_click)
    logger.debug("✅ Cleanup handler siap")

def _scan_cleanup_directories(logger) -> Dict[str, Any]:
    """Scan directories yang akan dibersihkan dengan file count"""
    data_path = Path('data')
    cleanup_dirs = ['train', 'valid', 'test', 'downloads']
    
    scan_results = {
        'directories': {},
        'total_files': 0,
        'total_size_mb': 0,
        'has_files': False
    }
    
    for dir_name in cleanup_dirs:
        dir_path = data_path / dir_name
        dir_info = _scan_directory(dir_path)
        
        scan_results['directories'][dir_name] = dir_info
        scan_results['total_files'] += dir_info['file_count']
        scan_results['total_size_mb'] += dir_info['size_mb']
        
        if dir_info['file_count'] > 0:
            scan_results['has_files'] = True
    
    logger.debug(f"🔍 Cleanup scan: {scan_results['total_files']} files, {scan_results['total_size_mb']:.1f}MB")
    return scan_results

def _scan_directory(dir_path: Path) -> Dict[str, Any]:
    """Scan single directory dengan file count dan size dengan one-liner calculations"""
    if not dir_path.exists():
        return {'exists': False, 'file_count': 0, 'size_mb': 0}
    
    try:
        # Count files dan calculate size
        all_files = list(dir_path.rglob('*'))
        files_only = [f for f in all_files if f.is_file()]
        total_size = sum(f.stat().st_size for f in files_only if f.exists())
        
        return {
            'exists': True,
            'file_count': len(files_only),
            'size_mb': total_size / (1024 * 1024)
        }
    except Exception as e:
        return {'exists': True, 'file_count': 0, 'size_mb': 0, 'error': str(e)}

def _execute_cleanup(cleanup_info: Dict[str, Any], logger) -> None:
    """Execute cleanup dengan progress feedback"""
    logger.info("🧹 Memulai cleanup...")
    
    cleaned_dirs = 0
    cleaned_files = 0
    
    for dir_name, dir_info in cleanup_info['directories'].items():
        if not dir_info['exists'] or dir_info['file_count'] == 0:
            continue
        
        dir_path = Path('data') / dir_name
        
        try:
            # Remove directory completely
            if dir_path.exists():
                shutil.rmtree(dir_path)
                cleaned_dirs += 1
                cleaned_files += dir_info['file_count']
                logger.info(f"🗑️ Cleaned {dir_name}: {dir_info['file_count']} files")
                
        except Exception as e:
            logger.error(f"❌ Error cleaning {dir_name}: {str(e)}")
    
    # Summary
    if cleaned_files > 0:
        logger.success(f"✅ Cleanup selesai: {cleaned_files} files dari {cleaned_dirs} directories")
    else:
        logger.info("ℹ️ Tidak ada file yang dibersihkan")

# Export
__all__ = ['setup_cleanup_handler']