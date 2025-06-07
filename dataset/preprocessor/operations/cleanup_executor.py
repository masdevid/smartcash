"""
File: smartcash/dataset/preprocessor/operations/cleanup_executor.py
Deskripsi: Service untuk cleanup operations dengan symlink protection dan comprehensive stats
"""

import shutil
import time
from pathlib import Path
from typing import Dict, Any, Optional, Callable

from smartcash.common.logger import get_logger
from smartcash.dataset.utils.path_validator import get_path_validator


class CleanupExecutor:
    """Service untuk cleanup operations dengan symlink-safe processing dan detailed stats."""
    
    def __init__(self, config: Dict[str, Any], logger=None):
        """Initialize cleanup executor dengan configuration."""
        self.config = config
        self.logger = logger or get_logger()
        self.path_validator = get_path_validator(logger)
        self._progress_callback: Optional[Callable] = None
        
    def register_progress_callback(self, callback: Callable) -> None:
        """Register progress callback untuk cleanup updates."""
        self._progress_callback = callback
    
    def cleanup_preprocessed_data(self, target_split: Optional[str] = None, 
                                 safe_mode: bool = True) -> Dict[str, Any]:
        """
        Cleanup preprocessed data dengan symlink protection dan comprehensive stats.
        
        Args:
            target_split: Target split untuk cleanup (None untuk semua)
            safe_mode: Aktifkan symlink protection
            
        Returns:
            Dictionary hasil cleanup dengan detailed stats
        """
        start_time = time.time()
        preprocessed_dir = Path(self.config.get('preprocessing', {}).get('output_dir', 'data/preprocessed'))
        
        try:
            # Phase 1: Analysis dan preparation (0-15%)
            self._notify_cleanup_progress(5, "Menganalisis data untuk cleanup")
            
            analysis_result = self._analyze_cleanup_targets(preprocessed_dir, target_split)
            if not analysis_result['has_data']:
                return {
                    'success': True, 'message': 'Tidak ada data untuk dibersihkan',
                    'stats': {'files_removed': 0, 'splits_cleaned': 0, 'bytes_freed': 0}
                }
            
            # Safety check untuk symlinks
            if safe_mode:
                safety_result = self._perform_safety_checks(analysis_result['targets'])
                if not safety_result['safe']:
                    return {'success': False, 'message': safety_result['message']}
            
            self._notify_cleanup_progress(15, f"Analysis selesai: {analysis_result['total_files']} files ditemukan")
            
            # Phase 2: Cleanup execution (15-85%)
            cleanup_result = self._execute_cleanup_operation(analysis_result, safe_mode)
            
            # Phase 3: Finalization (85-100%)
            self._notify_cleanup_progress(90, "Finalisasi cleanup dan collecting stats")
            
            final_stats = self._finalize_cleanup_stats(cleanup_result, time.time() - start_time)
            
            self._notify_cleanup_progress(100, f"Cleanup selesai: {final_stats['files_removed']} files dihapus")
            
            self.logger.success(
                f"✅ Cleanup berhasil: {final_stats['files_removed']} files, "
                f"{final_stats['bytes_freed'] / (1024*1024):.1f}MB freed"
            )
            
            return {'success': True, 'message': 'Cleanup completed successfully', 'stats': final_stats}
            
        except Exception as e:
            error_msg = f"Cleanup error: {str(e)}"
            self.logger.error(f"❌ {error_msg}")
            return {'success': False, 'message': error_msg}
    
    def safe_symlink_cleanup(self, directory: Path) -> Dict[str, Any]:
        """Cleanup dengan symlink-aware processing."""
        symlink_stats = {'total_symlinks': 0, 'augmentation_symlinks': 0, 'broken_symlinks': 0, 'removed_symlinks': 0}
        
        try:
            for item in directory.rglob('*'):
                if item.is_symlink():
                    symlink_stats['total_symlinks'] += 1
                    
                    try:
                        target = item.resolve()
                        target_str = str(target).lower()
                        
                        # Detect augmentation symlinks
                        if any(pattern in target_str for pattern in ['augment', 'synthetic', 'generated']):
                            symlink_stats['augmentation_symlinks'] += 1
                            self.logger.debug(f"🔗 Augmentation symlink detected: {item.name}")
                        
                        # Remove symlink (data asli tetap aman)
                        item.unlink()
                        symlink_stats['removed_symlinks'] += 1
                        
                    except OSError:
                        # Broken symlink
                        symlink_stats['broken_symlinks'] += 1
                        try:
                            item.unlink()
                            symlink_stats['removed_symlinks'] += 1
                        except OSError:
                            pass
            
            return {'success': True, 'symlink_stats': symlink_stats}
            
        except Exception as e:
            return {'success': False, 'message': f'Symlink cleanup error: {str(e)}'}
    
    def calculate_cleanup_stats(self, directory: Path) -> Dict[str, Any]:
        """Calculate comprehensive cleanup statistics."""
        stats = {
            'total_files': 0, 'total_directories': 0, 'total_size_bytes': 0,
            'file_types': {}, 'split_breakdown': {}
        }
        
        try:
            if not directory.exists():
                return stats
            
            # Scan directory structure
            for item in directory.rglob('*'):
                if item.is_file():
                    stats['total_files'] += 1
                    stats['total_size_bytes'] += item.stat().st_size
                    
                    # Track file types
                    extension = item.suffix.lower()
                    stats['file_types'][extension] = stats['file_types'].get(extension, 0) + 1
                    
                    # Track split breakdown
                    split_part = self._extract_split_from_path(item, directory)
                    if split_part:
                        if split_part not in stats['split_breakdown']:
                            stats['split_breakdown'][split_part] = {'files': 0, 'size_bytes': 0}
                        stats['split_breakdown'][split_part]['files'] += 1
                        stats['split_breakdown'][split_part]['size_bytes'] += item.stat().st_size
                
                elif item.is_dir():
                    stats['total_directories'] += 1
            
            return stats
            
        except Exception as e:
            self.logger.debug(f"🔧 Stats calculation error: {str(e)}")
            return stats
    
    def _analyze_cleanup_targets(self, preprocessed_dir: Path, target_split: Optional[str]) -> Dict[str, Any]:
        """Analyze cleanup targets dan calculate comprehensive stats."""
        analysis = {'has_data': False, 'targets': [], 'total_files': 0, 'total_size': 0}
        
        if not preprocessed_dir.exists():
            return analysis
        
        # Determine target splits
        if target_split:
            target_splits = [target_split]
        else:
            target_splits = ['train', 'valid', 'test']
        
        # Analyze each target
        for split in target_splits:
            split_path = preprocessed_dir / split
            if split_path.exists():
                split_stats = self.calculate_cleanup_stats(split_path)
                if split_stats['total_files'] > 0:
                    analysis['targets'].append({
                        'split': split, 'path': split_path, 'stats': split_stats
                    })
                    analysis['total_files'] += split_stats['total_files']
                    analysis['total_size'] += split_stats['total_size_bytes']
                    analysis['has_data'] = True
        
        # Also check metadata directory
        metadata_path = preprocessed_dir / 'metadata'
        if metadata_path.exists():
            metadata_stats = self.calculate_cleanup_stats(metadata_path)
            if metadata_stats['total_files'] > 0:
                analysis['targets'].append({
                    'split': 'metadata', 'path': metadata_path, 'stats': metadata_stats
                })
                analysis['total_files'] += metadata_stats['total_files']
                analysis['total_size'] += metadata_stats['total_size_bytes']
                analysis['has_data'] = True
        
        return analysis
    
    def _perform_safety_checks(self, targets: list) -> Dict[str, Any]:
        """Perform safety checks before cleanup."""
        safety_result = {'safe': True, 'message': '', 'warnings': []}
        
        for target in targets:
            target_path = target['path']
            
            # Check symlink safety
            symlink_safety = self.path_validator.is_symlink_safe_operation(str(target_path), 'cleanup')
            
            if not symlink_safety['safe']:
                safety_result['safe'] = False
                safety_result['message'] = f"Unsafe cleanup detected: {'; '.join(symlink_safety['warnings'])}"
                return safety_result
            
            # Collect warnings
            safety_result['warnings'].extend(symlink_safety['warnings'])
        
        return safety_result
    
    def _execute_cleanup_operation(self, analysis_result: Dict[str, Any], safe_mode: bool) -> Dict[str, Any]:
        """Execute cleanup operation dengan progress tracking.
        Hanya menghapus file gambar dan label, mempertahankan symlink augmentasi 'aug_*'"""
        cleanup_stats = {'files_removed': 0, 'splits_cleaned': 0, 'bytes_freed': 0, 'symlinks_preserved': 0}
        targets = analysis_result['targets']
        
        import os
        
        for i, target in enumerate(targets):
            target_path = target['path']
            split_name = target['split']
            
            # Progress calculation
            progress_base = 15 + (i / len(targets)) * 70
            self._notify_cleanup_progress(int(progress_base), f"Cleaning {split_name}")
            
            try:
                # Calculate size before cleanup
                target_size = target['stats']['total_size_bytes']
                files_removed = 0
                bytes_freed = 0
                
                if target_path.exists():
                    # Hapus file di subdirektori images dan labels, pertahankan struktur direktori
                    for subdir in ['images', 'labels']:
                        subdir_path = target_path / subdir
                        if subdir_path.exists():
                            # Hapus file satu per satu, pertahankan symlink aug_*
                            for file_path in subdir_path.glob('*.*'):
                                if file_path.is_file() and not file_path.is_symlink() and not file_path.name.startswith('aug_'):
                                    # Hanya hapus file reguler (bukan symlink) dan bukan file aug_*
                                    file_size = file_path.stat().st_size
                                    os.remove(file_path)
                                    files_removed += 1
                                    bytes_freed += file_size
                                elif file_path.is_symlink() and file_path.name.startswith('aug_'):
                                    # Hitung symlink yang dipertahankan
                                    cleanup_stats['symlinks_preserved'] += 1
                    
                    cleanup_stats['files_removed'] += files_removed
                    cleanup_stats['bytes_freed'] += bytes_freed
                    cleanup_stats['splits_cleaned'] += 1
                    
                    self.logger.info(f"🧹 Cleaned {split_name}: {files_removed} files, {cleanup_stats['symlinks_preserved']} symlinks preserved")
            
            except Exception as e:
                self.logger.warning(f"⚠️ Error cleaning {split_name}: {str(e)}")
        
        return cleanup_stats
    
    def _finalize_cleanup_stats(self, cleanup_result: Dict[str, Any], processing_time: float) -> Dict[str, Any]:
        """Finalize cleanup statistics dengan additional metrics."""
        final_stats = {
            **cleanup_result,
            'processing_time': processing_time,
            'cleanup_rate_files_per_sec': cleanup_result['files_removed'] / processing_time if processing_time > 0 else 0,
            'size_freed_mb': cleanup_result['bytes_freed'] / (1024 * 1024)
        }
        
        return final_stats
    
    def _extract_split_from_path(self, file_path: Path, base_path: Path) -> Optional[str]:
        """Extract split name dari file path."""
        try:
            relative_path = file_path.relative_to(base_path)
            parts = relative_path.parts
            
            # Check if any part is a known split
            for part in parts:
                if part in ['train', 'valid', 'test', 'metadata']:
                    return part
            
            return None
        except ValueError:
            return None
    
    def _notify_cleanup_progress(self, progress: int, message: str, **kwargs):
        """Internal progress notification untuk cleanup operations."""
        if self._progress_callback:
            try:
                self._progress_callback(progress=progress, message=message, **kwargs)
            except Exception as e:
                self.logger.debug(f"🔧 Cleanup progress callback error: {str(e)}")