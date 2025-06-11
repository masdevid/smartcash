"""
File: smartcash/dataset/preprocessor/api/cleanup_api.py
Deskripsi: Updated cleanup API menggunakan FileNamingManager patterns
"""

from typing import Dict, Any, List, Union, Optional, Callable
from pathlib import Path

from smartcash.common.logger import get_logger
from smartcash.common.utils.file_naming_manager import create_file_naming_manager

def cleanup_preprocessing_files(data_dir: Union[str, Path],
                               target: str = 'preprocessed',
                               splits: List[str] = None,
                               confirm: bool = False,
                               progress_callback: Optional[Callable] = None,
                               ui_components: Dict[str, Any] = None) -> Dict[str, Any]:
    """🧹 Cleanup dengan FileNamingManager patterns"""
    try:
        logger = get_logger(__name__)
        
        if not confirm:
            return {'success': False, 'message': "❌ Cleanup requires confirmation", 'files_removed': 0}
        
        # Setup progress bridge
        from ..utils.progress_bridge import create_preprocessing_bridge
        progress_bridge = None
        if ui_components and progress_callback:
            progress_bridge = create_preprocessing_bridge(ui_components)
            progress_bridge.register_callback(progress_callback)
        
        preprocessed_dir = Path(data_dir).parent / 'preprocessed' if 'preprocessed' not in str(data_dir) else Path(data_dir)
        
        if not preprocessed_dir.exists():
            return {'success': False, 'message': f"❌ Preprocessed directory not found: {preprocessed_dir}", 'files_removed': 0}
        
        if splits is None:
            splits = [d.name for d in preprocessed_dir.iterdir() if d.is_dir() and d.name in ['train', 'valid', 'test']]
        
        # Setup progress tracking
        if progress_bridge:
            progress_bridge.setup_split_processing(splits)
        
        total_removed = 0
        cleanup_stats = {}
        
        for split in splits:
            if progress_bridge:
                progress_bridge.start_split(split)
            
            split_stats = _cleanup_split_files_with_progress(preprocessed_dir / split, target, progress_bridge)
            cleanup_stats[split] = split_stats
            total_removed += split_stats['files_removed']
            
            if progress_bridge:
                progress_bridge.complete_split(split)
        
        return {'success': True, 'message': f"✅ Cleanup completed: {total_removed} files removed", 'target': target, 'files_removed': total_removed, 'by_split': cleanup_stats}
        
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"❌ Cleanup error: {str(e)}")
        return {'success': False, 'message': f"❌ Error: {str(e)}", 'files_removed': 0}

def get_cleanup_preview(data_dir: Union[str, Path],
                       target: str = 'preprocessed',
                       splits: List[str] = None) -> Dict[str, Any]:
    """👀 Preview cleanup menggunakan naming manager"""
    try:
        from ..core.file_processor import FileProcessor
        
        preprocessed_dir = Path(data_dir).parent / 'preprocessed' if 'preprocessed' not in str(data_dir) else Path(data_dir)
        
        if not preprocessed_dir.exists():
            return {'success': False, 'message': f"❌ Preprocessed directory not found: {preprocessed_dir}"}
        
        if splits is None:
            splits = [d.name for d in preprocessed_dir.iterdir() if d.is_dir() and d.name in ['train', 'valid', 'test']]
        
        naming_manager = create_file_naming_manager()
        fp = FileProcessor()
        preview = {'success': True, 'target': target, 'total_files': 0, 'total_size_mb': 0, 'by_split': {}}
        
        for split in splits:
            split_path = preprocessed_dir / split
            if not split_path.exists():
                continue
            
            files_to_remove = []
            
            if target in ['preprocessed', 'both']:
                # Scan preprocessed files
                for file_type in ['preprocessed']:
                    prefix = naming_manager.get_prefix(file_type)
                    if prefix:
                        # Images
                        npy_files = fp.scan_files(split_path / 'images', prefix, {'.npy'})
                        files_to_remove.extend(npy_files)
                        
                        # Labels
                        label_files = fp.scan_files(split_path / 'labels', prefix, {'.txt'})
                        files_to_remove.extend(label_files)
                        
                        # Metadata files
                        for npy_file in npy_files:
                            meta_file = npy_file.with_suffix('.meta.json')
                            if meta_file.exists():
                                files_to_remove.append(meta_file)
            
            if target in ['augmented', 'both']:
                # Scan augmented files
                for file_type in ['augmented']:
                    prefix = naming_manager.get_prefix(file_type)
                    if prefix:
                        # Images dengan variance pattern
                        aug_files = fp.scan_files(split_path / 'images', prefix, {'.npy', '.jpg'})
                        files_to_remove.extend(aug_files)
                        
                        # Labels dengan variance pattern
                        aug_labels = fp.scan_files(split_path / 'labels', prefix, {'.txt'})
                        files_to_remove.extend(aug_labels)
            
            if target in ['samples', 'both']:
                # Scan sample files
                for file_type in ['sample', 'augmented_sample']:
                    prefix = naming_manager.get_prefix(file_type)
                    if prefix:
                        sample_files = fp.scan_files(split_path / 'images', prefix)
                        files_to_remove.extend(sample_files)
            
            split_size = sum(fp.get_file_info(f).get('size_mb', 0) for f in files_to_remove)
            
            preview['by_split'][split] = {
                'files_count': len(files_to_remove),
                'size_mb': round(split_size, 2),
                'file_list': [str(f) for f in files_to_remove[:10]]
            }
            
            preview['total_files'] += len(files_to_remove)
            preview['total_size_mb'] += split_size
        
        preview['total_size_mb'] = round(preview['total_size_mb'], 2)
        return preview
        
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"❌ Cleanup preview error: {str(e)}")
        return {'success': False, 'message': f"❌ Error: {str(e)}"}

def _cleanup_split_files_with_progress(split_path: Path, target: str, progress_bridge=None) -> Dict[str, Any]:
    """🗑️ Internal cleanup menggunakan naming manager"""
    from ..core.file_processor import FileProcessor
    
    naming_manager = create_file_naming_manager()
    fp = FileProcessor()
    files_removed = 0
    size_removed = 0
    
    files_to_remove = []
    images_dir = split_path / 'images'
    labels_dir = split_path / 'labels'
    
    if target in ['preprocessed', 'both']:
        for file_type in ['preprocessed']:
            prefix = naming_manager.get_prefix(file_type)
            if prefix and images_dir.exists():
                npy_files = fp.scan_files(images_dir, prefix, {'.npy'})
                files_to_remove.extend(npy_files)
                
                for npy_file in npy_files:
                    meta_file = npy_file.with_suffix('.meta.json')
                    if meta_file.exists():
                        files_to_remove.append(meta_file)
            
            if prefix and labels_dir.exists():
                label_files = fp.scan_files(labels_dir, prefix, {'.txt'})
                files_to_remove.extend(label_files)
    
    if target in ['augmented', 'both']:
        for file_type in ['augmented']:
            prefix = naming_manager.get_prefix(file_type)
            if prefix and images_dir.exists():
                aug_files = fp.scan_files(images_dir, prefix, {'.npy', '.jpg'})
                files_to_remove.extend(aug_files)
            
            if prefix and labels_dir.exists():
                aug_labels = fp.scan_files(labels_dir, prefix, {'.txt'})
                files_to_remove.extend(aug_labels)
    
    if target in ['samples', 'both']:
        for file_type in ['sample', 'augmented_sample']:
            prefix = naming_manager.get_prefix(file_type)
            if prefix and images_dir.exists():
                sample_files = fp.scan_files(images_dir, prefix)
                files_to_remove.extend(sample_files)
    
    # Remove files dengan progress
    total_files = len(files_to_remove)
    for i, file_path in enumerate(files_to_remove):
        try:
            if progress_bridge:
                progress_bridge.update_split_progress(i + 1, total_files, f"Removing {file_path.name}")
            
            if file_path.exists():
                file_size = fp.get_file_info(file_path).get('size_mb', 0)
                file_path.unlink()
                files_removed += 1
                size_removed += file_size
        except Exception as e:
            logger = get_logger(__name__)
            logger.warning(f"⚠️ Failed to remove {file_path}: {str(e)}")
    
    return {'files_removed': files_removed, 'size_removed_mb': round(size_removed, 2), 'target': target}

def get_cleanup_summary(data_dir: Union[str, Path], target: str = 'preprocessed') -> Dict[str, Any]:
    """📊 Cleanup summary menggunakan naming manager"""
    try:
        valid_targets = ['preprocessed', 'augmented', 'samples', 'both']
        if target not in valid_targets:
            return {'success': False, 'message': f"❌ Invalid target '{target}'. Valid: {valid_targets}"}
        
        preview = get_cleanup_preview(data_dir, target)
        if not preview['success']:
            return preview
        
        naming_manager = create_file_naming_manager()
        
        summary = {
            'success': True,
            'target': target,
            'total_files': preview['total_files'],
            'total_size_mb': preview['total_size_mb'],
            'affected_splits': len(preview['by_split']),
            'file_patterns': []
        }
        
        # Add patterns yang akan dihapus
        if target in ['preprocessed', 'both']:
            summary['file_patterns'].extend([
                'pre_{nominal}_{uuid}.npy',
                'pre_{nominal}_{uuid}.txt',
                'pre_{nominal}_{uuid}.meta.json'
            ])
        
        if target in ['augmented', 'both']:
            summary['file_patterns'].extend([
                'aug_{nominal}_{uuid}_{variance}.npy',
                'aug_{nominal}_{uuid}_{variance}.txt'
            ])
        
        if target in ['samples', 'both']:
            summary['file_patterns'].extend([
                'sample_pre_{nominal}_{uuid}.jpg',
                'sample_aug_{nominal}_{uuid}_{variance}.jpg'
            ])
        
        return summary
        
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"❌ Cleanup summary error: {str(e)}")
        return {'success': False, 'message': f"❌ Error: {str(e)}"}

def cleanup_empty_directories(data_dir: Union[str, Path]) -> Dict[str, Any]:
    """📁 Cleanup empty directories"""
    try:
        data_path = Path(data_dir)
        removed_dirs = []
        
        for split_dir in data_path.iterdir():
            if split_dir.is_dir():
                for subdir in ['images', 'labels']:
                    dir_path = split_dir / subdir
                    if dir_path.exists() and not any(dir_path.iterdir()):
                        dir_path.rmdir()
                        removed_dirs.append(str(dir_path))
                
                if split_dir.exists() and not any(split_dir.iterdir()):
                    split_dir.rmdir()
                    removed_dirs.append(str(split_dir))
        
        return {'success': True, 'message': f"✅ Removed {len(removed_dirs)} empty directories", 'removed_directories': removed_dirs}
        
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"❌ Empty directory cleanup error: {str(e)}")
        return {'success': False, 'message': f"❌ Error: {str(e)}"}