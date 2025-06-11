"""
File: smartcash/dataset/preprocessor/api/cleanup_api.py
Deskripsi: Cleanup API untuk preprocessing artifacts dengan configurable options
"""

from typing import Dict, Any, List, Union
from pathlib import Path

from smartcash.common.logger import get_logger

def cleanup_preprocessing_files(data_dir: Union[str, Path],
                               target: str = 'preprocessed',
                               splits: List[str] = None,
                               confirm: bool = False) -> Dict[str, Any]:
    """🧹 Cleanup preprocessing artifacts dengan configurable target
    
    Args:
        data_dir: Base data directory
        target: Target files ('preprocessed', 'samples', 'both')
        splits: Target splits (None = all splits)
        confirm: Safety confirmation required
        
    Returns:
        Dict dengan cleanup results
        
    Example:
        >>> # Clean only preprocessing .npy files
        >>> result = cleanup_preprocessing_files('data', 'preprocessed', confirm=True)
        >>> # Clean only generated samples
        >>> result = cleanup_preprocessing_files('data', 'samples', confirm=True)
        >>> # Clean both
        >>> result = cleanup_preprocessing_files('data', 'both', confirm=True)
    """
    try:
        logger = get_logger(__name__)
        
        if not confirm:
            return {
                'success': False,
                'message': "❌ Cleanup requires confirmation. Set confirm=True",
                'files_removed': 0
            }
        
        valid_targets = ['preprocessed', 'samples', 'both']
        if target not in valid_targets:
            return {
                'success': False,
                'message': f"❌ Invalid target '{target}'. Valid: {valid_targets}",
                'files_removed': 0
            }
        
        data_path = Path(data_dir)
        if not data_path.exists():
            return {
                'success': False,
                'message': f"❌ Data directory not found: {data_dir}",
                'files_removed': 0
            }
        
        # Determine splits
        if splits is None:
            splits = [d.name for d in data_path.iterdir() if d.is_dir() and d.name in ['train', 'valid', 'test']]
        
        total_removed = 0
        cleanup_stats = {}
        
        for split in splits:
            split_stats = _cleanup_split_files(data_path / split, target)
            cleanup_stats[split] = split_stats
            total_removed += split_stats['files_removed']
        
        message = f"✅ Cleanup completed: {total_removed} files removed"
        if target == 'preprocessed':
            message += " (preprocessing .npy files only)"
        elif target == 'samples':
            message += " (sample images only)"
        else:
            message += " (preprocessing .npy and sample images)"
        
        return {
            'success': True,
            'message': message,
            'target': target,
            'files_removed': total_removed,
            'by_split': cleanup_stats
        }
        
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"❌ Cleanup error: {str(e)}")
        return {
            'success': False,
            'message': f"❌ Error: {str(e)}",
            'files_removed': 0
        }

def cleanup_split_files(data_dir: Union[str, Path],
                       split: str,
                       target: str = 'preprocessed',
                       confirm: bool = False) -> Dict[str, Any]:
    """🧹 Cleanup files untuk specific split
    
    Args:
        data_dir: Base data directory
        split: Target split name
        target: Target files ('preprocessed', 'samples', 'both')
        confirm: Safety confirmation
        
    Returns:
        Dict dengan cleanup results
    """
    if not confirm:
        return {
            'success': False,
            'message': "❌ Cleanup requires confirmation",
            'files_removed': 0
        }
    
    data_path = Path(data_dir)
    split_path = data_path / split
    
    if not split_path.exists():
        return {
            'success': False,
            'message': f"❌ Split directory not found: {split_path}",
            'files_removed': 0
        }
    
    stats = _cleanup_split_files(split_path, target)
    
    return {
        'success': True,
        'message': f"✅ Cleaned {split}: {stats['files_removed']} files removed",
        'split': split,
        'target': target,
        'files_removed': stats['files_removed'],
        'details': stats
    }

def get_cleanup_preview(data_dir: Union[str, Path],
                       target: str = 'preprocessed',
                       splits: List[str] = None) -> Dict[str, Any]:
    """👀 Preview files yang akan dihapus tanpa execute cleanup
    
    Args:
        data_dir: Base data directory
        target: Target files ('preprocessed', 'samples', 'both')
        splits: Target splits
        
    Returns:
        Dict dengan preview information
    """
    try:
        from ..core.file_processor import FileProcessor
        
        data_path = Path(data_dir)
        if not data_path.exists():
            return {
                'success': False,
                'message': f"❌ Data directory not found: {data_dir}"
            }
        
        # Determine splits
        if splits is None:
            splits = [d.name for d in data_path.iterdir() if d.is_dir() and d.name in ['train', 'valid', 'test']]
        
        fp = FileProcessor()
        preview = {
            'success': True,
            'target': target,
            'total_files': 0,
            'total_size_mb': 0,
            'by_split': {}
        }
        
        for split in splits:
            split_path = data_path / split
            if not split_path.exists():
                continue
            
            # Get files based on target
            files_to_remove = []
            
            if target in ['preprocessed', 'both']:
                # Preprocessing .npy files (pre_*.npy)
                npy_files = fp.scan_files(split_path / 'images', 'pre_', {'.npy'})
                files_to_remove.extend(npy_files)
                
                # Also remove corresponding .meta.json files
                for npy_file in npy_files:
                    meta_file = npy_file.with_suffix('.meta.json')
                    if meta_file.exists():
                        files_to_remove.append(meta_file)
            
            if target in ['samples', 'both']:
                # Sample image files (sample_*.jpg, sample_*.png, etc)
                sample_files = fp.scan_files(split_path / 'images', 'sample_')
                files_to_remove.extend(sample_files)
            
            # Calculate stats
            split_size = sum(fp.get_file_info(f).get('size_mb', 0) for f in files_to_remove)
            
            preview['by_split'][split] = {
                'files_count': len(files_to_remove),
                'size_mb': round(split_size, 2),
                'file_list': [str(f) for f in files_to_remove[:10]]  # First 10 for preview
            }
            
            preview['total_files'] += len(files_to_remove)
            preview['total_size_mb'] += split_size
        
        preview['total_size_mb'] = round(preview['total_size_mb'], 2)
        
        return preview
        
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"❌ Cleanup preview error: {str(e)}")
        return {
            'success': False,
            'message': f"❌ Error: {str(e)}"
        }

def cleanup_empty_directories(data_dir: Union[str, Path]) -> Dict[str, Any]:
    """📁 Cleanup empty directories yang tertinggal setelah cleanup
    
    Args:
        data_dir: Base data directory
        
    Returns:
        Dict dengan cleanup results
    """
    try:
        data_path = Path(data_dir)
        removed_dirs = []
        
        # Check untuk empty directories
        for split_dir in data_path.iterdir():
            if split_dir.is_dir():
                images_dir = split_dir / 'images'
                labels_dir = split_dir / 'labels'
                
                # Check jika images directory kosong
                if images_dir.exists() and not any(images_dir.iterdir()):
                    images_dir.rmdir()
                    removed_dirs.append(str(images_dir))
                
                # Check jika labels directory kosong
                if labels_dir.exists() and not any(labels_dir.iterdir()):
                    labels_dir.rmdir()
                    removed_dirs.append(str(labels_dir))
                
                # Check jika split directory kosong
                if split_dir.exists() and not any(split_dir.iterdir()):
                    split_dir.rmdir()
                    removed_dirs.append(str(split_dir))
        
        return {
            'success': True,
            'message': f"✅ Removed {len(removed_dirs)} empty directories",
            'removed_directories': removed_dirs
        }
        
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"❌ Empty directory cleanup error: {str(e)}")
        return {
            'success': False,
            'message': f"❌ Error: {str(e)}"
        }

def _cleanup_split_files(split_path: Path, target: str) -> Dict[str, Any]:
    """🗑️ Internal cleanup untuk single split"""
    from ..core.file_processor import FileProcessor
    
    fp = FileProcessor()
    files_removed = 0
    size_removed = 0
    
    # Get target files
    files_to_remove = []
    
    if target in ['preprocessed', 'both']:
        # Preprocessing .npy files
        npy_files = fp.scan_files(split_path / 'images', 'pre_', {'.npy'})
        files_to_remove.extend(npy_files)
        
        # Corresponding metadata files
        for npy_file in npy_files:
            meta_file = npy_file.with_suffix('.meta.json')
            if meta_file.exists():
                files_to_remove.append(meta_file)
    
    if target in ['samples', 'both']:
        # Sample files
        sample_files = fp.scan_files(split_path / 'images', 'sample_')
        files_to_remove.extend(sample_files)
    
    # Remove files
    for file_path in files_to_remove:
        try:
            if file_path.exists():
                file_size = fp.get_file_info(file_path).get('size_mb', 0)
                file_path.unlink()
                files_removed += 1
                size_removed += file_size
        except Exception as e:
            logger = get_logger(__name__)
            logger.warning(f"⚠️ Failed to remove {file_path}: {str(e)}")
    
    return {
        'files_removed': files_removed,
        'size_removed_mb': round(size_removed, 2),
        'target': target
    }