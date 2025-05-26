"""
File: smartcash/dataset/augmentor/processors/file.py
Deskripsi: File operations dengan one-liner optimized untuk dataset management dan file handling
"""

import os
import shutil
import glob
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

from smartcash.common.logger import get_logger

# One-liner helper functions
ensure_dir = lambda path: Path(path).mkdir(parents=True, exist_ok=True)
copy_file = lambda src, dst: shutil.copy2(src, dst) if Path(src).exists() else False
move_file = lambda src, dst: shutil.move(src, dst) if Path(src).exists() else False
delete_file = lambda path: Path(path).unlink() if Path(path).exists() else False
file_exists = lambda path: Path(path).exists()
get_file_size = lambda path: Path(path).stat().st_size if Path(path).exists() else 0
get_file_stem = lambda path: Path(path).stem
get_file_parent = lambda path: Path(path).parent

def find_files(directory: str, patterns: Union[str, List[str]] = "*", recursive: bool = True) -> List[str]:
    """Find files dengan pattern matching one-liner."""
    patterns = [patterns] if isinstance(patterns, str) else patterns
    search_func = Path(directory).rglob if recursive else Path(directory).glob
    return [str(f) for pattern in patterns for f in search_func(pattern) if f.is_file()]

class FileProcessor:
    """Processor untuk operasi file dengan optimized batch operations."""
    
    def __init__(self, logger=None):
        """
        Inisialisasi FileProcessor.
        
        Args:
            logger: Logger untuk logging operations
        """
        self.logger = logger or get_logger(__name__)
        self.copied_count = 0
        self.moved_count = 0
        self.deleted_count = 0
    
    def batch_copy(self, file_mappings: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Batch copy files dengan progress tracking.
        
        Args:
            file_mappings: List dict dengan 'src' dan 'dst' keys
            
        Returns:
            Dictionary hasil operasi
        """
        results = {'success': [], 'failed': [], 'skipped': []}
        
        for mapping in file_mappings:
            src, dst = mapping.get('src'), mapping.get('dst')
            
            if not src or not dst:
                results['failed'].append({'mapping': mapping, 'error': 'Invalid mapping'})
                continue
            
            try:
                # Ensure destination directory
                ensure_dir(get_file_parent(dst))
                
                # Copy dengan validation
                if Path(src).exists():
                    if copy_file(src, dst):
                        results['success'].append({'src': src, 'dst': dst})
                        self.copied_count += 1
                    else:
                        results['failed'].append({'src': src, 'dst': dst, 'error': 'Copy failed'})
                else:
                    results['skipped'].append({'src': src, 'error': 'Source not found'})
                    
            except Exception as e:
                results['failed'].append({'src': src, 'dst': dst, 'error': str(e)})
        
        # Log summary
        self.logger.info(f"📁 Batch copy: {len(results['success'])} berhasil, {len(results['failed'])} gagal")
        
        return results
    
    def batch_move(self, file_mappings: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Batch move files dengan progress tracking.
        
        Args:
            file_mappings: List dict dengan 'src' dan 'dst' keys
            
        Returns:
            Dictionary hasil operasi
        """
        results = {'success': [], 'failed': [], 'skipped': []}
        
        for mapping in file_mappings:
            src, dst = mapping.get('src'), mapping.get('dst')
            
            if not src or not dst:
                results['failed'].append({'mapping': mapping, 'error': 'Invalid mapping'})
                continue
            
            try:
                # Ensure destination directory
                ensure_dir(get_file_parent(dst))
                
                # Move dengan validation
                if Path(src).exists():
                    if move_file(src, dst):
                        results['success'].append({'src': src, 'dst': dst})
                        self.moved_count += 1
                    else:
                        results['failed'].append({'src': src, 'dst': dst, 'error': 'Move failed'})
                else:
                    results['skipped'].append({'src': src, 'error': 'Source not found'})
                    
            except Exception as e:
                results['failed'].append({'src': src, 'dst': dst, 'error': str(e)})
        
        # Log summary
        self.logger.info(f"📁 Batch move: {len(results['success'])} berhasil, {len(results['failed'])} gagal")
        
        return results
    
    def cleanup_files(self, file_patterns: List[str], directories: List[str]) -> Dict[str, Any]:
        """
        Cleanup files berdasarkan pattern di multiple directories.
        
        Args:
            file_patterns: List pattern file (e.g., ['aug_*', '*.tmp'])
            directories: List direktori untuk dicari
            
        Returns:
            Dictionary hasil cleanup
        """
        deleted_files = []
        failed_files = []
        
        for directory in directories:
            if not Path(directory).exists():
                continue
                
            # Find files dengan patterns
            for pattern in file_patterns:
                try:
                    files_to_delete = find_files(directory, pattern, recursive=True)
                    
                    for file_path in files_to_delete:
                        try:
                            if delete_file(file_path):
                                deleted_files.append(file_path)
                                self.deleted_count += 1
                            else:
                                failed_files.append({'file': file_path, 'error': 'Delete failed'})
                        except Exception as e:
                            failed_files.append({'file': file_path, 'error': str(e)})
                            
                except Exception as e:
                    self.logger.error(f"❌ Error cleanup pattern {pattern} di {directory}: {str(e)}")
        
        # Log summary
        self.logger.info(f"🧹 Cleanup: {len(deleted_files)} file dihapus, {len(failed_files)} gagal")
        
        return {
            'deleted_files': deleted_files,
            'failed_files': failed_files,
            'deleted_count': len(deleted_files),
            'failed_count': len(failed_files)
        }
    
    def find_image_label_pairs(self, images_dir: str, labels_dir: str) -> List[Dict[str, str]]:
        """
        Find pairs image dan label files.
        
        Args:
            images_dir: Direktori gambar
            labels_dir: Direktori label
            
        Returns:
            List pairs dengan 'image' dan 'label' keys
        """
        # Find image files
        image_patterns = ['*.jpg', '*.jpeg', '*.png']
        image_files = find_files(images_dir, image_patterns, recursive=False)
        
        pairs = []
        for image_file in image_files:
            image_stem = get_file_stem(image_file)
            label_file = str(Path(labels_dir) / f"{image_stem}.txt")
            
            if file_exists(label_file):
                pairs.append({
                    'image': image_file,
                    'label': label_file,
                    'stem': image_stem
                })
        
        return pairs
    
    def organize_dataset_structure(
        self, 
        source_dir: str, 
        target_dir: str, 
        splits: List[str] = ['train', 'valid', 'test']
    ) -> Dict[str, Any]:
        """
        Organize dataset ke struktur YOLO standard.
        
        Args:
            source_dir: Source directory
            target_dir: Target directory
            splits: List splits yang akan dibuat
            
        Returns:
            Dictionary hasil organize
        """
        results = {'created_dirs': [], 'organized_files': 0, 'errors': []}
        
        try:
            # Create directory structure
            for split in splits:
                split_dirs = [
                    Path(target_dir) / split / 'images',
                    Path(target_dir) / split / 'labels'
                ]
                
                for dir_path in split_dirs:
                    ensure_dir(dir_path)
                    results['created_dirs'].append(str(dir_path))
            
            # Find and organize files
            pairs = self.find_image_label_pairs(
                str(Path(source_dir) / 'images'),
                str(Path(source_dir) / 'labels')
            )
            
            # Distribute files ke splits (simple round-robin)
            for i, pair in enumerate(pairs):
                split = splits[i % len(splits)]
                
                # Target paths
                target_image = Path(target_dir) / split / 'images' / Path(pair['image']).name
                target_label = Path(target_dir) / split / 'labels' / Path(pair['label']).name
                
                # Copy files
                try:
                    if copy_file(pair['image'], str(target_image)):
                        if copy_file(pair['label'], str(target_label)):
                            results['organized_files'] += 1
                        else:
                            results['errors'].append(f"Failed copy label: {pair['label']}")
                    else:
                        results['errors'].append(f"Failed copy image: {pair['image']}")
                except Exception as e:
                    results['errors'].append(f"Error organizing pair {pair['stem']}: {str(e)}")
            
            self.logger.info(f"📊 Dataset organized: {results['organized_files']} files ke {len(splits)} splits")
            
        except Exception as e:
            results['errors'].append(f"Fatal error: {str(e)}")
            self.logger.error(f"❌ Error organize dataset: {str(e)}")
        
        return results
    
    def create_symlinks(
        self, 
        source_files: List[str], 
        target_dir: str, 
        prefix: str = ""
    ) -> Dict[str, Any]:
        """
        Create symlinks untuk files dengan optional prefix.
        
        Args:
            source_files: List source files
            target_dir: Target directory untuk symlinks
            prefix: Optional prefix untuk symlink names
            
        Returns:
            Dictionary hasil create symlinks
        """
        results = {'success': [], 'failed': [], 'skipped': []}
        
        # Ensure target directory
        ensure_dir(target_dir)
        
        for source_file in source_files:
            try:
                source_path = Path(source_file)
                if not source_path.exists():
                    results['skipped'].append({'file': source_file, 'error': 'Source not found'})
                    continue
                
                # Generate symlink name dengan prefix
                symlink_name = f"{prefix}{source_path.name}" if prefix else source_path.name
                symlink_path = Path(target_dir) / symlink_name
                
                # Create symlink jika belum ada
                if not symlink_path.exists():
                    symlink_path.symlink_to(source_path.resolve())
                    results['success'].append({'source': source_file, 'symlink': str(symlink_path)})
                else:
                    results['skipped'].append({'file': source_file, 'error': 'Symlink already exists'})
                    
            except Exception as e:
                results['failed'].append({'file': source_file, 'error': str(e)})
        
        # Log summary
        self.logger.info(f"🔗 Symlinks: {len(results['success'])} berhasil, {len(results['failed'])} gagal")
        
        return results
    
    def validate_dataset_structure(self, dataset_dir: str) -> Dict[str, Any]:
        """
        Validasi struktur dataset YOLO.
        
        Args:
            dataset_dir: Root directory dataset
            
        Returns:
            Dictionary hasil validasi
        """
        validation = {
            'valid': True,
            'splits_found': [],
            'total_pairs': 0,
            'issues': []
        }
        
        expected_splits = ['train', 'valid', 'test']
        
        for split in expected_splits:
            split_path = Path(dataset_dir) / split
            images_path = split_path / 'images'
            labels_path = split_path / 'labels'
            
            if split_path.exists():
                validation['splits_found'].append(split)
                
                # Count pairs
                if images_path.exists() and labels_path.exists():
                    pairs = self.find_image_label_pairs(str(images_path), str(labels_path))
                    validation['total_pairs'] += len(pairs)
                    validation[f'{split}_pairs'] = len(pairs)
                else:
                    validation['issues'].append(f"Missing images/labels directories in {split}")
                    validation['valid'] = False
            else:
                validation['issues'].append(f"Missing {split} split directory")
        
        # Overall validation
        if not validation['splits_found']:
            validation['valid'] = False
            validation['issues'].append("No valid splits found")
        
        return validation
    
    def get_directory_stats(self, directory: str) -> Dict[str, Any]:
        """
        Dapatkan statistik directory.
        
        Args:
            directory: Path directory
            
        Returns:
            Dictionary statistik
        """
        stats = {
            'exists': False,
            'total_files': 0,
            'total_size': 0,
            'file_types': {},
            'subdirs': []
        }
        
        dir_path = Path(directory)
        if not dir_path.exists():
            return stats
        
        stats['exists'] = True
        
        try:
            # Count files dan calculate size
            for file_path in dir_path.rglob('*'):
                if file_path.is_file():
                    stats['total_files'] += 1
                    stats['total_size'] += get_file_size(str(file_path))
                    
                    # Count by extension
                    ext = file_path.suffix.lower()
                    stats['file_types'][ext] = stats['file_types'].get(ext, 0) + 1
                elif file_path.is_dir() and file_path != dir_path:
                    stats['subdirs'].append(str(file_path.relative_to(dir_path)))
            
            # Convert size to human readable
            stats['size_mb'] = round(stats['total_size'] / (1024 * 1024), 2)
            
        except Exception as e:
            stats['error'] = str(e)
        
        return stats
    
    def get_processing_stats(self) -> Dict[str, int]:
        """
        Dapatkan statistik processing.
        
        Returns:
            Dictionary statistik
        """
        return {
            'copied_files': self.copied_count,
            'moved_files': self.moved_count,
            'deleted_files': self.deleted_count,
            'total_operations': self.copied_count + self.moved_count + self.deleted_count
        }
    
    def reset_stats(self) -> None:
        """Reset counter statistik."""
        self.copied_count = 0
        self.moved_count = 0
        self.deleted_count = 0
    
    def __repr__(self) -> str:
        """String representation."""
        return f"FileProcessor(copied={self.copied_count}, moved={self.moved_count}, deleted={self.deleted_count})"