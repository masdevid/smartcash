"""
File: smartcash/dataset/preprocessor/utils/preprocessing_paths.py
Deskripsi: Path resolution dan management untuk preprocessing operations
"""

from pathlib import Path
from typing import Dict, Any, Optional

from smartcash.common.logger import get_logger
from smartcash.dataset.utils.path_validator import get_path_validator


class PreprocessingPaths:
    """Path manager untuk preprocessing operations dengan intelligent resolution."""
    
    def __init__(self, config: Dict[str, Any], logger=None):
        """Initialize path manager dengan configuration."""
        self.config = config
        self.logger = logger or get_logger()
        self.path_validator = get_path_validator(logger)
        
        # Base paths dari config
        self.data_dir = Path(config.get('data', {}).get('dir', 'data'))
        self.preprocessed_dir = Path(config.get('preprocessing', {}).get('output_dir', 'data/preprocessed'))
    
    def resolve_source_paths(self, split: str) -> Dict[str, Any]:
        """
        Resolve source paths untuk split dengan comprehensive validation.
        
        Args:
            split: Split name untuk resolve
            
        Returns:
            Dictionary dengan source path information
        """
        try:
            # Get split path dengan val->valid mapping
            split_path = self.path_validator.get_split_path(str(self.data_dir), split)
            
            if not split_path.exists():
                return {
                    'valid': False,
                    'message': f'Split directory tidak ditemukan: {split_path}',
                    'split_path': str(split_path)
                }
            
            # Validate images dan labels directories
            images_dir = split_path / 'images'
            labels_dir = split_path / 'labels'
            
            validation_issues = []
            if not images_dir.exists():
                validation_issues.append(f'Images directory tidak ditemukan: {images_dir}')
            if not labels_dir.exists():
                validation_issues.append(f'Labels directory tidak ditemukan: {labels_dir}')
            
            if validation_issues:
                return {
                    'valid': False,
                    'message': '; '.join(validation_issues),
                    'split_path': str(split_path),
                    'images_dir': str(images_dir),
                    'labels_dir': str(labels_dir)
                }
            
            return {
                'valid': True,
                'message': f'Source paths valid untuk split {split}',
                'split_path': str(split_path),
                'images_dir': str(images_dir),
                'labels_dir': str(labels_dir),
                'split': split
            }
            
        except Exception as e:
            return {
                'valid': False,
                'message': f'Error resolving source paths: {str(e)}',
                'split': split
            }
    
    def setup_target_paths(self, split: str) -> Dict[str, Any]:
        """
        Setup target paths untuk preprocessing output dengan directory creation.
        
        Args:
            split: Split name untuk setup
            
        Returns:
            Dictionary dengan target path information
        """
        try:
            # Target split directory
            target_split_dir = self.preprocessed_dir / split
            target_images_dir = target_split_dir / 'images'
            target_labels_dir = target_split_dir / 'labels'
            
            # Create directories
            target_images_dir.mkdir(parents=True, exist_ok=True)
            target_labels_dir.mkdir(parents=True, exist_ok=True)
            
            # Validate creation success
            if not (target_images_dir.exists() and target_labels_dir.exists()):
                return {
                    'success': False,
                    'message': 'Failed to create target directories',
                    'target_split_dir': str(target_split_dir)
                }
            
            return {
                'success': True,
                'message': f'Target paths setup untuk split {split}',
                'target_split_dir': target_split_dir,
                'images_dir': target_images_dir,
                'labels_dir': target_labels_dir,
                'split': split
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f'Error setting up target paths: {str(e)}',
                'split': split
            }
    
    def validate_path_accessibility(self, paths: Dict[str, str]) -> Dict[str, Any]:
        """
        Validate accessibility untuk multiple paths.
        
        Args:
            paths: Dictionary path_name -> path_string
            
        Returns:
            Dictionary validation result
        """
        validation_result = {'all_accessible': True, 'path_status': {}, 'issues': []}
        
        for path_name, path_str in paths.items():
            path_obj = Path(path_str)
            
            path_status = {
                'exists': path_obj.exists(),
                'readable': False,
                'writable': False,
                'is_directory': False
            }
            
            if path_obj.exists():
                try:
                    path_status['readable'] = os.access(path_obj, os.R_OK)
                    path_status['writable'] = os.access(path_obj, os.W_OK)
                    path_status['is_directory'] = path_obj.is_dir()
                except Exception:
                    path_status['accessible'] = False
            
            validation_result['path_status'][path_name] = path_status
            
            # Check critical issues
            if not path_status['exists']:
                validation_result['issues'].append(f'{path_name}: Path tidak exists')
                validation_result['all_accessible'] = False
            elif not path_status['readable']:
                validation_result['issues'].append(f'{path_name}: Path tidak readable')
                validation_result['all_accessible'] = False
        
        return validation_result
    
    def get_preprocessing_paths(self) -> Dict[str, str]:
        """Get all preprocessing-related paths."""
        return {
            'data_root': str(self.data_dir),
            'preprocessed_root': str(self.preprocessed_dir),
            'train_source': str(self.data_dir / 'train'),
            'valid_source': str(self.data_dir / 'valid'),
            'test_source': str(self.data_dir / 'test'),
            'train_target': str(self.preprocessed_dir / 'train'),
            'valid_target': str(self.preprocessed_dir / 'valid'),
            'test_target': str(self.preprocessed_dir / 'test'),
            'metadata': str(self.preprocessed_dir / 'metadata')
        }
    
    def check_storage_requirements(self, estimated_files: int, avg_file_size_mb: float = 0.25) -> Dict[str, Any]:
        """
        Check storage requirements untuk preprocessing operation.
        
        Args:
            estimated_files: Estimated jumlah files yang akan diproses
            avg_file_size_mb: Average ukuran file preprocessed dalam MB
            
        Returns:
            Dictionary storage analysis
        """
        try:
            # Calculate storage requirements
            estimated_size_mb = estimated_files * avg_file_size_mb
            estimated_size_gb = estimated_size_mb / 1024
            
            # Check available space di target directory
            target_stats = self.preprocessed_dir.stat() if self.preprocessed_dir.exists() else None
            
            storage_info = {
                'estimated_files': estimated_files,
                'estimated_size_mb': round(estimated_size_mb, 2),
                'estimated_size_gb': round(estimated_size_gb, 2),
                'target_directory': str(self.preprocessed_dir),
                'sufficient_space': True,  # Default assumption
                'recommendations': []
            }
            
            # Add recommendations berdasarkan size
            if estimated_size_gb > 5:
                storage_info['recommendations'].append(
                    f"📦 Large operation: {estimated_size_gb:.1f}GB estimated. Ensure sufficient disk space."
                )
            
            if estimated_files > 10000:
                storage_info['recommendations'].append(
                    f"📁 Many files: {estimated_files:,} files akan dibuat. Pertimbangkan batch processing."
                )
            
            return storage_info
            
        except Exception as e:
            return {
                'estimated_files': estimated_files,
                'error': f'Storage analysis error: {str(e)}'
            }
    
    def cleanup_empty_directories(self, base_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Cleanup empty directories dari preprocessing paths.
        
        Args:
            base_path: Base path untuk cleanup (default: preprocessed_dir)
            
        Returns:
            Dictionary cleanup result
        """
        if base_path is None:
            base_path = self.preprocessed_dir
        
        cleanup_stats = {'directories_removed': 0, 'directories_checked': 0}
        
        try:
            if not base_path.exists():
                return cleanup_stats
            
            # Walk through directories bottom-up
            for dir_path in sorted(base_path.rglob('*'), key=lambda p: len(p.parts), reverse=True):
                if dir_path.is_dir():
                    cleanup_stats['directories_checked'] += 1
                    
                    try:
                        # Check if directory is empty
                        if not any(dir_path.iterdir()):
                            dir_path.rmdir()
                            cleanup_stats['directories_removed'] += 1
                            self.logger.debug(f"🗂️ Removed empty directory: {dir_path}")
                    except OSError:
                        # Directory not empty or permission issue
                        pass
            
            return cleanup_stats
            
        except Exception as e:
            self.logger.warning(f"⚠️ Cleanup error: {str(e)}")
            return cleanup_stats