"""
File: smartcash/dataset/utils/path_validator.py
Deskripsi: Enhanced path validator dengan symlink-safe operations dan consistent val->valid mapping
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional
from smartcash.common.constants.paths import get_paths_for_environment
from smartcash.common.environment import get_environment_manager


class DatasetPathValidator:
    """Enhanced validator untuk path dataset dengan symlink-safe operations dan consistent mapping."""
    
    def __init__(self, logger=None):
        self.logger = logger
        self.env_manager = get_environment_manager()
        
        # Enhanced split mapping untuk konsistensi
        self.split_mapping = {
            'val': 'valid',
            'validation': 'valid',
            'validate': 'valid'
        }
        
        # Symlink patterns untuk detection
        self.symlink_patterns = {
            'augmentation': ['augment', 'synthetic', 'generated', 'aug_', 'synth_'],
            'backup': ['backup', 'bak_', 'original'],
            'cache': ['cache', 'temp', 'tmp_'],
            'external': ['external', 'ext_', 'linked']
        }
    
    def normalize_split_name(self, split: str) -> str:
        """Enhanced normalize split name untuk konsistensi."""
        normalized = self.split_mapping.get(split.lower(), split.lower())
        return normalized
    
    def get_dataset_paths(self, base_dir: Optional[str] = None) -> Dict[str, str]:
        """Get dataset paths berdasarkan environment dengan enhanced structure."""
        if base_dir:
            return {
                'data_root': base_dir,
                'train': f"{base_dir}/train",
                'valid': f"{base_dir}/valid", 
                'test': f"{base_dir}/test",
                'downloads': f"{base_dir}/downloads",
                'preprocessed': f"{base_dir}/preprocessed",
                'augmented': f"{base_dir}/augmented",
                'backup': f"{base_dir}/backup"
            }
        
        paths = get_paths_for_environment(
            self.env_manager.is_colab,
            self.env_manager.is_drive_mounted
        )
        
        # Enhanced paths dengan additional directories
        enhanced_paths = dict(paths)
        enhanced_paths.update({
            'preprocessed': f"{paths['data_root']}/preprocessed",
            'augmented': f"{paths['data_root']}/augmented",
            'backup': f"{paths['data_root']}/backup"
        })
        
        return enhanced_paths
    
    def detect_available_splits(self, data_dir: str) -> List[str]:
        """Enhanced deteksi splits dengan symlink-aware checking."""
        data_path = Path(data_dir)
        available_splits = []
        
        # Check standard splits dengan enhanced validation
        for split in ['train', 'valid', 'test']:
            split_dir = data_path / split
            if self._is_valid_split_dir(split_dir):
                available_splits.append(split)
        
        # Check legacy 'val' directory dengan enhanced mapping
        val_dir = data_path / 'val'
        if self._is_valid_split_dir(val_dir) and 'valid' not in available_splits:
            available_splits.append('valid')  # Always normalize ke 'valid'
        
        # Sort untuk consistency
        split_order = {'train': 0, 'valid': 1, 'test': 2}
        available_splits.sort(key=lambda x: split_order.get(x, 999))
        
        return available_splits
    
    def _is_valid_split_dir(self, split_dir: Path) -> bool:
        """Enhanced check apakah split directory valid dengan symlink consideration."""
        if not split_dir.exists():
            return False
        
        images_dir = split_dir / 'images'
        labels_dir = split_dir / 'labels'
        
        # Check basic structure
        if not (images_dir.exists() and labels_dir.exists()):
            return False
        
        # Enhanced file count dengan symlink-aware counting
        image_files = self._count_valid_files(images_dir, ['.jpg', '.jpeg', '.png', '.bmp'])
        label_files = self._count_valid_files(labels_dir, ['.txt'])
        
        return image_files > 0 and label_files > 0
    
    def _count_valid_files(self, directory: Path, extensions: List[str]) -> int:
        """Count valid files dengan symlink-aware handling."""
        if not directory.exists():
            return 0
        
        count = 0
        try:
            for file_path in directory.rglob('*'):
                if file_path.is_file() and file_path.suffix.lower() in extensions:
                    # Count both regular files dan valid symlinks
                    if file_path.is_symlink():
                        try:
                            # Check jika symlink valid (target exists)
                            resolved_path = file_path.resolve()
                            if resolved_path.exists():
                                count += 1
                        except Exception:
                            # Broken symlink, skip
                            continue
                    else:
                        count += 1
        except Exception as e:
            self.logger and self.logger.debug(f"🔍 File counting error in {directory}: {str(e)}")
        
        return count
    
    def get_split_path(self, data_dir: str, split: str) -> Path:
        """Enhanced get path untuk split dengan symlink-aware resolution."""
        data_path = Path(data_dir)
        normalized_split = self.normalize_split_name(split)
        
        # Check normalized path first
        split_path = data_path / normalized_split
        if split_path.exists():
            return split_path
        
        # Check legacy 'val' jika split adalah 'valid'
        if normalized_split == 'valid':
            val_path = data_path / 'val'
            if val_path.exists():
                return val_path
        
        return split_path
    
    def validate_dataset_structure(self, data_dir: str) -> Dict[str, any]:
        """Enhanced validate struktur dataset dengan symlink analysis."""
        data_path = Path(data_dir)
        
        result = {
            'valid': data_path.exists(),
            'data_dir': str(data_path),
            'splits': {},
            'issues': [],
            'total_images': 0,
            'total_labels': 0,
            'symlink_analysis': {}
        }
        
        if not result['valid']:
            result['issues'].append(f"❌ Dataset directory tidak ditemukan: {data_dir}")
            return result
        
        # Enhanced check each split dengan symlink analysis
        available_splits = self.detect_available_splits(data_dir)
        
        for split in ['train', 'valid', 'test']:
            split_path = self.get_split_path(data_dir, split)
            images_dir = split_path / 'images'
            labels_dir = split_path / 'labels'
            
            if split in available_splits:
                # Enhanced counting dengan symlink details
                image_count, image_symlinks = self._count_files_with_symlinks(images_dir, ['.jpg', '.jpeg', '.png', '.bmp'])
                label_count, label_symlinks = self._count_files_with_symlinks(labels_dir, ['.txt'])
                
                result['splits'][split] = {
                    'exists': True,
                    'path': str(split_path),
                    'images': image_count,
                    'labels': label_count,
                    'images_dir': str(images_dir),
                    'labels_dir': str(labels_dir),
                    'symlinks': {
                        'images': image_symlinks,
                        'labels': label_symlinks
                    }
                }
                
                result['total_images'] += image_count
                result['total_labels'] += label_count
                
                # Enhanced issue detection
                if image_count == 0:
                    result['issues'].append(f"⚠️ Split {split}: Tidak ada gambar")
                if label_count == 0:
                    result['issues'].append(f"⚠️ Split {split}: Tidak ada label")
                if image_count != label_count:
                    result['issues'].append(f"⚠️ Split {split}: Gambar ({image_count}) ≠ Label ({label_count})")
                
                # Symlink-specific issues
                if image_symlinks['broken'] > 0 or label_symlinks['broken'] > 0:
                    total_broken = image_symlinks['broken'] + label_symlinks['broken']
                    result['issues'].append(f"🔗 Split {split}: {total_broken} broken symlinks")
            else:
                result['splits'][split] = {
                    'exists': False,
                    'path': str(split_path),
                    'images': 0,
                    'labels': 0,
                    'symlinks': {'images': {'total': 0, 'broken': 0, 'types': {}}, 'labels': {'total': 0, 'broken': 0, 'types': {}}}
                }
                result['issues'].append(f"❌ Split {split}: Directory tidak ditemukan")
        
        # Overall symlink analysis
        result['symlink_analysis'] = self._analyze_overall_symlinks(result['splits'])
        
        return result
    
    def _count_files_with_symlinks(self, directory: Path, extensions: List[str]) -> Tuple[int, Dict[str, any]]:
        """Count files dengan detailed symlink analysis."""
        symlink_info = {
            'total': 0,
            'broken': 0,
            'types': {},
            'targets': []
        }
        
        if not directory.exists():
            return 0, symlink_info
        
        total_files = 0
        
        try:
            for file_path in directory.rglob('*'):
                if file_path.is_file() and file_path.suffix.lower() in extensions:
                    if file_path.is_symlink():
                        symlink_info['total'] += 1
                        
                        try:
                            target = file_path.resolve()
                            if target.exists():
                                total_files += 1
                                
                                # Categorize symlink type
                                symlink_type = self._categorize_symlink(str(target))
                                symlink_info['types'][symlink_type] = symlink_info['types'].get(symlink_type, 0) + 1
                                symlink_info['targets'].append({
                                    'source': str(file_path),
                                    'target': str(target),
                                    'type': symlink_type
                                })
                            else:
                                symlink_info['broken'] += 1
                        except Exception:
                            symlink_info['broken'] += 1
                    else:
                        total_files += 1
        except Exception as e:
            self.logger and self.logger.debug(f"🔍 Symlink analysis error in {directory}: {str(e)}")
        
        return total_files, symlink_info
    
    def _categorize_symlink(self, target_path: str) -> str:
        """Categorize symlink berdasarkan target path patterns."""
        target_lower = target_path.lower()
        
        for category, patterns in self.symlink_patterns.items():
            if any(pattern in target_lower for pattern in patterns):
                return category
        
        return 'other'
    
    def _analyze_overall_symlinks(self, splits: Dict[str, Dict[str, any]]) -> Dict[str, any]:
        """Analyze overall symlinks across all splits."""
        analysis = {
            'total_symlinks': 0,
            'total_broken': 0,
            'by_type': {},
            'by_split': {},
            'recommendations': []
        }
        
        for split_name, split_data in splits.items():
            if split_data['exists']:
                split_symlinks = 0
                split_broken = 0
                
                for file_type in ['images', 'labels']:
                    symlink_data = split_data['symlinks'][file_type]
                    split_symlinks += symlink_data['total']
                    split_broken += symlink_data['broken']
                    
                    # Aggregate by type
                    for symlink_type, count in symlink_data['types'].items():
                        analysis['by_type'][symlink_type] = analysis['by_type'].get(symlink_type, 0) + count
                
                analysis['by_split'][split_name] = {
                    'total': split_symlinks,
                    'broken': split_broken
                }
                
                analysis['total_symlinks'] += split_symlinks
                analysis['total_broken'] += split_broken
        
        # Generate recommendations
        if analysis['total_broken'] > 0:
            analysis['recommendations'].append(f"🔗 Perbaiki {analysis['total_broken']} broken symlinks")
        
        if analysis['by_type'].get('augmentation', 0) > 0:
            aug_count = analysis['by_type']['augmentation']
            analysis['recommendations'].append(f"📎 {aug_count} augmentation symlinks terdeteksi - akan dipertahankan saat cleanup")
        
        if analysis['total_symlinks'] > analysis['total_broken']:
            valid_symlinks = analysis['total_symlinks'] - analysis['total_broken']
            analysis['recommendations'].append(f"✅ {valid_symlinks} valid symlinks akan dikelola dengan aman")
        
        return analysis
    
    def get_preprocessed_paths(self, base_dir: Optional[str] = None) -> Dict[str, str]:
        """Get preprocessed dataset paths dengan enhanced structure."""
        paths = self.get_dataset_paths(base_dir)
        
        return {
            'preprocessed_root': paths['preprocessed'],
            'train': f"{paths['preprocessed']}/train",
            'valid': f"{paths['preprocessed']}/valid",
            'test': f"{paths['preprocessed']}/test",
            'metadata': f"{paths['preprocessed']}/metadata"
        }
    
    def validate_preprocessed_structure(self, preprocessed_dir: str) -> Dict[str, any]:
        """Enhanced validate struktur preprocessed dataset dengan symlink support."""
        preprocessed_path = Path(preprocessed_dir)
        
        result = {
            'valid': preprocessed_path.exists(),
            'preprocessed_dir': str(preprocessed_path),
            'splits': {},
            'total_processed': 0,
            'symlink_analysis': {}
        }
        
        if not result['valid']:
            return result
        
        # Enhanced check preprocessed splits
        for split in ['train', 'valid', 'test']:
            split_path = preprocessed_path / split
            
            if split_path.exists():
                # Count dengan symlink awareness
                processed_count, symlink_info = self._count_preprocessed_files(split_path)
                
                result['splits'][split] = {
                    'exists': True,
                    'path': str(split_path),
                    'processed': processed_count,
                    'symlinks': symlink_info
                }
                result['total_processed'] += processed_count
            else:
                result['splits'][split] = {
                    'exists': False,
                    'path': str(split_path),
                    'processed': 0,
                    'symlinks': {'total': 0, 'broken': 0, 'types': {}}
                }
        
        # Overall symlink analysis untuk preprocessed
        result['symlink_analysis'] = self._analyze_overall_symlinks(result['splits'])
        
        return result
    
    def _count_preprocessed_files(self, split_path: Path) -> Tuple[int, Dict[str, any]]:
        """Count preprocessed files dengan symlink analysis."""
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.txt']
        return self._count_files_with_symlinks(split_path, extensions)
    
    def is_symlink_safe_operation(self, operation_path: str, operation_type: str = 'cleanup') -> Dict[str, any]:
        """Check apakah operation aman untuk symlinks."""
        path = Path(operation_path)
        safety_info = {
            'safe': True,
            'warnings': [],
            'symlink_count': 0,
            'augmentation_symlinks': 0,
            'recommendations': []
        }
        
        if not path.exists():
            safety_info['warnings'].append(f"Path tidak exists: {operation_path}")
            return safety_info
        
        try:
            # Count symlinks
            for item in path.rglob('*'):
                if item.is_symlink():
                    safety_info['symlink_count'] += 1
                    
                    try:
                        target = item.resolve()
                        target_str = str(target).lower()
                        
                        if self._categorize_symlink(target_str) == 'augmentation':
                            safety_info['augmentation_symlinks'] += 1
                    except Exception:
                        pass  # Broken symlink
            
            # Generate safety recommendations
            if safety_info['symlink_count'] > 0:
                if operation_type == 'cleanup':
                    safety_info['recommendations'].append(f"🔗 {safety_info['symlink_count']} symlinks akan dihapus")
                    
                    if safety_info['augmentation_symlinks'] > 0:
                        safety_info['recommendations'].append(f"📎 {safety_info['augmentation_symlinks']} augmentation symlinks - data original tetap aman")
                
                elif operation_type == 'move':
                    safety_info['warnings'].append("Symlinks mungkin broken setelah move operation")
                    safety_info['recommendations'].append("Consider copying instead of moving untuk preserve symlinks")
                
                elif operation_type == 'backup':
                    safety_info['recommendations'].append("Use rsync atau cp -L untuk resolve symlinks saat backup")
        
        except Exception as e:
            safety_info['safe'] = False
            safety_info['warnings'].append(f"Error analyzing symlinks: {str(e)}")
        
        return safety_info


# Singleton instance
_path_validator = None

def get_path_validator(logger=None) -> DatasetPathValidator:
    """Get singleton path validator dengan enhanced symlink support."""
    global _path_validator
    if _path_validator is None:
        _path_validator = DatasetPathValidator(logger)
    return _path_validator

def validate_symlink_safety(operation_path: str, operation_type: str = 'cleanup', logger=None) -> Dict[str, any]:
    """
    Utility function untuk validate symlink safety sebelum operations.
    
    Args:
        operation_path: Path yang akan dioperasi
        operation_type: Tipe operasi ('cleanup', 'move', 'backup', etc.)
        logger: Logger instance
        
    Returns:
        Dictionary berisi safety analysis
    """
    validator = get_path_validator(logger)
    return validator.is_symlink_safe_operation(operation_path, operation_type)

def get_symlink_summary(data_dir: str, logger=None) -> Dict[str, any]:
    """
    Get comprehensive symlink summary untuk sebuah directory.
    
    Args:
        data_dir: Directory untuk dianalysis
        logger: Logger instance
        
    Returns:
        Dictionary berisi symlink summary
    """
    validator = get_path_validator(logger)
    validation_result = validator.validate_dataset_structure(data_dir)
    
    return {
        'total_symlinks': validation_result.get('symlink_analysis', {}).get('total_symlinks', 0),
        'broken_symlinks': validation_result.get('symlink_analysis', {}).get('total_broken', 0),
        'by_type': validation_result.get('symlink_analysis', {}).get('by_type', {}),
        'by_split': validation_result.get('symlink_analysis', {}).get('by_split', {}),
        'recommendations': validation_result.get('symlink_analysis', {}).get('recommendations', [])
    }

def repair_broken_symlinks(data_dir: str, logger=None) -> Dict[str, any]:
    """
    Utility function untuk repair broken symlinks.
    
    Args:
        data_dir: Directory untuk repair symlinks
        logger: Logger instance
        
    Returns:
        Dictionary berisi repair results
    """
    validator = get_path_validator(logger)
    data_path = Path(data_dir)
    
    repair_results = {
        'total_checked': 0,
        'broken_found': 0,
        'repaired': 0,
        'failed_repairs': 0,
        'removed_broken': 0,
        'details': []
    }
    
    if not data_path.exists():
        repair_results['details'].append(f"❌ Directory tidak exists: {data_dir}")
        return repair_results
    
    try:
        for item in data_path.rglob('*'):
            if item.is_symlink():
                repair_results['total_checked'] += 1
                
                try:
                    # Check if symlink is broken
                    target = item.resolve()
                    if not target.exists():
                        repair_results['broken_found'] += 1
                        
                        # Try to repair or remove
                        if logger:
                            logger.debug(f"🔗 Broken symlink found: {item} -> {target}")
                        
                        # For now, just remove broken symlinks
                        item.unlink()
                        repair_results['removed_broken'] += 1
                        repair_results['details'].append(f"🗑️ Removed broken symlink: {item.name}")
                        
                except Exception as e:
                    repair_results['failed_repairs'] += 1
                    repair_results['details'].append(f"❌ Failed to process {item}: {str(e)}")
    
    except Exception as e:
        repair_results['details'].append(f"❌ Error during repair: {str(e)}")
    
    if logger and repair_results['broken_found'] > 0:
        logger.info(f"🔧 Symlink repair completed: {repair_results['removed_broken']} broken links removed")
    
    return repair_results