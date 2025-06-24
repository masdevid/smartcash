"""
File: smartcash/dataset/preprocessor/utils/path_manager.py
Deskripsi: Konsolidasi path management dengan auto-creation dan validation
"""

import os
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass

from smartcash.common.logger import get_logger

@dataclass
class PathConfig:
    """📁 Path configuration structure"""
    source_dir: Path
    output_dir: Path
    images_subdir: str = 'images'
    labels_subdir: str = 'labels'
    create_missing: bool = True
    
    def __post_init__(self):
        self.source_dir = Path(self.source_dir)
        self.output_dir = Path(self.output_dir)

class PathManager:
    """📁 Konsolidasi path management dengan auto-creation dan validation"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = get_logger(__name__)
        
        # Base directories dari config
        self.data_root = Path(self.config.get('data', {}).get('dir', 'data'))
        self.preprocessing_config = self.config.get('preprocessing', {})
        self.output_root = Path(self.preprocessing_config.get('output_dir', 'data/preprocessed'))
        
        # Split configuration
        self.splits = ['train', 'valid', 'test']
        self.local_paths = self.config.get('data', {}).get('local', {})
        
        # Auto-creation setting
        self.auto_create = self.config.get('auto_create_dirs', True)
    
    # === SOURCE PATH RESOLUTION ===
    
    def get_source_split_dir(self, split: str) -> Path:
        """📂 Get source directory untuk split dengan fallback logic"""
        # Priority 1: Explicit local path dari config
        if split in self.local_paths:
            return Path(self.local_paths[split])
        
        # Priority 2: Standard structure
        return self.data_root / split
    
    def get_source_images_dir(self, split: str) -> Path:
        """🖼️ Get source images directory untuk split"""
        return self.get_source_split_dir(split) / 'images'
    
    def get_source_labels_dir(self, split: str) -> Path:
        """📋 Get source labels directory untuk split"""
        return self.get_source_split_dir(split) / 'labels'
    
    def get_source_paths(self, split: str) -> Tuple[Path, Path]:
        """📁 Get both source images dan labels directories"""
        split_dir = self.get_source_split_dir(split)
        return split_dir / 'images', split_dir / 'labels'
    
    # === OUTPUT PATH RESOLUTION ===
    
    def get_output_split_dir(self, split: str) -> Path:
        """📤 Get output directory untuk split"""
        organize_by_split = self.preprocessing_config.get('output', {}).get('organize_by_split', True)
        
        if organize_by_split:
            return self.output_root / split
        return self.output_root
    
    def get_output_images_dir(self, split: str) -> Path:
        """🖼️ Get output images directory untuk split"""
        return self.get_output_split_dir(split) / 'images'
    
    def get_output_labels_dir(self, split: str) -> Path:
        """📋 Get output labels directory untuk split"""
        return self.get_output_split_dir(split) / 'labels'
    
    def get_output_paths(self, split: str) -> Tuple[Path, Path]:
        """📁 Get both output images dan labels directories"""
        split_dir = self.get_output_split_dir(split)
        return split_dir / 'images', split_dir / 'labels'
    
    # === SPECIALIZED DIRECTORIES ===
    
    def get_backup_dir(self, split: Optional[str] = None) -> Path:
        """🗄️ Get backup directory dengan optional split"""
        backup_base = Path(self.preprocessing_config.get('backup_dir', 'data/backup'))
        return backup_base / split if split else backup_base
    
    def get_invalid_dir(self, split: Optional[str] = None) -> Path:
        """❌ Get invalid files directory"""
        invalid_base = Path(self.preprocessing_config.get('validation', {}).get('invalid_dir', 'data/invalid'))
        return invalid_base / split if split else invalid_base
    
    def get_temp_dir(self, operation: str = 'preprocessing') -> Path:
        """🔄 Get temporary directory untuk operation"""
        return self.output_root / '.temp' / operation
    
    def get_visualization_dir(self, split: Optional[str] = None) -> Path:
        """📊 Get visualization output directory"""
        vis_base = Path(self.preprocessing_config.get('vis_dir', 'visualizations/preprocessing'))
        return vis_base / split if split else vis_base
    
    # === PATH VALIDATION & CREATION ===
    
    def validate_source_structure(self, splits: Optional[List[str]] = None) -> Dict[str, Any]:
        """✅ Validate source directory structure"""
        splits = splits or self.splits
        validation_result = {
            'is_valid': True,
            'splits': {},
            'missing_dirs': [],
            'empty_dirs': [],
            'total_images': 0
        }
        
        for split in splits:
            img_dir, label_dir = self.get_source_paths(split)
            
            split_info = {
                'images_dir_exists': img_dir.exists(),
                'labels_dir_exists': label_dir.exists(),
                'images_count': 0,
                'labels_count': 0,
                'status': 'unknown'
            }
            
            # Check directories
            if not img_dir.exists():
                validation_result['missing_dirs'].append(str(img_dir))
                validation_result['is_valid'] = False
                split_info['status'] = 'missing_images_dir'
            elif not any(img_dir.glob('*')):
                validation_result['empty_dirs'].append(str(img_dir))
                split_info['status'] = 'empty_images_dir'
            
            if not label_dir.exists():
                validation_result['missing_dirs'].append(str(label_dir))
                validation_result['is_valid'] = False
                if split_info['status'] == 'unknown':
                    split_info['status'] = 'missing_labels_dir'
            elif not any(label_dir.glob('*')):
                validation_result['empty_dirs'].append(str(label_dir))
                if split_info['status'] == 'unknown':
                    split_info['status'] = 'empty_labels_dir'
            
            # Count files jika directories exist
            if img_dir.exists():
                img_files = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png')) + list(img_dir.glob('*.jpeg'))
                split_info['images_count'] = len(img_files)
                validation_result['total_images'] += len(img_files)
            
            if label_dir.exists():
                label_files = list(label_dir.glob('*.txt'))
                split_info['labels_count'] = len(label_files)
            
            if split_info['status'] == 'unknown':
                split_info['status'] = 'valid' if split_info['images_count'] > 0 else 'empty'
            
            validation_result['splits'][split] = split_info
        
        return validation_result
    
    def create_output_structure(self, splits: Optional[List[str]] = None, 
                              force: bool = False) -> Dict[str, bool]:
        """📁 Create output directory structure"""
        splits = splits or self.splits
        results = {}
        
        # Create base output directory
        try:
            self.output_root.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            self.logger.error(f"❌ Failed to create output root {self.output_root}: {str(e)}")
            return {split: False for split in splits}
        
        for split in splits:
            try:
                img_dir, label_dir = self.get_output_paths(split)
                
                # Check jika sudah ada dan force=False
                if not force and img_dir.exists() and any(img_dir.glob('*')):
                    self.logger.warning(f"⚠️ Output directory {img_dir} already exists with files")
                    results[split] = False
                    continue
                
                # Create directories
                img_dir.mkdir(parents=True, exist_ok=True)
                label_dir.mkdir(parents=True, exist_ok=True)
                
                results[split] = True
                self.logger.debug(f"✅ Created output structure for {split}")
                
            except Exception as e:
                self.logger.error(f"❌ Failed to create output structure for {split}: {str(e)}")
                results[split] = False
        
        return results
    
    def create_auxiliary_dirs(self) -> Dict[str, bool]:
        """📁 Create auxiliary directories (backup, invalid, temp, etc.)"""
        auxiliary_dirs = {
            'backup': self.get_backup_dir(),
            'invalid': self.get_invalid_dir(),
            'temp': self.get_temp_dir(),
            'visualization': self.get_visualization_dir()
        }
        
        results = {}
        for name, directory in auxiliary_dirs.items():
            try:
                directory.mkdir(parents=True, exist_ok=True)
                results[name] = True
                self.logger.debug(f"✅ Created {name} directory: {directory}")
            except Exception as e:
                self.logger.error(f"❌ Failed to create {name} directory {directory}: {str(e)}")
                results[name] = False
        
        return results
    
    # === FILE OPERATIONS ===
    
    def resolve_relative_path(self, file_path: Path, base_dir: Path) -> str:
        """📐 Resolve relative path dari base directory"""
        try:
            return str(file_path.relative_to(base_dir))
        except ValueError:
            return str(file_path)
    
    def create_output_path(self, input_path: Path, input_base: Path, 
                          output_base: Path, new_filename: Optional[str] = None) -> Path:
        """📤 Create output path dari input path dengan structure preservation"""
        try:
            # Get relative path dari input base
            rel_path = input_path.relative_to(input_base)
            
            # Create output path
            if new_filename:
                output_path = output_base / rel_path.parent / new_filename
            else:
                output_path = output_base / rel_path
            
            # Ensure parent directory exists
            if self.auto_create:
                output_path.parent.mkdir(parents=True, exist_ok=True)
            
            return output_path
            
        except ValueError:
            # Fallback jika tidak bisa resolve relative path
            filename = new_filename or input_path.name
            output_path = output_base / filename
            
            if self.auto_create:
                output_path.parent.mkdir(parents=True, exist_ok=True)
            
            return output_path
    
    def move_to_invalid(self, file_path: Path, split: str, reason: str = "validation_failed") -> bool:
        """❌ Move file ke invalid directory dengan reason"""
        try:
            invalid_dir = self.get_invalid_dir(split)
            invalid_dir.mkdir(parents=True, exist_ok=True)
            
            # Create reason subdirectory
            reason_dir = invalid_dir / reason
            reason_dir.mkdir(exist_ok=True)
            
            # Move file
            target_path = reason_dir / file_path.name
            shutil.move(str(file_path), str(target_path))
            
            self.logger.debug(f"🗑️ Moved {file_path.name} to invalid/{split}/{reason}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to move {file_path} to invalid: {str(e)}")
            return False
    
    def create_backup(self, source_path: Path, split: Optional[str] = None, 
                     backup_name: Optional[str] = None) -> Optional[Path]:
        """🗄️ Create backup dari source path"""
        try:
            backup_dir = self.get_backup_dir(split)
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate backup name
            if backup_name:
                backup_path = backup_dir / backup_name
            else:
                timestamp = __import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')
                backup_path = backup_dir / f"{source_path.name}_backup_{timestamp}"
            
            # Create backup
            if source_path.is_file():
                shutil.copy2(source_path, backup_path)
            elif source_path.is_dir():
                shutil.copytree(source_path, backup_path, dirs_exist_ok=True)
            
            self.logger.info(f"💾 Created backup: {backup_path}")
            return backup_path
            
        except Exception as e:
            self.logger.error(f"❌ Failed to create backup for {source_path}: {str(e)}")
            return None
    
    # === CLEANUP OPERATIONS ===
    
    def cleanup_output_dirs(self, splits: Optional[List[str]] = None, 
                           confirm: bool = False) -> Dict[str, int]:
        """🧹 Cleanup output directories"""
        if not confirm:
            self.logger.warning("⚠️ Cleanup requires confirmation. Set confirm=True")
            return {}
        
        splits = splits or self.splits
        stats = {'files_removed': 0, 'dirs_removed': 0}
        
        for split in splits:
            try:
                split_dir = self.get_output_split_dir(split)
                
                if split_dir.exists():
                    # Count files before removal
                    file_count = sum(1 for _ in split_dir.rglob('*') if _.is_file())
                    
                    # Remove directory
                    shutil.rmtree(split_dir)
                    stats['files_removed'] += file_count
                    stats['dirs_removed'] += 1
                    
                    self.logger.info(f"🗑️ Cleaned {split}: {file_count} files removed")
                    
            except Exception as e:
                self.logger.error(f"❌ Failed to cleanup {split}: {str(e)}")
        
        return stats
    
    def cleanup_temp_dirs(self) -> int:
        """🧹 Cleanup temporary directories"""
        temp_base = self.output_root / '.temp'
        files_removed = 0
        
        try:
            if temp_base.exists():
                files_removed = sum(1 for _ in temp_base.rglob('*') if _.is_file())
                shutil.rmtree(temp_base)
                self.logger.info(f"🗑️ Cleaned temp directories: {files_removed} files")
        except Exception as e:
            self.logger.error(f"❌ Failed to cleanup temp: {str(e)}")
        
        return files_removed
    
    # === INFORMATION METHODS ===
    
    def get_path_summary(self) -> Dict[str, Any]:
        """📊 Get comprehensive path summary"""
        return {
            'data_root': str(self.data_root),
            'output_root': str(self.output_root),
            'configured_splits': list(self.local_paths.keys()),
            'available_splits': self.splits,
            'auto_create': self.auto_create,
            'structure_validation': self.validate_source_structure()
        }
    
    def check_disk_space(self, required_gb: float = 1.0) -> Dict[str, Any]:
        """💾 Check available disk space"""
        try:
            statvfs = os.statvfs(self.output_root.parent)
            available_bytes = statvfs.f_frsize * statvfs.f_available
            available_gb = available_bytes / (1024 ** 3)
            
            return {
                'available_gb': round(available_gb, 2),
                'required_gb': required_gb,
                'sufficient': available_gb >= required_gb,
                'percentage_used': round((1 - statvfs.f_available / statvfs.f_blocks) * 100, 1)
            }
        except Exception as e:
            self.logger.error(f"❌ Failed to check disk space: {str(e)}")
            return {'available_gb': 0, 'required_gb': required_gb, 'sufficient': False}

# === FACTORY FUNCTIONS ===

def create_path_manager(config: Dict[str, Any] = None) -> PathManager:
    """🏭 Factory untuk create PathManager"""
    return PathManager(config)

def create_path_config(source_dir: Union[str, Path], output_dir: Union[str, Path], 
                      create_missing: bool = True) -> PathConfig:
    """🏭 Factory untuk create PathConfig"""
    return PathConfig(source_dir, output_dir, create_missing=create_missing)

# === CONVENIENCE FUNCTIONS ===

def validate_source_safe(config: Dict[str, Any], splits: Optional[List[str]] = None) -> bool:
    """✅ One-liner safe source validation"""
    manager = create_path_manager(config)
    result = manager.validate_source_structure(splits)
    return result['is_valid']

def create_output_safe(config: Dict[str, Any], splits: Optional[List[str]] = None) -> bool:
    """📁 One-liner safe output creation"""
    manager = create_path_manager(config)
    results = manager.create_output_structure(splits)
    return all(results.values())

def get_paths_safe(config: Dict[str, Any], split: str) -> Tuple[Optional[Path], Optional[Path]]:
    """📁 One-liner safe path getting"""
    try:
        manager = create_path_manager(config)
        return manager.get_source_paths(split)
    except Exception:
        return None, None