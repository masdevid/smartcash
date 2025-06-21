"""
File: smartcash/dataset/preprocessor/validation/directory_validator.py
Deskripsi: Directory structure validation dengan auto-creation
"""

from pathlib import Path
from typing import Dict, Any, List, Union

from smartcash.common.logger import get_logger

class DirectoryValidator:
    """📁 Directory structure validator dengan auto-fix"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = get_logger(__name__)
        self.auto_create = self.config.get('auto_fix', True)
    
    def validate_structure(self, base_dir: Union[str, Path], 
                         splits: List[str] = None) -> Dict[str, Any]:
        """✅ Validate directory structure"""
        base_path = Path(base_dir)
        splits = splits or ['train', 'valid', 'test']
        
        results = {
            'is_valid': True,
            'base_exists': base_path.exists(),
            'splits': {},
            'missing_dirs': [],
            'created_dirs': []
        }
        
        # Check base directory
        if not results['base_exists']:
            if self.auto_create:
                base_path.mkdir(parents=True, exist_ok=True)
                results['created_dirs'].append(str(base_path))
                results['base_exists'] = True
                self.logger.info(f"📁 Created base directory: {base_path}")
            else:
                results['is_valid'] = False
                results['missing_dirs'].append(str(base_path))
        
        # Check split directories
        for split in splits:
            split_result = self._validate_split_structure(base_path, split)
            results['splits'][split] = split_result
            
            if not split_result['is_valid']:
                results['is_valid'] = False
            
            results['missing_dirs'].extend(split_result.get('missing_dirs', []))
            results['created_dirs'].extend(split_result.get('created_dirs', []))
        
        return results
    
    def _validate_split_structure(self, base_path: Path, split: str) -> Dict[str, Any]:
        """📂 Validate single split structure"""
        split_path = base_path / split
        images_path = split_path / 'images'
        labels_path = split_path / 'labels'
        
        result = {
            'is_valid': True,
            'split_exists': split_path.exists(),
            'images_exists': images_path.exists(),
            'labels_exists': labels_path.exists(),
            'missing_dirs': [],
            'created_dirs': []
        }
        
        # Check dan create directories
        for dir_path, dir_name in [
            (split_path, f'{split}'),
            (images_path, f'{split}/images'),
            (labels_path, f'{split}/labels')
        ]:
            if not dir_path.exists():
                if self.auto_create:
                    dir_path.mkdir(parents=True, exist_ok=True)
                    result['created_dirs'].append(str(dir_path))
                    self.logger.info(f"📁 Created directory: {dir_path}")
                else:
                    result['is_valid'] = False
                    result['missing_dirs'].append(str(dir_path))
        
        # Update exists flags
        result['split_exists'] = split_path.exists()
        result['images_exists'] = images_path.exists()
        result['labels_exists'] = labels_path.exists()
        
        return result
    
    def validate_output_structure(self, output_dir: Union[str, Path], 
                                splits: List[str] = None) -> Dict[str, Any]:
        """📤 Validate output directory structure"""
        return self.validate_structure(output_dir, splits)
    
    def create_preprocessing_structure(self, base_dir: Union[str, Path],
                                     splits: List[str] = None) -> bool:
        """🏗️ Create complete preprocessing structure"""
        try:
            base_path = Path(base_dir)
            splits = splits or ['train', 'valid', 'test']
            
            # Create base structure
            base_path.mkdir(parents=True, exist_ok=True)
            
            # Create split structures
            for split in splits:
                split_path = base_path / split
                (split_path / 'images').mkdir(parents=True, exist_ok=True)
                (split_path / 'labels').mkdir(parents=True, exist_ok=True)
            
            self.logger.info(f"🏗️ Created preprocessing structure in {base_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Structure creation error: {str(e)}")
            return False