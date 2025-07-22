"""
File: smartcash/dataset/preprocessor/utils/path_manager.py
Deskripsi: Path management untuk preprocessor dengan auto-creation
"""

from pathlib import Path
from typing import Dict, Any, List, Union, Tuple

from smartcash.common.logger import get_logger

class PathManager:
    """📁 Path management dengan auto-creation dan validation"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = get_logger(__name__)
        
        # Base paths dari config
        self.data_root = Path(self.config.get('data', {}).get('dir', 'data'))
        self.preprocessed_root = Path(self.config.get('data', {}).get('preprocessed_dir', 'data/preprocessed'))
        self.samples_root = Path(self.config.get('data', {}).get('samples_dir', 'data/samples'))
        
        # Auto-creation setting
        self.auto_create = True
    
    def get_source_paths(self, split: str) -> Tuple[Path, Path]:
        """📂 Get source images dan labels paths"""
        split_dir = self.data_root / split
        return split_dir / 'images', split_dir / 'labels'
    
    def get_output_paths(self, split: str) -> Tuple[Path, Path]:
        """📤 Get output images dan labels paths"""
        split_dir = self.preprocessed_root / split
        return split_dir / 'images', split_dir / 'labels'
    
    def get_samples_path(self, split: str = None) -> Path:
        """🎲 Get samples directory path"""
        return self.samples_root / split if split else self.samples_root
    
    def create_output_structure(self, splits: List[str]) -> bool:
        """🏗️ Create complete output structure"""
        try:
            for split in splits:
                img_dir, label_dir = self.get_output_paths(split)
                img_dir.mkdir(parents=True, exist_ok=True)
                label_dir.mkdir(parents=True, exist_ok=True)
            
            self.logger.info(f"🏗️ Created output structure for {len(splits)} splits")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Structure creation error: {str(e)}")
            return False
    
    def validate_source_structure(self, splits: List[str]) -> Dict[str, Any]:
        """✅ Validate source directory structure"""
        results = {'is_valid': True, 'missing_dirs': [], 'total_images': 0}
        
        for split in splits:
            img_dir, label_dir = self.get_source_paths(split)
            
            if not img_dir.exists():
                results['is_valid'] = False
                results['missing_dirs'].append(str(img_dir))
            elif img_dir.exists():
                # Count images
                image_count = len(list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png')))
                results['total_images'] += image_count
        
        return results
    
    def cleanup_output_dirs(self, splits: List[str] = None) -> Dict[str, int]:
        """🧹 Cleanup output directories"""
        import shutil
        
        splits = splits or ['train', 'valid', 'test']
        files_removed = 0
        
        for split in splits:
            split_dir = self.preprocessed_root / split
            if split_dir.exists():
                file_count = sum(1 for _ in split_dir.rglob('*') if _.is_file())
                shutil.rmtree(split_dir)
                files_removed += file_count
                self.logger.info(f"🗑️ Cleaned {split}: {file_count} files")
        
        return {'files_removed': files_removed}