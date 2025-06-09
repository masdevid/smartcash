"""
File: smartcash/dataset/augmentor/utils/symlink_manager.py
Deskripsi: Manager untuk symlink creation ke preprocessed folder
"""

import os
from pathlib import Path

class SymlinkManager:
    """🔗 Manager untuk symlink creation"""
    
    def __init__(self, config):
        self.config = config
    
    def create_augmented_symlinks(self, aug_path: str, prep_path: str) -> Dict[str, Any]:
        """Create symlinks dari augmented ke preprocessed folder"""
        try:
            aug_dir = Path(aug_path)
            prep_dir = Path(prep_path)
            
            # Ensure preprocessed directory exists
            (prep_dir / 'images').mkdir(parents=True, exist_ok=True)
            (prep_dir / 'labels').mkdir(parents=True, exist_ok=True)
            
            created_count = 0
            
            # Create symlinks untuk images
            for img_file in (aug_dir / 'images').glob('aug_*.jpg'):
                symlink_path = prep_dir / 'images' / img_file.name
                if not symlink_path.exists():
                    os.symlink(img_file.absolute(), symlink_path)
                    created_count += 1
            
            # Create symlinks untuk labels
            for label_file in (aug_dir / 'labels').glob('aug_*.txt'):
                symlink_path = prep_dir / 'labels' / label_file.name
                if not symlink_path.exists():
                    os.symlink(label_file.absolute(), symlink_path)
                    created_count += 1
            
            return {'status': 'success', 'total_created': created_count}
            
        except Exception as e:
            return {'status': 'error', 'message': str(e), 'total_created': 0}