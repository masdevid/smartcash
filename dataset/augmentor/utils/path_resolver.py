"""
File: smartcash/dataset/augmentor/utils/path_resolver.py
Deskripsi: Path resolver untuk smart path handling
"""

class PathResolver:
    """🗺️ Resolver untuk smart path handling dengan defaults"""
    
    def __init__(self, config):
        self.config = config
        self.data_dir = config.get('data', {}).get('dir', 'data')
    
    def get_raw_path(self, split: str) -> str:
        """Get raw data path untuk split"""
        return str(Path(self.data_dir) / split)
    
    def get_augmented_path(self, split: str) -> str:
        """Get augmented data path untuk split"""
        return str(Path(self.data_dir) / 'augmented' / split)
    
    def get_preprocessed_path(self, split: str) -> str:
        """Get preprocessed data path untuk split"""
        return str(Path(self.data_dir) / 'preprocessed' / split)
    
    def get_all_paths(self) -> Dict[str, str]:
        """Get all path mappings"""
        return {
            'data_dir': self.data_dir,
            'raw_train': self.get_raw_path('train'),
            'raw_valid': self.get_raw_path('valid'),
            'raw_test': self.get_raw_path('test'),
            'aug_train': self.get_augmented_path('train'),
            'aug_valid': self.get_augmented_path('valid'),
            'aug_test': self.get_augmented_path('test'),
            'prep_train': self.get_preprocessed_path('train'),
            'prep_valid': self.get_preprocessed_path('valid'),
            'prep_test': self.get_preprocessed_path('test')
        }