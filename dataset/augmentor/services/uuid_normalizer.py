"""
File: smartcash/dataset/augmentor/services/uuid_normalizer.py
Deskripsi: Normalizer service menggunakan core normalizer reuse  
"""

class UUIDNormalizer:
    """Normalizer menggunakan core normalizer reuse"""
    
    def __init__(self, config: Dict[str, Any], paths: Dict[str, str], naming_manager, communicator=None):
        self.config, self.paths, self.naming_manager, self.comm = config, paths, naming_manager, communicator
        # Reuse normalizer instead of duplicating
        self.normalizer = NormalizationEngine(config, communicator)
    
    def process_uuid_normalization(self, target_split: str, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Process menggunakan normalizer reuse"""
        return self.normalizer.normalize_augmented_data(
            f"{self.paths['aug_dir']}/{target_split}",
            self.paths['prep_dir'], 
            target_split
        )
    
    def get_status(self) -> Dict[str, Any]:
        """Get status dengan normalizer integration"""
        return {'normalizer_ready': True, 'engine_integrated': True, 'uuid_aware': True}