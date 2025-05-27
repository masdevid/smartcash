class UUIDAugmentor:
    """Augmentor menggunakan engine reuse untuk avoid duplication"""
    
    def __init__(self, config: Dict[str, Any], paths: Dict[str, str], naming_manager, communicator=None):
        self.config, self.paths, self.naming_manager, self.comm = config, paths, naming_manager, communicator
        # Reuse engine instead of duplicating logic
        self.engine = AugmentationEngine(config, communicator)
    
    def process_split_augmentation(self, validation_result: Dict[str, Any], target_split: str, 
                                 progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Process menggunakan engine reuse"""
        return self.engine.run_augmentation_pipeline(target_split, progress_callback)
    
    def get_status(self) -> Dict[str, Any]:
        """Get status dengan engine integration"""
        return {'augmentor_ready': True, 'engine_integrated': True, 'uuid_aware': True}