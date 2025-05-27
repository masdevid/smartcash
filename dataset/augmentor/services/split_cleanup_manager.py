"""
File: smartcash/dataset/augmentor/services/split_cleanup_manager.py
Deskripsi: Cleanup manager menggunakan SRP cleanup reuse
"""

class SplitCleanupManager:
    """Cleanup manager menggunakan SRP cleanup operations reuse"""
    
    def __init__(self, config: Dict[str, Any], paths: Dict[str, str], naming_manager, communicator=None):
        self.config, self.paths, self.naming_manager, self.comm = config, paths, naming_manager, communicator
        self.progress = create_progress_tracker(communicator)
    
    def cleanup_split_data(self, include_preprocessed: bool = True, target_split: str = 'train') -> Dict[str, Any]:
        """Cleanup menggunakan SRP cleanup operations"""
        return cleanup_split_aware(self.paths['aug_dir'], 
                                 self.paths['prep_dir'] if include_preprocessed else None,
                                 target_split, self.progress)
    
    def get_status(self) -> Dict[str, Any]:
        """Get cleanup status"""
        return {'cleanup_ready': True, 'split_aware': True, 'progress_integrated': True}

# One-liner utilities menggunakan full SRP reuse
create_augmentation_orchestrator = lambda config, ui_components=None: AugmentationOrchestrator(config, ui_components)
create_split_validator = lambda config, paths, comm=None: SplitValidator(config, paths, comm)
create_uuid_augmentor = lambda config, paths, naming_manager, comm=None: UUIDAugmentor(config, paths, naming_manager, comm)
create_uuid_normalizer = lambda config, paths, naming_manager, comm=None: UUIDNormalizer(config, paths, naming_manager, comm)
create_cleanup_manager = lambda config, paths, naming_manager, comm=None: SplitCleanupManager(config, paths, naming_manager, comm)

# Pipeline execution one-liners
run_full_pipeline = lambda config, ui_components=None, target_split='train': create_augmentation_orchestrator(config, ui_components).run_full_pipeline(target_split)
cleanup_augmented_data = lambda config, target_split='train', include_prep=True: create_augmentation_orchestrator(config).cleanup_augmented_data(include_prep, target_split)
validate_split_dataset = lambda config, paths, target_split: create_split_validator(config, paths).validate_dataset_with_uuid_check(target_split)