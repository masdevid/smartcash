"""
File: smartcash/dataset/augmentor/service.py
Deskripsi: Updated service menggunakan refactored SRP modules dan orchestrator reuse
"""

from typing import Dict, Any, Optional, Callable

# Reuse dari updated orchestrator dan SRP modules
from smartcash.dataset.augmentor.services.augmentation_orchestrator import AugmentationOrchestrator
from smartcash.dataset.augmentor.utils.config_extractor import extract_split_aware_config

class AugmentationService:
    """Updated service dengan full orchestrator reuse dan SRP integration"""
    
    def __init__(self, config: Dict[str, Any], ui_components: Dict[str, Any] = None):
        # Extract config menggunakan SRP module
        self.config = extract_split_aware_config(config)
        # Reuse orchestrator instead of duplicating logic
        self.orchestrator = AugmentationOrchestrator(self.config, ui_components)
    
    def run_full_augmentation_pipeline(self, target_split: str = "train", progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Execute pipeline menggunakan orchestrator reuse"""
        return self.orchestrator.run_full_pipeline(target_split, progress_callback)
    
    def cleanup_augmented_data(self, include_preprocessed: bool = True, target_split: str = None) -> Dict[str, Any]:
        """Cleanup menggunakan orchestrator reuse"""
        return self.orchestrator.cleanup_augmented_data(include_preprocessed, target_split)
    
    def get_augmentation_status(self) -> Dict[str, Any]:
        """Get status dari orchestrator"""
        return self.orchestrator.get_augmentation_status()
    
    def generate_consistency_report(self) -> Dict[str, Any]:
        """Generate report dari orchestrator"""
        return self.orchestrator.generate_consistency_report()

# Factory functions menggunakan SRP config extractor
def create_service_from_ui(ui_components: Dict[str, Any]) -> AugmentationService:
    """Create service dari UI components dengan SRP config extraction"""
    from smartcash.dataset.augmentor.config import extract_ui_config
    config = extract_ui_config(ui_components)
    return AugmentationService(config, ui_components)

# One-liner utilities dengan orchestrator reuse
create_augmentation_service = lambda config, ui_components=None: AugmentationService(config, ui_components)
run_augmentation_pipeline = lambda config, ui_components=None, target_split='train': create_augmentation_service(config, ui_components).run_full_augmentation_pipeline(target_split)
cleanup_augmented_files = lambda config, include_prep=True, target_split=None: create_augmentation_service(config).cleanup_augmented_data(include_prep, target_split)
get_service_status = lambda config: create_augmentation_service(config).get_augmentation_status()