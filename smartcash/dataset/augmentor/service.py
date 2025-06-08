"""
File: smartcash/dataset/augmentor/service.py
Deskripsi: Updated service dengan unified progress tracking integration
"""

from typing import Dict, Any, Optional, Callable

class AugmentationService:
    """Updated service dengan unified progress tracking integration"""
    
    def __init__(self, config: Dict[str, Any], ui_components: Dict[str, Any] = None):
        self.config = self._normalize_config(config)
        self.ui_components = ui_components or {}
        
        # Setup progress communicator menggunakan UI progress manager
        self.communicator = self._create_progress_communicator()
        
        # Initialize orchestrator dengan communicator
        from smartcash.dataset.augmentor.services.augmentation_orchestrator import AugmentationOrchestrator
        self.orchestrator = AugmentationOrchestrator(self.config, self.ui_components)
    
    def run_full_augmentation_pipeline(self, target_split: str = "train", 
                                     progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Execute pipeline dengan unified progress tracking"""
        try:
            # Start operation tracking
            if self.communicator:
                self.communicator.start_operation("Augmentation Pipeline")
            
            # Execute dengan progress callback
            result = self.orchestrator.run_full_pipeline(target_split, progress_callback)
            
            # Complete operation tracking
            if self.communicator:
                if result.get('status') == 'success':
                    total_generated = result.get('total_generated', 0)
                    message = f"Pipeline berhasil: {total_generated} file dihasilkan"
                    self.communicator.complete_operation("Augmentation Pipeline", message)
                else:
                    error_msg = result.get('message', 'Unknown error')
                    self.communicator.error_operation("Augmentation Pipeline", error_msg)
            
            return result
            
        except Exception as e:
            if self.communicator:
                self.communicator.error_operation("Augmentation Pipeline", str(e))
            return {'status': 'error', 'message': str(e), 'total_generated': 0}
    
    def cleanup_augmented_data(self, include_preprocessed: bool = True, 
                             target_split: str = None) -> Dict[str, Any]:
        """Cleanup dengan progress tracking"""
        try:
            if self.communicator:
                self.communicator.start_operation("Cleanup Dataset")
            
            result = self.orchestrator.cleanup_augmented_data(include_preprocessed, target_split)
            
            if self.communicator:
                if result.get('status') == 'success':
                    total_deleted = result.get('total_deleted', 0)
                    message = f"Cleanup berhasil: {total_deleted} file dihapus"
                    self.communicator.complete_operation("Cleanup Dataset", message)
                else:
                    self.communicator.error_operation("Cleanup Dataset", result.get('message', 'Unknown error'))
            
            return result
            
        except Exception as e:
            if self.communicator:
                self.communicator.error_operation("Cleanup Dataset", str(e))
            return {'status': 'error', 'message': str(e), 'total_deleted': 0}
    
    def get_augmentation_status(self) -> Dict[str, Any]:
        """Get status dari orchestrator"""
        return self.orchestrator.get_augmentation_status()
    
    def generate_consistency_report(self) -> Dict[str, Any]:
        """Generate report dari orchestrator"""
        return self.orchestrator.generate_consistency_report()
    
    def _create_progress_communicator(self):
        """Create progress communicator dari UI components"""
        if not self.ui_components:
            return None
        
        try:
            from smartcash.ui.dataset.augmentation.utils.progress_utils import create_backend_communicator
            return create_backend_communicator(self.ui_components)
        except ImportError:
            return None
    
    def _normalize_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize config untuk backend compatibility"""
        # Extract config sections dengan safe defaults
        data_config = config.get('data', {})
        aug_config = config.get('augmentation', {})
        prep_config = config.get('preprocessing', {})
        
        return {
            'data': {
                'dir': data_config.get('dir', 'data')
            },
            'augmentation': {
                'num_variations': aug_config.get('num_variations', 3),
                'target_count': aug_config.get('target_count', 500),
                'target_split': aug_config.get('target_split', 'train'),
                'balance_classes': aug_config.get('balance_classes', True),
                'types': aug_config.get('types', ['combined']),
                'output_dir': aug_config.get('output_dir', 'data/augmented'),
                
                # Position parameters
                'fliplr': aug_config.get('fliplr', 0.5),
                'degrees': aug_config.get('degrees', 10),
                'translate': aug_config.get('translate', 0.1),
                'scale': aug_config.get('scale', 0.1),
                
                # Lighting parameters
                'hsv_h': aug_config.get('hsv_h', 0.015),
                'hsv_s': aug_config.get('hsv_s', 0.7),
                'brightness': aug_config.get('brightness', 0.2),
                'contrast': aug_config.get('contrast', 0.2)
            },
            'preprocessing': {
                'output_dir': prep_config.get('output_dir', 'data/preprocessed'),
                'normalization': prep_config.get('normalization', {
                    'enabled': True,
                    'method': 'minmax',
                    'target_size': [640, 640]
                })
            }
        }

def create_service_from_ui(ui_components: Dict[str, Any]) -> AugmentationService:
    """Create service dari UI components dengan config extraction"""
    try:
        # Extract config dari UI components
        from smartcash.ui.dataset.augmentation.handlers.config_extractor import extract_augmentation_config
        config = extract_augmentation_config(ui_components)
        
        return AugmentationService(config, ui_components)
        
    except Exception as e:
        # Log error ke UI jika ada
        try:
            from smartcash.ui.dataset.augmentation.utils.progress_utils import create_unified_progress_manager
            progress_manager = create_unified_progress_manager(ui_components)
            progress_manager._log_ui(f"❌ Service creation error: {str(e)}", 'error')
        except Exception:
            print(f"❌ Service creation error: {str(e)}")
        
        # Return service dengan minimal config
        minimal_config = {
            'data': {'dir': 'data'},
            'augmentation': {'num_variations': 3, 'target_count': 500, 'target_split': 'train'},
            'preprocessing': {'output_dir': 'data/preprocessed'}
        }
        return AugmentationService(minimal_config, ui_components)

# One-liner utilities
create_augmentation_service = lambda config, ui_components=None: AugmentationService(config, ui_components)
run_augmentation_pipeline = lambda config, ui_components=None, target_split='train': create_service_from_ui(ui_components).run_full_augmentation_pipeline(target_split)
cleanup_augmented_files = lambda ui_components, include_prep=True, target_split=None: create_service_from_ui(ui_components).cleanup_augmented_data(include_prep, target_split)