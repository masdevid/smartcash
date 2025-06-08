"""
File: smartcash/dataset/augmentor/service.py
Deskripsi: Enhanced service dengan granular progress tracking dan dataset comparison
"""

from typing import Dict, Any, Optional, Callable

class AugmentationService:
    """Enhanced service dengan UI integration dan dataset comparison"""
    
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
        """Execute pipeline dengan granular progress tracking"""
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
                    total_normalized = result.get('total_normalized', 0)
                    message = f"Pipeline berhasil: {total_generated} generated, {total_normalized} normalized"
                    self.communicator.complete_operation("Augmentation Pipeline", message)
                else:
                    error_msg = result.get('message', 'Unknown error')
                    self.communicator.error_operation("Augmentation Pipeline", error_msg)
            
            return result
            
        except Exception as e:
            if self.communicator:
                self.communicator.error_operation("Augmentation Pipeline", str(e))
            return {'status': 'error', 'message': str(e), 'total_generated': 0}
    
    def check_dataset_readiness(self, target_split: str = "train") -> Dict[str, Any]:
        """🆕 Check dataset readiness dengan raw vs preprocessed comparison"""
        try:
            if self.communicator:
                self.communicator.start_operation("Dataset Check")
                self.communicator.progress('overall', 10, 100, "🔍 Mencari lokasi data")
            
            # Get data directories
            from smartcash.dataset.augmentor.utils.path_operations import get_best_data_location
            data_location = get_best_data_location()
            
            if self.communicator:
                self.communicator.progress('overall', 30, 100, "📊 Menganalisis raw dataset")
            
            # Check raw dataset
            from smartcash.dataset.augmentor.utils.dataset_detector import detect_split_structure, compare_raw_vs_preprocessed
            raw_info = detect_split_structure(data_location)
            
            if self.communicator:
                self.communicator.progress('overall', 50, 100, "🔄 Mengecek preprocessed dataset")
            
            # Compare with preprocessed
            prep_location = f"{data_location}/preprocessed"
            comparison = compare_raw_vs_preprocessed(data_location, prep_location)
            
            if self.communicator:
                self.communicator.progress('overall', 70, 100, "🔧 Mengecek augmented dataset")
            
            # Check existing augmented data
            aug_location = f"{data_location}/augmented"
            aug_info = detect_split_structure(aug_location)
            
            if self.communicator:
                self.communicator.progress('overall', 90, 100, "📋 Menyusun laporan")
            
            # Compile results
            result = {
                'status': 'success',
                'data_location': data_location,
                'raw_dataset': raw_info,
                'preprocessed_dataset': {'status': 'checked', 'location': prep_location},
                'augmented_dataset': aug_info,
                'comparison': comparison,
                'ready_for_augmentation': comparison['augmentation_ready'],
                'target_split': target_split,
                'recommendations': comparison['recommendations']
            }
            
            # Log detailed results
            if self.communicator:
                self.communicator.log_info(f"📁 Data location: {data_location}")
                
                if raw_info['status'] == 'success':
                    total_raw = raw_info.get('total_images', 0)
                    available_splits = raw_info.get('available_splits', [])
                    self.communicator.log_success(f"✅ Raw Dataset: {total_raw} gambar di {len(available_splits)} splits")
                    if available_splits:
                        self.communicator.log_info(f"📂 Available splits: {', '.join(available_splits)}")
                else:
                    self.communicator.log_error(f"❌ Raw dataset: {raw_info.get('message', 'Not found')}")
                
                # Preprocessed status
                if comparison['preprocessed_exists']:
                    prep_images = sum(details.get('preprocessed_images', 0) for details in comparison['split_comparison'].values())
                    self.communicator.log_success(f"✅ Preprocessed Dataset: {prep_images} file")
                else:
                    self.communicator.log_warning("🔄 Preprocessed Dataset: Belum tersedia")
                
                # Augmented status
                if aug_info['status'] == 'success' and aug_info.get('total_images', 0) > 0:
                    aug_images = aug_info.get('total_images', 0)
                    self.communicator.log_success(f"✅ Augmented Dataset: {aug_images} file")
                else:
                    self.communicator.log_info("🔄 Augmented Dataset: Belum ada file")
                
                # Final readiness
                ready_status = "✅ Siap" if comparison['augmentation_ready'] else "❌ Perlu preprocessing"
                self.communicator.log_info(f"🎯 Status augmentasi: {ready_status}")
                
                self.communicator.complete_operation("Dataset Check", f"Check completed - {ready_status}")
            
            return result
            
        except Exception as e:
            if self.communicator:
                self.communicator.error_operation("Dataset Check", str(e))
            return {'status': 'error', 'message': str(e), 'ready_for_augmentation': False}
    
    def cleanup_augmented_data(self, include_preprocessed: bool = True, 
                             target_split: str = None) -> Dict[str, Any]:
        """Cleanup dengan granular progress tracking"""
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
        """Get comprehensive status dengan dataset comparison"""
        try:
            orchestrator_status = self.orchestrator.get_augmentation_status()
            
            # Add dataset readiness check
            from smartcash.dataset.augmentor.utils.path_operations import get_best_data_location
            from smartcash.dataset.augmentor.utils.dataset_detector import compare_raw_vs_preprocessed
            
            data_location = get_best_data_location()
            prep_location = f"{data_location}/preprocessed"
            comparison = compare_raw_vs_preprocessed(data_location, prep_location)
            
            return {
                **orchestrator_status,
                'dataset_comparison': comparison,
                'ready_for_augmentation': comparison['augmentation_ready']
            }
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
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
    """Create service dari UI components dengan enhanced config extraction"""
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
check_dataset_readiness = lambda ui_components, target_split='train': create_service_from_ui(ui_components).check_dataset_readiness(target_split)
cleanup_augmented_files = lambda ui_components, include_prep=True, target_split=None: create_service_from_ui(ui_components).cleanup_augmented_data(include_prep, target_split)