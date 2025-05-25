"""
File: smartcash/ui/dataset/preprocessing/handlers/preprocessing_executor.py
Deskripsi: SRP handler untuk eksekusi preprocessing dengan integrasi service layer baru
"""

from typing import Dict, Any, Callable
from smartcash.dataset.preprocessor.utils.preprocessing_factory import PreprocessingFactory
from smartcash.ui.dataset.preprocessing.utils.config_extractor import get_config_extractor
from smartcash.ui.dataset.preprocessing.utils.progress_bridge import create_preprocessing_progress_bridge
from smartcash.ui.utils.button_state_manager import get_button_state_manager

def setup_preprocessing_executor(ui_components: Dict[str, Any], env=None) -> Dict[str, Any]:
    """Setup preprocessing executor dengan service layer integration."""
    
    def execute_preprocessing_action(button=None) -> None:
        """Execute preprocessing dengan comprehensive service integration."""
        logger = ui_components.get('logger')
        button_manager = get_button_state_manager(ui_components)
        
        with button_manager.operation_context('preprocessing'):
            try:
                logger and logger.info("🚀 Memulai preprocessing dataset")
                
                # Clear UI outputs
                _clear_ui_outputs(ui_components)
                
                # Setup progress tracking
                ui_components.get('show_for_operation', lambda x: None)('download')
                
                # Extract config dari UI
                config_extractor = get_config_extractor(ui_components)
                processing_params = config_extractor.extract_processing_parameters()
                config = config_extractor.get_full_config()
                
                logger and logger.info(f"🔧 Config: {processing_params['summary']}")
                
                # Create progress bridge
                progress_bridge = create_preprocessing_progress_bridge(ui_components)
                
                # Create preprocessing service
                preprocessing_service = PreprocessingFactory.create_preprocessing_manager(
                    config, logger, progress_bridge.notify_progress
                )
                
                # Execute preprocessing
                result = preprocessing_service.coordinate_preprocessing(
                    split=processing_params['split'],
                    force_reprocess=processing_params.get('force_reprocess', False),
                    **processing_params['config']
                )
                
                # Handle results
                if result['success']:
                    _handle_preprocessing_success(ui_components, result, logger)
                else:
                    raise Exception(result['message'])
                    
            except Exception as e:
                logger and logger.error(f"💥 Error preprocessing: {str(e)}")
                ui_components.get('error_operation', lambda x: None)(f"Preprocessing gagal: {str(e)}")
                raise
    
    # Register handler
    if 'preprocess_button' in ui_components:
        ui_components['preprocess_button'].on_click(execute_preprocessing_action)
    
    # Add action reference
    ui_components['execute_preprocessing'] = execute_preprocessing_action
    
    return ui_components

def _clear_ui_outputs(ui_components: Dict[str, Any]) -> None:
    """Clear UI outputs untuk fresh start."""
    for output_key in ['log_output', 'status', 'confirmation_area']:
        if output_key in ui_components and hasattr(ui_components[output_key], 'clear_output'):
            ui_components[output_key].clear_output(wait=True)

def _handle_preprocessing_success(ui_components: Dict[str, Any], result: Dict[str, Any], logger) -> None:
    """Handle successful preprocessing completion."""
    total_images = result.get('total_images', 0)
    processing_time = result.get('processing_time', 0)
    
    # Update progress completion
    ui_components.get('complete_operation', lambda x: None)(
        f"Preprocessing selesai: {total_images:,} gambar dalam {processing_time:.1f}s"
    )
    
    # Update status panel
    from smartcash.ui.components.status_panel import update_status_panel
    if 'status_panel' in ui_components:
        update_status_panel(
            ui_components['status_panel'],
            f"✅ Preprocessing berhasil: {total_images:,} gambar diproses",
            "success"
        )
    
    # Log detailed stats
    if logger:
        stats = result.get('split_stats', {})
        for split, split_stats in stats.items():
            if split_stats.get('complete', False):
                logger.success(f"📊 {split}: {split_stats['images']:,} gambar berhasil diproses")