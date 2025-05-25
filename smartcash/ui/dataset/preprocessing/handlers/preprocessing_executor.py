"""
File: smartcash/ui/dataset/preprocessing/handlers/preprocessing_executor.py
Deskripsi: Fixed preprocessing executor dengan proper button state management
"""

from typing import Dict, Any
from smartcash.dataset.preprocessor.utils.preprocessing_factory import PreprocessingFactory
from smartcash.ui.dataset.preprocessing.utils.config_extractor import get_config_extractor
from smartcash.ui.dataset.preprocessing.utils.progress_bridge import create_preprocessing_progress_bridge
from smartcash.ui.utils.button_state_manager import get_button_state_manager

def setup_preprocessing_executor(ui_components: Dict[str, Any], env=None) -> Dict[str, Any]:
    """Setup preprocessing executor dengan fixed button state management."""
    
    def execute_preprocessing_action(button=None) -> None:
        """Execute preprocessing dengan fixed button state management."""
        logger = ui_components.get('logger')
        button_manager = get_button_state_manager(ui_components)
        
        # Clear outputs first
        _clear_ui_outputs_and_confirmations(ui_components)
        
        with button_manager.operation_context('preprocessing'):
            try:
                logger and logger.info("🚀 Memulai preprocessing dataset")
                
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
                
                # Fixed: Extract parameters properly untuk avoid duplicate 'split' keyword
                split_param = processing_params['split']
                config_params = processing_params['config'].copy()
                force_reprocess = processing_params.get('force_reprocess', False)
                
                # Execute preprocessing dengan separated parameters
                result = preprocessing_service.coordinate_preprocessing(
                    split=split_param,
                    force_reprocess=force_reprocess,
                    **config_params
                )
                
                # Handle results
                if result['success']:
                    _handle_preprocessing_success(ui_components, result, logger)
                else:
                    raise Exception(result['message'])
                    
            except Exception as e:
                logger and logger.error(f"💥 Error preprocessing: {str(e)}")
                ui_components.get('error_operation', lambda x: None)(f"Preprocessing gagal: {str(e)}")
                _update_status_panel_error(ui_components, f"Preprocessing gagal: {str(e)}")
                raise
    
    # Register handler
    if 'preprocess_button' in ui_components:
        ui_components['preprocess_button'].on_click(execute_preprocessing_action)
    
    ui_components['execute_preprocessing'] = execute_preprocessing_action
    return ui_components

def _clear_ui_outputs_and_confirmations(ui_components: Dict[str, Any]) -> None:
    """Clear UI outputs dan confirmation area untuk fresh start."""
    for output_key in ['log_output', 'status', 'confirmation_area']:
        if output_key in ui_components and hasattr(ui_components[output_key], 'clear_output'):
            ui_components[output_key].clear_output(wait=True)

def _handle_preprocessing_success(ui_components: Dict[str, Any], result: Dict[str, Any], logger) -> None:
    """Handle successful preprocessing completion dengan status panel update."""
    total_images = result.get('total_images', 0)
    processing_time = result.get('processing_time', 0)
    
    # Update progress completion
    ui_components.get('complete_operation', lambda x: None)(
        f"Preprocessing selesai: {total_images:,} gambar dalam {processing_time:.1f}s"
    )
    
    # Update status panel
    _update_status_panel_success(ui_components, f"✅ Preprocessing berhasil: {total_images:,} gambar diproses")
    
    # Log detailed stats
    if logger:
        stats = result.get('split_stats', {})
        for split, split_stats in stats.items():
            if split_stats.get('complete', False):
                logger.success(f"📊 {split}: {split_stats['images']:,} gambar berhasil diproses")

def _update_status_panel_success(ui_components: Dict[str, Any], message: str) -> None:
    """Update status panel dengan success state."""
    from smartcash.ui.components.status_panel import update_status_panel
    if 'status_panel' in ui_components:
        update_status_panel(ui_components['status_panel'], message, "success")

def _update_status_panel_error(ui_components: Dict[str, Any], message: str) -> None:
    """Update status panel dengan error state."""
    from smartcash.ui.components.status_panel import update_status_panel
    if 'status_panel' in ui_components:
        update_status_panel(ui_components['status_panel'], message, "error")