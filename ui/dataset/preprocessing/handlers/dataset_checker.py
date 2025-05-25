"""
File: smartcash/ui/dataset/preprocessing/handlers/dataset_checker.py
Deskripsi: Fixed dataset checker dengan proper button state management
"""

from typing import Dict, Any
from smartcash.dataset.preprocessor.utils.preprocessing_factory import PreprocessingFactory
from smartcash.ui.dataset.preprocessing.utils.config_extractor import get_config_extractor
from smartcash.ui.utils.button_state_manager import get_button_state_manager

def setup_dataset_checker(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Setup dataset checker dengan proper button state management."""
    
    def execute_check_action(button=None) -> None:
        """Execute dataset checking dengan comprehensive analysis dan progress."""
        logger = ui_components.get('logger')
        button_manager = get_button_state_manager(ui_components)
        
        # Clear outputs first
        _clear_ui_outputs(ui_components)
        
        with button_manager.operation_context('check'):
            try:
                logger and logger.info("🔍 Memeriksa dataset untuk preprocessing")
                
                # Show progress untuk check operation
                ui_components.get('show_for_operation', lambda x: None)('check')
                
                # Phase 1: Setup (0-20%)
                _update_progress(ui_components, 10, "Memulai analisis dataset")
                
                # Get config
                config_extractor = get_config_extractor(ui_components)
                config = config_extractor.get_full_config()
                
                # Create checker service
                checker = PreprocessingFactory.create_dataset_checker(config, logger)
                
                # Phase 2: Check source (20-60%)
                _update_progress(ui_components, 30, "Menganalisis source dataset")
                source_result = checker.check_source_dataset(detailed=True)
                
                _update_progress(ui_components, 50, "Validasi struktur dataset")
                
                # Phase 3: Check preprocessed (60-80%)
                _update_progress(ui_components, 70, "Menganalisis preprocessed dataset")
                preprocessed_result = checker.check_preprocessed_dataset(detailed=True)
                
                # Phase 4: Generate reports (80-100%)
                _update_progress(ui_components, 90, "Membuat laporan analisis")
                
                # Display results
                _display_check_results(ui_components, source_result, preprocessed_result, logger)
                
                _update_progress(ui_components, 100, "Analisis dataset selesai")
                ui_components.get('complete_operation', lambda x: None)("Analisis dataset selesai")
                
            except Exception as e:
                logger and logger.error(f"💥 Error checking dataset: {str(e)}")
                ui_components.get('error_operation', lambda x: None)(f"Check dataset gagal: {str(e)}")
                _update_status_panel_error(ui_components, f"Check dataset gagal: {str(e)}")
                raise
    
    # Register handler
    if 'check_button' in ui_components:
        ui_components['check_button'].on_click(execute_check_action)
    
    ui_components['execute_check'] = execute_check_action
    return ui_components

def _clear_ui_outputs(ui_components: Dict[str, Any]) -> None:
    """Clear UI outputs untuk fresh display."""
    for output_key in ['log_output', 'status', 'confirmation_area']:
        if output_key in ui_components and hasattr(ui_components[output_key], 'clear_output'):
            ui_components[output_key].clear_output(wait=True)

def _update_progress(ui_components: Dict[str, Any], progress: int, message: str) -> None:
    """Update progress dengan bounds checking."""
    progress = max(0, min(100, progress))
    ui_components.get('update_progress', lambda *args: None)('overall', progress, message)

def _display_check_results(ui_components: Dict[str, Any], source_result: Dict[str, Any], 
                          preprocessed_result: Dict[str, Any], logger) -> None:
    """Display comprehensive check results."""
    from IPython.display import display, HTML
    
    # Source dataset results
    if source_result['valid']:
        logger and logger.success(f"✅ Source dataset: {source_result['total_images']:,} gambar valid")
        _log_split_details(source_result['splits'], logger, "Source")
        _update_status_panel_success(ui_components, f"Dataset siap: {source_result['total_images']:,} gambar tersedia")
    else:
        logger and logger.error(f"❌ Source dataset invalid: {source_result['message']}")
        _update_status_panel_error(ui_components, "Dataset tidak valid untuk preprocessing")
        return
    
    # Preprocessed dataset results
    if preprocessed_result['valid']:
        logger and logger.success(f"💾 Preprocessed dataset: {preprocessed_result['total_processed']:,} gambar")
        _log_split_details(preprocessed_result['splits'], logger, "Preprocessed")
    else:
        logger and logger.info(f"ℹ️ Preprocessed: {preprocessed_result['message']}")
    
    # Display detailed report
    if 'log_output' in ui_components:
        with ui_components['log_output']:
            # Source report
            if 'report' in source_result:
                display(HTML(f"<pre style='background:#f8f9fa;padding:10px;border-radius:5px;font-size:12px;'>{source_result['report']}</pre>"))
            
            # Preprocessed report jika ada
            if preprocessed_result['valid'] and 'report' in preprocessed_result:
                display(HTML(f"<pre style='background:#f0f8ff;padding:10px;border-radius:5px;margin-top:10px;font-size:12px;'>{preprocessed_result['report']}</pre>"))

def _log_split_details(splits: Dict[str, Any], logger, dataset_type: str) -> None:
    """Log split details dengan format yang rapi."""
    if not logger:
        return
    
    for split in ['train', 'valid', 'test']:
        split_data = splits.get(split, {})
        if split_data.get('exists', False):
            if dataset_type == "Source":
                count = split_data.get('images', 0)
                labels = split_data.get('labels', 0)
                logger.info(f"📂 {split}: {count:,} gambar, {labels:,} label")
            else:
                count = split_data.get('processed', 0)
                logger.info(f"💾 {split}: {count:,} preprocessed")

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