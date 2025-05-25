"""
File: smartcash/ui/dataset/preprocessing/handlers/dataset_checker.py
Deskripsi: Fixed dataset checker dengan safe key access dan proper error handling
"""

from typing import Dict, Any
from smartcash.dataset.preprocessor.utils.preprocessing_factory import PreprocessingFactory
from smartcash.ui.dataset.preprocessing.utils.config_extractor import get_config_extractor
from smartcash.ui.utils.button_state_manager import get_button_state_manager

def setup_dataset_checker(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Setup dataset checker dengan safe error handling dan comprehensive analysis."""
    
    def execute_check_action(button=None) -> None:
        """Execute dataset checking dengan safe key access."""
        logger = ui_components.get('logger')
        button_manager = get_button_state_manager(ui_components)
        
        _clear_ui_outputs(ui_components)
        
        with button_manager.operation_context('check'):
            try:
                logger and logger.info("🔍 Memeriksa dataset untuk preprocessing")
                
                ui_components.get('show_for_operation', lambda x: None)('check')
                _update_progress(ui_components, 10, "Memulai analisis dataset")
                
                # Get config dengan fallback
                config_extractor = get_config_extractor(ui_components)
                config = config_extractor.get_full_config()
                
                # Create checker service
                checker = PreprocessingFactory.create_dataset_checker(config, logger)
                
                # Phase 1: Check source dataset (20-60%)
                _update_progress(ui_components, 30, "Menganalisis source dataset")
                source_result = _safe_check_dataset(checker, 'source', detailed=True)
                
                _update_progress(ui_components, 50, "Validasi struktur dataset")
                
                # Phase 2: Check preprocessed dataset (60-80%) 
                _update_progress(ui_components, 70, "Menganalisis preprocessed dataset")
                preprocessed_result = _safe_check_dataset(checker, 'preprocessed', detailed=True)
                
                # Phase 3: Generate reports (80-100%)
                _update_progress(ui_components, 90, "Membuat laporan analisis")
                
                # Display results dengan safe key access
                _display_check_results_safe(ui_components, source_result, preprocessed_result, logger)
                
                _update_progress(ui_components, 100, "Analisis dataset selesai")
                ui_components.get('complete_operation', lambda x: None)("Analisis dataset selesai")
                
            except Exception as e:
                error_msg = f"Check dataset gagal: {str(e)}"
                logger and logger.error(f"💥 {error_msg}")
                ui_components.get('error_operation', lambda x: None)(error_msg)
                _update_status_panel_error(ui_components, error_msg)
                raise
    
    # Register handler
    if 'check_button' in ui_components:
        ui_components['check_button'].on_click(execute_check_action)
    
    ui_components['execute_check'] = execute_check_action
    return ui_components

def _safe_check_dataset(checker, dataset_type: str, detailed: bool = True) -> Dict[str, Any]:
    """Safe wrapper untuk dataset checking dengan proper error handling."""
    try:
        if dataset_type == 'source':
            return checker.check_source_dataset(detailed=detailed)
        elif dataset_type == 'preprocessed':
            return checker.check_preprocessed_dataset(detailed=detailed)
        else:
            return {'valid': False, 'message': f'Unknown dataset type: {dataset_type}'}
    except Exception as e:
        return {
            'valid': False,
            'message': f'Error checking {dataset_type} dataset: {str(e)}',
            'total_images': 0,
            'splits': {},
            'error': True
        }

def _display_check_results_safe(ui_components: Dict[str, Any], source_result: Dict[str, Any], 
                               preprocessed_result: Dict[str, Any], logger) -> None:
    """Display check results dengan safe key access untuk mencegah KeyError."""
    from IPython.display import display, HTML
    
    # Safe access untuk source dataset
    source_valid = source_result.get('valid', False)
    source_total = source_result.get('total_images', 0)
    
    if source_valid and source_total > 0:
        logger and logger.success(f"✅ Source dataset: {source_total:,} gambar valid")
        _log_split_details_safe(source_result.get('splits', {}), logger, "Source")
        _update_status_panel_success(ui_components, f"Dataset siap: {source_total:,} gambar tersedia")
    else:
        source_msg = source_result.get('message', 'Dataset tidak valid')
        logger and logger.error(f"❌ Source dataset invalid: {source_msg}")
        _update_status_panel_error(ui_components, "Dataset tidak valid untuk preprocessing")
        return
    
    # Safe access untuk preprocessed dataset
    preprocessed_valid = preprocessed_result.get('valid', False)
    
    if preprocessed_valid:
        preprocessed_total = preprocessed_result.get('total_processed', 0)
        logger and logger.success(f"💾 Preprocessed dataset: {preprocessed_total:,} gambar")
        _log_split_details_safe(preprocessed_result.get('splits', {}), logger, "Preprocessed")
    else:
        preprocessed_msg = preprocessed_result.get('message', 'Belum ada preprocessed dataset')
        logger and logger.info(f"ℹ️ Preprocessed: {preprocessed_msg}")
    
    # Display detailed report dengan safe HTML generation
    if 'log_output' in ui_components:
        with ui_components['log_output']:
            # Source report
            source_report = source_result.get('report', 'Tidak ada detail laporan tersedia')
            if source_report and source_report != 'Tidak ada detail laporan tersedia':
                display(HTML(f"<pre style='background:#f8f9fa;padding:10px;border-radius:5px;font-size:12px;'>{source_report}</pre>"))
            
            # Preprocessed report
            if preprocessed_valid:
                preprocessed_report = preprocessed_result.get('report', 'Tidak ada detail laporan tersedia')
                if preprocessed_report and preprocessed_report != 'Tidak ada detail laporan tersedia':
                    display(HTML(f"<pre style='background:#f0f8ff;padding:10px;border-radius:5px;margin-top:10px;font-size:12px;'>{preprocessed_report}</pre>"))

def _log_split_details_safe(splits: Dict[str, Any], logger, dataset_type: str) -> None:
    """Log split details dengan safe key access untuk mencegah KeyError."""
    if not logger or not splits:
        return
    
    for split_name in ['train', 'valid', 'test']:
        split_data = splits.get(split_name, {})
        
        if not split_data or not split_data.get('exists', False):
            continue
            
        try:
            if dataset_type == "Source":
                # Safe access untuk source dataset
                images_count = split_data.get('images', 0)
                labels_count = split_data.get('labels', 0)
                
                if images_count > 0:
                    logger.info(f"📂 {split_name}: {images_count:,} gambar, {labels_count:,} label")
            else:
                # Safe access untuk preprocessed dataset
                processed_count = split_data.get('processed', split_data.get('images', 0))
                
                if processed_count > 0:
                    logger.info(f"💾 {split_name}: {processed_count:,} preprocessed")
                    
        except Exception as e:
            logger.debug(f"🔧 Error logging {split_name} details: {str(e)}")

def _clear_ui_outputs(ui_components: Dict[str, Any]) -> None:
    """Clear UI outputs untuk fresh display."""
    for output_key in ['log_output', 'status', 'confirmation_area']:
        if output_key in ui_components and hasattr(ui_components[output_key], 'clear_output'):
            ui_components[output_key].clear_output(wait=True)

def _update_progress(ui_components: Dict[str, Any], progress: int, message: str) -> None:
    """Update progress dengan bounds checking."""
    progress = max(0, min(100, progress))
    ui_components.get('update_progress', lambda *args: None)('overall', progress, message)

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