"""
File: smartcash/ui/dataset/preprocessing/handlers/preprocessing_handlers.py
Deskripsi: Unified handlers yang terintegrasi dengan dataset/preprocessors untuk eliminasi duplikasi
"""

from typing import Dict, Any
from smartcash.dataset.preprocessor.core.preprocessing_manager import PreprocessingManager
from smartcash.dataset.preprocessor.operations.dataset_checker import DatasetChecker
from smartcash.dataset.preprocessor.operations.cleanup_executor import CleanupExecutor
from smartcash.ui.components.confirmation_dialog import create_destructive_confirmation
from smartcash.ui.utils.button_state_manager import get_button_state_manager
from smartcash.common.config.manager import get_config_manager

def setup_preprocessing_handlers(ui_components: Dict[str, Any], config: Dict[str, Any], env=None) -> Dict[str, Any]:
    """Setup unified handlers dengan integrasi preprocessors"""
    
    # Setup progress callback
    def create_progress_callback():
        def progress_callback(**kwargs):
            progress = kwargs.get('progress', 0)
            message = kwargs.get('message', 'Processing...')
            step = kwargs.get('step', 0)
            ui_components.get('update_progress', lambda *a: None)('overall', progress, message)
        return progress_callback
    
    ui_components['progress_callback'] = create_progress_callback()
    
    # Setup handlers
    setup_preprocessing_handler(ui_components, config)
    setup_check_handler(ui_components, config)
    setup_cleanup_handler(ui_components, config)
    setup_config_handlers(ui_components, config)
    
    return ui_components

def setup_preprocessing_handler(ui_components: Dict[str, Any], config: Dict[str, Any]):
    """Setup preprocessing handler terintegrasi"""
    
    def execute_preprocessing(button=None):
        button_manager = get_button_state_manager(ui_components)
        logger = ui_components.get('logger')
        
        _clear_outputs(ui_components)
        
        with button_manager.operation_context('preprocessing'):
            try:
                logger and logger.info("🚀 Memulai preprocessing dataset")
                ui_components.get('show_for_operation', lambda x: None)('preprocessing')
                
                # Extract config dari UI
                params = _extract_processing_params(ui_components)
                processing_config = {**config, 'preprocessing': {**config.get('preprocessing', {}), **params}}
                
                # Create manager dengan progress callback
                manager = PreprocessingManager(processing_config, logger)
                manager.register_progress_callback(ui_components['progress_callback'])
                
                # Execute preprocessing
                result = manager.preprocess_with_uuid_consistency(
                    split=params.get('split', 'all'),
                    force_reprocess=params.get('force_reprocess', False)
                )
                
                if result['success']:
                    total = result.get('total_images', 0)
                    time_taken = result.get('processing_time', 0)
                    ui_components.get('complete_operation', lambda x: None)(
                        f"Preprocessing selesai: {total:,} gambar dalam {time_taken:.1f}s"
                    )
                    _update_status_panel(ui_components, f"✅ Preprocessing berhasil: {total:,} gambar", "success")
                else:
                    raise Exception(result['message'])
                    
            except Exception as e:
                error_msg = f"Preprocessing gagal: {str(e)}"
                logger and logger.error(f"💥 {error_msg}")
                ui_components.get('error_operation', lambda x: None)(error_msg)
                _update_status_panel(ui_components, error_msg, "error")
    
    ui_components['preprocess_button'].on_click(execute_preprocessing)

def setup_check_handler(ui_components: Dict[str, Any], config: Dict[str, Any]):
    """Setup dataset checker terintegrasi"""
    
    def execute_check(button=None):
        button_manager = get_button_state_manager(ui_components)
        logger = ui_components.get('logger')
        
        _clear_outputs(ui_components)
        
        with button_manager.operation_context('check'):
            try:
                logger and logger.info("🔍 Checking dataset")
                ui_components.get('show_for_operation', lambda x: None)('check')
                
                checker = DatasetChecker(config, logger)
                
                # Check source
                ui_components.get('update_progress', lambda *a: None)('overall', 30, "Checking source dataset")
                source_result = checker.check_source_dataset(detailed=True)
                
                # Check preprocessed
                ui_components.get('update_progress', lambda *a: None)('overall', 70, "Checking preprocessed dataset")
                preprocessed_result = checker.check_preprocessed_dataset(detailed=True)
                
                # Display results
                _display_check_results(ui_components, source_result, preprocessed_result, logger)
                
                ui_components.get('complete_operation', lambda x: None)("Dataset check selesai")
                
            except Exception as e:
                error_msg = f"Check gagal: {str(e)}"
                logger and logger.error(f"💥 {error_msg}")
                ui_components.get('error_operation', lambda x: None)(error_msg)
    
    ui_components['check_button'].on_click(execute_check)

def setup_cleanup_handler(ui_components: Dict[str, Any], config: Dict[str, Any]):
    """Setup cleanup handler terintegrasi"""
    
    def execute_cleanup(button=None):
        _clear_outputs(ui_components)
        
        def confirmed_cleanup():
            button_manager = get_button_state_manager(ui_components)
            logger = ui_components.get('logger')
            
            with button_manager.operation_context('cleanup'):
                try:
                    logger and logger.info("🧹 Cleanup preprocessed data")
                    ui_components.get('show_for_operation', lambda x: None)('cleanup')
                    
                    executor = CleanupExecutor(config, logger)
                    executor.register_progress_callback(ui_components['progress_callback'])
                    
                    result = executor.cleanup_preprocessed_data(safe_mode=True)
                    
                    if result['success']:
                        stats = result.get('stats', {})
                        files_removed = stats.get('files_removed', 0)
                        ui_components.get('complete_operation', lambda x: None)(
                            f"Cleanup selesai: {files_removed:,} file dihapus"
                        )
                        _update_status_panel(ui_components, f"🧹 Cleanup berhasil: {files_removed:,} file", "success")
                    else:
                        raise Exception(result['message'])
                        
                except Exception as e:
                    error_msg = f"Cleanup gagal: {str(e)}"
                    logger and logger.error(f"💥 {error_msg}")
                    ui_components.get('error_operation', lambda x: None)(error_msg)
        
        # Show confirmation dialog
        if 'confirmation_area' in ui_components:
            from IPython.display import display, clear_output
            with ui_components['confirmation_area']:
                clear_output(wait=True)
                dialog = create_destructive_confirmation(
                    title="⚠️ Konfirmasi Cleanup Dataset",
                    message="Operasi ini akan menghapus SEMUA data preprocessed. Data asli tetap aman.\n\nLanjutkan?",
                    on_confirm=lambda b: (confirmed_cleanup(), _clear_outputs(ui_components)),
                    on_cancel=lambda b: _clear_outputs(ui_components),
                    item_name="data preprocessed"
                )
                display(dialog)
    
    ui_components['cleanup_button'].on_click(execute_cleanup)

def setup_config_handlers(ui_components: Dict[str, Any], config: Dict[str, Any]):
    """Setup config save/reset handlers"""
    config_manager = get_config_manager()
    
    def save_config(button=None):
        try:
            _clear_outputs(ui_components)
            params = _extract_processing_params(ui_components)
            save_success = config_manager.save_config({'preprocessing': params}, 'preprocessing')
            status = "✅ Konfigurasi tersimpan" if save_success else "❌ Gagal simpan konfigurasi"
            _update_status_panel(ui_components, status, "success" if save_success else "error")
        except Exception as e:
            _update_status_panel(ui_components, f"❌ Error: {str(e)}", "error")
    
    def reset_config(button=None):
        try:
            _clear_outputs(ui_components)
            _apply_default_config(ui_components)
            _update_status_panel(ui_components, "🔄 Konfigurasi direset ke default", "info")
        except Exception as e:
            _update_status_panel(ui_components, f"❌ Error: {str(e)}", "error")
    
    ui_components['save_button'].on_click(save_config)
    ui_components['reset_button'].on_click(reset_config)

# Helper functions
def _extract_processing_params(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Extract parameters dari UI components"""
    resolution = getattr(ui_components.get('resolution_dropdown'), 'value', '640x640')
    width, height = map(int, resolution.split('x'))
    
    return {
        'img_size': [width, height],
        'normalize': getattr(ui_components.get('normalization_dropdown'), 'value', 'minmax') != 'none',
        'num_workers': getattr(ui_components.get('worker_slider'), 'value', 4),
        'split': getattr(ui_components.get('split_dropdown'), 'value', 'all'),
        'force_reprocess': False
    }

def _apply_default_config(ui_components: Dict[str, Any]):
    """Apply default config ke UI"""
    if 'resolution_dropdown' in ui_components:
        ui_components['resolution_dropdown'].value = '640x640'
    if 'normalization_dropdown' in ui_components:
        ui_components['normalization_dropdown'].value = 'minmax'
    if 'worker_slider' in ui_components:
        ui_components['worker_slider'].value = 4
    if 'split_dropdown' in ui_components:
        ui_components['split_dropdown'].value = 'all'

def _display_check_results(ui_components: Dict[str, Any], source_result: Dict[str, Any], 
                          preprocessed_result: Dict[str, Any], logger):
    """Display check results"""
    from IPython.display import display, HTML
    
    # Log results
    if source_result.get('valid', False):
        total = source_result.get('total_images', 0)
        logger and logger.success(f"✅ Source dataset: {total:,} gambar valid")
        _update_status_panel(ui_components, f"Dataset siap: {total:,} gambar", "success")
    else:
        msg = source_result.get('message', 'Dataset tidak valid')
        logger and logger.error(f"❌ Source invalid: {msg}")
        _update_status_panel(ui_components, "Dataset tidak valid", "error")
        return
    
    if preprocessed_result.get('valid', False):
        total = preprocessed_result.get('total_processed', 0)
        logger and logger.success(f"💾 Preprocessed: {total:,} gambar")
    else:
        logger and logger.info("ℹ️ Belum ada preprocessed dataset")
    
    # Display reports
    if 'log_output' in ui_components:
        with ui_components['log_output']:
            for result, title in [(source_result, "Source Dataset"), (preprocessed_result, "Preprocessed Dataset")]:
                report = result.get('report', '')
                if report:
                    display(HTML(f"<pre style='background:#f8f9fa;padding:10px;border-radius:5px;font-size:12px;'><strong>{title}:</strong>\n{report}</pre>"))

def _clear_outputs(ui_components: Dict[str, Any]):
    """Clear UI outputs"""
    for key in ['log_output', 'status', 'confirmation_area']:
        widget = ui_components.get(key)
        if widget and hasattr(widget, 'clear_output'):
            widget.clear_output(wait=True)

def _update_status_panel(ui_components: Dict[str, Any], message: str, status_type: str = "info"):
    """Update status panel"""
    from smartcash.ui.components.status_panel import update_status_panel
    if 'status_panel' in ui_components:
        update_status_panel(ui_components['status_panel'], message, status_type)