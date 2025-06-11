"""
File: smartcash/ui/dataset/downloader/handlers/download_handler.py  
Deskripsi: Enhanced download handler dengan cleanup analysis integration dan SRP principle
"""

from typing import Dict, Any
from smartcash.ui.dataset.downloader.utils.ui_utils import log_download_config, display_check_results, show_download_success, clear_outputs, handle_ui_error, show_ui_success
from smartcash.ui.dataset.downloader.utils.button_manager import get_button_manager
from smartcash.ui.dataset.downloader.utils.progress_utils import create_progress_callback
from smartcash.ui.dataset.downloader.utils.backend_utils import check_existing_dataset, create_backend_downloader, get_cleanup_targets, create_backend_cleanup_service

def setup_download_handlers(ui_components: Dict[str, Any], config: Dict[str, Any], env=None) -> Dict[str, Any]:
    """Setup download handlers dengan cleanup behavior integration"""
    ui_components['progress_callback'] = create_progress_callback(ui_components)
    
    # Use enhanced handler dengan cleanup analysis
    from smartcash.ui.dataset.downloader.handlers.download_handler_with_cleanup import setup_download_handler_with_cleanup_analysis
    setup_download_handler_with_cleanup_analysis(ui_components, config)
    
    setup_check_handler(ui_components, config)
    setup_cleanup_handler(ui_components, config)  # Enhanced cleanup
    setup_config_handlers(ui_components)
    
    return ui_components

def setup_check_handler(ui_components: Dict[str, Any], config: Dict[str, Any]):
    """Setup check handler dengan backend scanner integration"""
    
    def execute_check(button=None):
        button_manager = get_button_manager(ui_components)
        
        clear_outputs(ui_components)
        button_manager.disable_buttons('check_button')
        
        try:
            _setup_progress_tracker(ui_components, "Dataset Check")
            
            # Use backend scanner
            from smartcash.dataset.downloader.dataset_scanner import create_dataset_scanner
            scanner = create_dataset_scanner(ui_components.get('logger'))
            scanner.set_progress_callback(ui_components['progress_callback'])
            
            result = scanner.scan_existing_dataset_parallel()
            
            if result.get('status') == 'success':
                display_check_results(ui_components, result)
                show_ui_success(ui_components, "Dataset check selesai", button_manager)
            else:
                handle_ui_error(ui_components, result.get('message', 'Scan failed'), button_manager)
                
        except Exception as e:
            handle_ui_error(ui_components, f"Error check handler: {str(e)}", button_manager)
    
    check_button = ui_components.get('check_button')
    if check_button:
        check_button.on_click(execute_check)

def setup_cleanup_handler(ui_components: Dict[str, Any], config: Dict[str, Any]):
    """Setup enhanced cleanup handler dengan behavior analysis"""
    
    def execute_cleanup(button=None):
        button_manager = get_button_manager(ui_components)
        
        clear_outputs(ui_components)
        button_manager.disable_buttons('cleanup_button')
        
        try:
            logger = ui_components.get('logger')
            
            # Initialize cleanup behavior analyzer
            from smartcash.dataset.downloader.cleanup_behavior import DownloaderCleanupBehavior
            cleanup_analyzer = DownloaderCleanupBehavior(logger)
            
            # Analyze cleanup behavior
            cleanup_analysis = cleanup_analyzer.analyze_current_cleanup_process(config)
            
            if logger:
                safety_measures = len(cleanup_analysis.get('safety_measures', []))
                logger.info(f"🔍 Cleanup analysis: {safety_measures} safety measures found")
            
            # Get cleanup targets
            targets_result = get_cleanup_targets(logger)
            
            if targets_result.get('status') != 'success':
                handle_ui_error(ui_components, "Gagal mendapatkan cleanup targets", button_manager)
                return
            
            summary = targets_result.get('summary', {})
            total_files = summary.get('total_files', 0)
            
            if total_files == 0:
                show_ui_success(ui_components, "Tidak ada file untuk dibersihkan", button_manager)
                return
            
            # Enhanced confirmation dengan analysis
            from smartcash.ui.dataset.downloader.utils.enhanced_confirmation import show_cleanup_confirmation
            show_cleanup_confirmation(
                ui_components, targets_result, cleanup_analysis,
                lambda: _execute_analyzed_cleanup(targets_result, ui_components, button_manager, cleanup_analysis)
            )
            
        except Exception as e:
            handle_ui_error(ui_components, f"Error enhanced cleanup: {str(e)}", button_manager)
    
    cleanup_button = ui_components.get('cleanup_button')
    if cleanup_button:
        cleanup_button.on_click(execute_cleanup)

def setup_config_handlers(ui_components: Dict[str, Any]):
    """Setup save/reset handlers"""
    def save_config_handler(button=None):
        try:
            config_handler = ui_components.get('config_handler')
            if not config_handler:
                handle_ui_error(ui_components, "Config handler tidak tersedia")
                return
            
            success = config_handler.save_config(ui_components)
            if success:
                show_ui_success(ui_components, "✅ Konfigurasi berhasil disimpan")
            else:
                handle_ui_error(ui_components, "❌ Gagal menyimpan konfigurasi")
        except Exception as e:
            handle_ui_error(ui_components, f"❌ Error saat save: {str(e)}")
    
    def reset_config_handler(button=None):
        try:
            config_handler = ui_components.get('config_handler')
            if not config_handler:
                handle_ui_error(ui_components, "Config handler tidak tersedia")
                return
            
            success = config_handler.reset_config(ui_components)
            if success:
                show_ui_success(ui_components, "🔄 Konfigurasi berhasil direset")
            else:
                handle_ui_error(ui_components, "❌ Gagal reset konfigurasi")
        except Exception as e:
            handle_ui_error(ui_components, f"❌ Error saat reset: {str(e)}")
    
    save_button = ui_components.get('save_button')
    reset_button = ui_components.get('reset_button')
    if save_button:
        save_button.on_click(save_config_handler)
    if reset_button:
        reset_button.on_click(reset_config_handler)

def _execute_analyzed_cleanup(targets_result: Dict[str, Any], ui_components: Dict[str, Any], 
                            button_manager, cleanup_analysis: Dict[str, Any]):
    """Execute cleanup dengan analysis monitoring"""
    try:
        logger = ui_components.get('logger')
        
        # Log analysis summary
        if logger:
            safety_measures = len(cleanup_analysis.get('safety_measures', []))
            logger.info(f"🔍 Executing cleanup dengan {safety_measures} safety measures")
        
        _setup_progress_tracker(ui_components, "Enhanced Dataset Cleanup")
        
        # Create cleanup service
        cleanup_service = create_backend_cleanup_service(logger)
        if not cleanup_service:
            handle_ui_error(ui_components, "Gagal membuat cleanup service", button_manager)
            return
        
        # Enhanced progress callback
        enhanced_callback = _create_analysis_aware_progress_callback(ui_components, cleanup_analysis)
        cleanup_service.set_progress_callback(enhanced_callback)
        
        # Execute cleanup
        result = cleanup_service.cleanup_dataset_files(targets_result.get('targets', {}))
        
        if result.get('status') == 'success':
            cleaned_count = len(result.get('cleaned_targets', []))
            success_msg = f"Enhanced cleanup selesai: {cleaned_count} direktori dengan analysis"
            show_ui_success(ui_components, success_msg, button_manager)
            
            if logger:
                logger.success(f"✅ {success_msg}")
        else:
            handle_ui_error(ui_components, result.get('message', 'Enhanced cleanup failed'), button_manager)
            
    except Exception as e:
        handle_ui_error(ui_components, f"Error enhanced cleanup: {str(e)}", button_manager)

def _create_analysis_aware_progress_callback(ui_components: Dict[str, Any], cleanup_analysis: Dict[str, Any]):
    """Create progress callback dengan cleanup analysis context"""
    original_callback = ui_components.get('progress_callback')
    logger = ui_components.get('logger')
    
    def enhanced_callback(step: str, current: int, total: int, message: str):
        if original_callback:
            original_callback(step, current, total, message)
        
        # Enhanced logging dengan analysis context
        if logger and step in ['cleanup', 'analyze', 'validate']:
            analysis_emoji = "🔍" if step == 'analyze' else "🗑️" if step == 'cleanup' else "✅"
            logger.info(f"{analysis_emoji} Analysis Step: {step} - {message}")
    
    return enhanced_callback

def _setup_progress_tracker(ui_components: Dict[str, Any], operation_name: str):
    """Setup progress tracker untuk operation"""
    progress_tracker = ui_components.get('progress_tracker')
    if progress_tracker:
        progress_tracker.show(operation_name)
        progress_tracker.update_overall(0, f"🚀 Memulai {operation_name.lower()}...")
    
    logger = ui_components.get('logger')
    if logger:
        logger.info(f"🚀 Memulai {operation_name.lower()}")