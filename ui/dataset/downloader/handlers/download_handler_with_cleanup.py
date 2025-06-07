"""
File: smartcash/ui/dataset/downloader/handlers/download_handler_with_cleanup.py
Deskripsi: Integrasi cleanup behavior analyzer dalam download flow
"""

from typing import Dict, Any
from smartcash.ui.dataset.downloader.utils.confirmation_dialog import show_downloader_confirmation_dialog
from smartcash.ui.dataset.downloader.utils.backend_utils import check_existing_dataset, create_backend_downloader
from smartcash.dataset.downloader.cleanup_behavior import DownloaderCleanupBehavior

def setup_download_handler_with_cleanup_analysis(ui_components: Dict[str, Any], config: Dict[str, Any]):
    """Setup download handler dengan integrated cleanup behavior analysis"""
    
    def execute_download_with_analysis(button=None):
        from smartcash.ui.dataset.downloader.utils.button_manager import get_button_manager
        from smartcash.ui.dataset.downloader.utils.ui_utils import clear_outputs, handle_ui_error
        
        button_manager = get_button_manager(ui_components)
        clear_outputs(ui_components)
        button_manager.disable_buttons('download_button')
        
        try:
            # 1. Validate config
            config_handler = ui_components.get('config_handler')
            ui_config = config_handler.extract_config(ui_components)
            validation = config_handler.validate_config(ui_config)
            
            if not validation['valid']:
                handle_ui_error(ui_components, f"Config invalid: {', '.join(validation['errors'])}", button_manager)
                return
            
            # 2. Analyze cleanup behavior
            cleanup_analyzer = DownloaderCleanupBehavior(ui_components.get('logger'))
            cleanup_analysis = cleanup_analyzer.analyze_current_cleanup_process(ui_config)
            
            # 3. Check existing data dengan cleanup context
            has_content, total_images, summary_data = check_existing_dataset(ui_components.get('logger'))
            
            if has_content:
                # 4. Implement safe cleanup procedure
                success = cleanup_analyzer.implement_safe_cleanup_procedure(ui_config, ui_components)
                if not success:
                    handle_ui_error(ui_components, "Gagal setup safe cleanup procedure", button_manager)
                    return
                
                # 5. Show enhanced confirmation dengan cleanup analysis
                _show_enhanced_confirmation_with_analysis(
                    ui_components, total_images, ui_config, cleanup_analysis,
                    lambda btn: _execute_analyzed_download(ui_config, ui_components, button_manager, cleanup_analysis)
                )
            else:
                # No existing data, proceed directly
                _execute_analyzed_download(ui_config, ui_components, button_manager, cleanup_analysis)
                
        except Exception as e:
            handle_ui_error(ui_components, f"Error download analysis: {str(e)}", button_manager)
    
    download_button = ui_components.get('download_button')
    if download_button:
        download_button.on_click(execute_download_with_analysis)

def _show_enhanced_confirmation_with_analysis(ui_components: Dict[str, Any], existing_count: int, 
                                            config: Dict[str, Any], cleanup_analysis: Dict[str, Any],
                                            on_confirm):
    """Show confirmation dengan enhanced cleanup analysis info"""
    confirmation_area = ui_components.get('confirmation_area')
    if not confirmation_area:
        on_confirm(None)  # Skip confirmation jika tidak ada UI
        return
    
    # Build enhanced message dengan cleanup analysis
    message = _build_enhanced_confirmation_message(existing_count, config, cleanup_analysis)
    
    from IPython.display import display, clear_output
    from smartcash.ui.components.dialogs import show_destructive_confirmation
    
    def safe_confirm(btn):
        with confirmation_area:
            clear_output(wait=True)
        on_confirm(btn)
    
    def safe_cancel(btn):
        with confirmation_area:
            clear_output(wait=True)
        _handle_download_cancellation(ui_components)
    
    # Force confirmation area visible
    if hasattr(confirmation_area, 'layout'):
        confirmation_area.layout.display = 'block'
        confirmation_area.layout.visibility = 'visible'
        confirmation_area.layout.height = 'auto'
    
    # Display enhanced dialog
    with confirmation_area:
        clear_output(wait=True)
        dialog = show_destructive_confirmation(
            "⚠️ Konfirmasi Download dengan Cleanup Analysis",
            message,
            "dataset existing",
            safe_confirm,
            safe_cancel
        )
        display(dialog)

def _build_enhanced_confirmation_message(existing_count: int, config: Dict[str, Any], 
                                       cleanup_analysis: Dict[str, Any]) -> str:
    """Build message dengan cleanup behavior analysis"""
    roboflow = config.get('data', {}).get('roboflow', {})
    download = config.get('download', {})
    backup_enabled = download.get('backup_existing', False)
    
    # Safety assessment
    safety_level = "🔒 AMAN" if backup_enabled else "🚨 BERBAHAYA"
    
    lines = [
        f"🗂️ Dataset Existing: {existing_count:,} file akan dihapus",
        f"🎯 Target: {roboflow.get('workspace')}/{roboflow.get('project')}:v{roboflow.get('version')}",
        f"🛡️ Safety Level: {safety_level}",
        "",
        "📋 Cleanup Behavior Analysis:",
    ]
    
    # Add cleanup phases dari analysis
    for phase in cleanup_analysis.get('cleanup_phases', [])[:3]:  # Show first 3 critical phases
        risk_emoji = "🚨" if "cleanup" in phase['phase'] else "ℹ️"
        lines.append(f"  {risk_emoji} {phase['description']}")
    
    lines.extend([
        "",
        f"💾 Backup: {'✅ AKTIF' if backup_enabled else '❌ TIDAK AKTIF'}",
        f"🔄 UUID Rename: {'✅' if download.get('rename_files', True) else '❌'}",
        f"✅ Validasi: {'✅' if download.get('validate_download', True) else '❌'}",
        "",
    ])
    
    # Add safety recommendations
    recommendations = cleanup_analysis.get('recommendations', [])
    if recommendations and not backup_enabled:
        lines.extend([
            "⚠️ REKOMENDASI KEAMANAN:",
            f"  • {recommendations[0]}",  # First recommendation
            ""
        ])
    
    # Final warning
    if backup_enabled:
        lines.append("✅ Data existing akan dibackup sebelum dihapus")
    else:
        lines.append("🚨 DATA EXISTING AKAN HILANG PERMANEN!")
    
    return '\n'.join(lines)

def _execute_analyzed_download(ui_config: Dict[str, Any], ui_components: Dict[str, Any], 
                             button_manager, cleanup_analysis: Dict[str, Any]):
    """Execute download dengan cleanup analysis monitoring"""
    try:
        logger = ui_components.get('logger')
        
        # Log cleanup analysis
        _log_cleanup_analysis(logger, cleanup_analysis)
        
        # Create downloader
        downloader = create_backend_downloader(ui_config, logger)
        if not downloader:
            from smartcash.ui.dataset.downloader.utils.ui_utils import handle_ui_error
            handle_ui_error(ui_components, "Gagal membuat download service", button_manager)
            return
        
        # Setup progress dengan cleanup context
        if hasattr(downloader, 'set_progress_callback'):
            enhanced_callback = _create_cleanup_aware_progress_callback(ui_components, cleanup_analysis)
            downloader.set_progress_callback(enhanced_callback)
        
        # Execute download
        from smartcash.ui.dataset.downloader.utils.ui_utils import log_download_config
        log_download_config(ui_components, ui_config)
        
        if logger:
            logger.info("🚀 Memulai download dengan cleanup analysis")
        
        result = downloader.download_dataset()
        
        # Handle result dengan cleanup context
        _handle_download_result_with_cleanup(result, ui_components, button_manager, cleanup_analysis)
        
    except Exception as e:
        from smartcash.ui.dataset.downloader.utils.ui_utils import handle_ui_error
        handle_ui_error(ui_components, f"Error analyzed download: {str(e)}", button_manager)

def _log_cleanup_analysis(logger, cleanup_analysis: Dict[str, Any]):
    """Log cleanup analysis summary"""
    if not logger:
        return
    
    safety_measures = cleanup_analysis.get('safety_measures', [])
    backup_behavior = cleanup_analysis.get('backup_behavior', {})
    
    logger.info(f"🔍 Cleanup Analysis: {len(safety_measures)} safety measures aktif")
    logger.info(f"💾 Backup Status: {backup_behavior.get('backup_enabled', False)}")

def _create_cleanup_aware_progress_callback(ui_components: Dict[str, Any], cleanup_analysis: Dict[str, Any]):
    """Create progress callback yang aware dengan cleanup phases"""
    original_callback = ui_components.get('progress_callback')
    logger = ui_components.get('logger')
    
    def enhanced_callback(step: str, current: int, total: int, message: str):
        # Call original callback
        if original_callback:
            original_callback(step, current, total, message)
        
        # Enhanced logging untuk cleanup phases
        if logger and step in ['organize', 'cleanup', 'backup']:
            cleanup_emoji = "🗑️" if step == 'cleanup' else "📦" if step == 'organize' else "💾"
            logger.info(f"{cleanup_emoji} Cleanup Phase: {step} - {message}")
    
    return enhanced_callback

def _handle_download_result_with_cleanup(result: Dict[str, Any], ui_components: Dict[str, Any], 
                                       button_manager, cleanup_analysis: Dict[str, Any]):
    """Handle download result dengan cleanup analysis context"""
    from smartcash.ui.dataset.downloader.utils.ui_utils import show_download_success, handle_ui_error, show_ui_success
    
    if result and result.get('status') == 'success':
        # Enhanced success message dengan cleanup info
        stats = result.get('stats', {})
        logger = ui_components.get('logger')
        
        if logger:
            backup_behavior = cleanup_analysis.get('backup_behavior', {})
            if backup_behavior.get('backup_enabled'):
                logger.success("✅ Download selesai dengan backup data lama tersimpan")
            else:
                logger.success("✅ Download selesai (data lama telah dihapus)")
        
        show_download_success(ui_components, result)
        show_ui_success(ui_components, "Download berhasil dengan cleanup analysis", button_manager)
    else:
        error_msg = result.get('message', 'Download gagal') if result else 'No response from service'
        handle_ui_error(ui_components, error_msg, button_manager)

def _handle_download_cancellation(ui_components: Dict[str, Any]):
    """Handle download cancellation"""
    logger = ui_components.get('logger')
    if logger:
        logger.info("🚫 Download dibatalkan setelah cleanup analysis")
    

    confirmation_area = ui_components.get('confirmation_area')
    if confirmation_area and hasattr(confirmation_area, 'layout'):
        confirmation_area.layout.height = '0px'
    from smartcash.ui.dataset.downloader.utils.button_manager import get_button_manager
    button_manager = get_button_manager(ui_components)
    button_manager.enable_buttons()