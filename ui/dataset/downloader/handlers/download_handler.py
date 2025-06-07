"""
File: smartcash/ui/dataset/downloader/handlers/download_handler.py  
Deskripsi: Enable enhanced handler dengan cleanup analysis integration
"""

def setup_download_handlers(ui_components: Dict[str, Any], config: Dict[str, Any], env=None) -> Dict[str, Any]:
    """Setup download handlers dengan cleanup behavior integration - ENABLED"""
    ui_components['progress_callback'] = create_progress_callback(ui_components)
    
    # CHANGED: Use enhanced handler dengan cleanup analysis (was commented out)
    from smartcash.ui.dataset.downloader.handlers.download_handler_with_cleanup import setup_download_handler_with_cleanup_analysis
    setup_download_handler_with_cleanup_analysis(ui_components, config)
    
    setup_check_handler(ui_components, config)
    setup_cleanup_handler_with_enhanced_analysis(ui_components, config)  # Enhanced cleanup too
    setup_config_handlers(ui_components)
    
    return ui_components

def setup_cleanup_handler_with_enhanced_analysis(ui_components: Dict[str, Any], config: Dict[str, Any]):
    """Setup cleanup handler dengan enhanced analysis (NEW)"""
    
    def execute_cleanup_with_analysis(button=None):
        button_manager = get_button_manager(ui_components)
        
        clear_outputs(ui_components)
        button_manager.disable_buttons('cleanup_button')
        
        try:
            logger = ui_components.get('logger')
            
            # 1. Initialize cleanup behavior analyzer
            from smartcash.dataset.downloader.cleanup_behavior import DownloaderCleanupBehavior
            cleanup_analyzer = DownloaderCleanupBehavior(logger)
            
            # 2. Analyze cleanup behavior
            cleanup_analysis = cleanup_analyzer.analyze_current_cleanup_process(config)
            
            if logger:
                logger.info("🔍 Cleanup behavior analysis completed")
                safety_measures = len(cleanup_analysis.get('safety_measures', []))
                logger.info(f"🛡️ Found {safety_measures} safety measures")
            
            # 3. Get cleanup targets dengan analysis context
            targets_result = get_cleanup_targets(logger)
            
            if targets_result.get('status') != 'success':
                handle_ui_error(ui_components, "Gagal mendapatkan cleanup targets", button_manager)
                return
            
            summary = targets_result.get('summary', {})
            total_files = summary.get('total_files', 0)
            
            if total_files == 0:
                show_ui_success(ui_components, "Tidak ada file untuk dibersihkan", button_manager)
                return
            
            # 4. Enhanced confirmation dengan cleanup analysis
            _show_enhanced_cleanup_confirmation_with_analysis(
                ui_components, targets_result, cleanup_analysis,
                lambda: _execute_analyzed_cleanup(targets_result, ui_components, button_manager, cleanup_analysis)
            )
            
        except Exception as e:
            handle_ui_error(ui_components, f"Error cleanup analysis: {str(e)}", button_manager)
    
    cleanup_button = ui_components.get('cleanup_button')
    if cleanup_button:
        cleanup_button.on_click(execute_cleanup_with_analysis)

def _show_enhanced_cleanup_confirmation_with_analysis(ui_components: Dict[str, Any], 
                                                    targets_result: Dict[str, Any],
                                                    cleanup_analysis: Dict[str, Any],
                                                    on_confirm_callback):
    """Show enhanced cleanup confirmation dengan behavior analysis"""
    from IPython.display import display, clear_output
    
    confirmation_area = ui_components.get('confirmation_area')
    logger = ui_components.get('logger')
    
    if not confirmation_area:
        if logger:
            logger.warning("⚠️ No confirmation area, executing cleanup directly")
        on_confirm_callback()
        return
    
    try:
        # Force confirmation area visible
        if hasattr(confirmation_area, 'layout'):
            confirmation_area.layout.display = 'block'
            confirmation_area.layout.visibility = 'visible'
            confirmation_area.layout.height = 'auto'
            confirmation_area.layout.min_height = '200px'
            confirmation_area.layout.max_height = '600px'
        
        # Build enhanced message dengan cleanup analysis
        message = _build_enhanced_cleanup_message_with_analysis(targets_result, cleanup_analysis)
        
        # Create enhanced confirmation widget
        confirmation_widget = _create_enhanced_cleanup_confirmation_widget(
            message,
            lambda: _handle_enhanced_cleanup_confirm(confirmation_area, on_confirm_callback, logger),
            lambda: _handle_enhanced_cleanup_cancel(confirmation_area, ui_components, logger)
        )
        
        # Display dengan force update
        with confirmation_area:
            clear_output(wait=True)
            display(confirmation_widget)
        
        if logger:
            logger.info("✅ Enhanced cleanup confirmation with analysis displayed")
            
    except Exception as e:
        if logger:
            logger.error(f"❌ Error showing enhanced cleanup confirmation: {str(e)}")
        on_confirm_callback()

def _build_enhanced_cleanup_message_with_analysis(targets_result: Dict[str, Any], 
                                                cleanup_analysis: Dict[str, Any]) -> str:
    """Build enhanced cleanup message dengan behavior analysis"""
    summary = targets_result.get('summary', {})
    targets = targets_result.get('targets', {})
    
    total_files = summary.get('total_files', 0)
    size_formatted = summary.get('size_formatted', '0 B')
    
    lines = [
        f"🗑️ Enhanced Cleanup Analysis: {total_files:,} file akan dihapus",
        f"📊 Total Size: {size_formatted}",
        "",
        "🔍 Cleanup Behavior Analysis:",
    ]
    
    # Add cleanup phases dari analysis
    cleanup_phases = cleanup_analysis.get('cleanup_phases', [])
    for phase in cleanup_phases[:3]:  # Show first 3 phases
        phase_desc = phase.get('description', 'Unknown phase')
        lines.append(f"  📋 {phase_desc}")
    
    lines.extend([
        "",
        "📂 Target Cleanup dengan Analysis:",
    ])
    
    # Add target details dengan safety assessment
    safety_measures = cleanup_analysis.get('safety_measures', [])
    safety_count = len(safety_measures)
    
    for target_name, target_info in targets.items():
        file_count = target_info.get('file_count', 0)
        target_size = target_info.get('size_formatted', '0 B')
        lines.append(f"  • {target_name}: {file_count:,} file ({target_size})")
    
    lines.extend([
        "",
        f"🛡️ Safety Measures Active: {safety_count}/4",
        "🔒 Backup Behavior: Analysis completed",
        "",
        "⚠️ ENHANCED WARNING: Cleanup analysis menunjukkan:",
    ])
    
    # Add recommendations dari analysis
    recommendations = cleanup_analysis.get('recommendations', [])
    if recommendations:
        lines.append(f"  💡 {recommendations[0]}")  # Show first recommendation
    
    lines.append("  🚨 File yang dihapus TIDAK BISA di-recovery!")
    
    return '\n'.join(lines)

def _create_enhanced_cleanup_confirmation_widget(message: str, on_confirm, on_cancel):
    """Create enhanced cleanup confirmation widget"""
    import ipywidgets as widgets
    
    # Enhanced title dengan analysis indicator
    title = widgets.HTML(
        '<h3 style="color: #dc3545; margin: 0 0 15px 0;">🔍 Enhanced Cleanup dengan Behavior Analysis</h3>',
        layout=widgets.Layout(margin='0 0 10px 0')
    )
    
    # Enhanced message dengan analysis styling
    message_widget = widgets.HTML(
        f'<div style="background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%); padding: 15px; border-radius: 5px; margin: 10px 0; border-left: 4px solid #e17055;"><pre style="white-space: pre-wrap; margin: 0; font-family: monospace; font-size: 12px; color: #2d3436;">{message}</pre></div>',
        layout=widgets.Layout(margin='10px 0')
    )
    
    # Enhanced buttons
    confirm_button = widgets.Button(
        description="Ya, Analisis Selesai - Hapus",
        button_style='danger',
        icon='trash',
        layout=widgets.Layout(width='220px', height='35px', margin='0 10px 0 0')
    )
    
    cancel_button = widgets.Button(
        description="Batal",
        button_style='',
        icon='times', 
        layout=widgets.Layout(width='100px', height='35px')
    )
    
    # Button handlers
    def safe_confirm(btn):
        btn.disabled = True
        cancel_button.disabled = True
        btn.description = "Executing Analysis..."
        on_confirm()
    
    def safe_cancel(btn):
        btn.disabled = True
        confirm_button.disabled = True
        on_cancel()
    
    confirm_button.on_click(safe_confirm)
    cancel_button.on_click(safe_cancel)
    
    # Button container
    button_container = widgets.HBox(
        [confirm_button, cancel_button],
        layout=widgets.Layout(justify_content='flex-end', margin='15px 0 0 0')
    )
    
    # Enhanced main container dengan gradient border
    return widgets.VBox([
        title,
        message_widget,
        button_container
    ], layout=widgets.Layout(
        width='100%',
        padding='20px',
        border='2px solid #e17055',
        border_radius='8px',
        background_color='#fff'
    ))

def _execute_analyzed_cleanup(targets_result: Dict[str, Any], ui_components: Dict[str, Any], 
                            button_manager, cleanup_analysis: Dict[str, Any]):
    """Execute cleanup dengan analysis monitoring"""
    try:
        logger = ui_components.get('logger')
        
        # Log cleanup analysis summary
        if logger:
            safety_measures = len(cleanup_analysis.get('safety_measures', []))
            logger.info(f"🔍 Executing cleanup dengan {safety_measures} safety measures")
        
        # Setup progress tracker
        _setup_progress_tracker(ui_components, "Enhanced Dataset Cleanup")
        
        # Create cleanup service
        cleanup_service = create_backend_cleanup_service(logger)
        if not cleanup_service:
            handle_ui_error(ui_components, "Gagal membuat cleanup service", button_manager)
            return
        
        # Enhanced progress callback dengan analysis context
        enhanced_callback = _create_analysis_aware_progress_callback(ui_components, cleanup_analysis)
        cleanup_service.set_progress_callback(enhanced_callback)
        
        # Execute cleanup
        result = cleanup_service.cleanup_dataset_files(targets_result.get('targets', {}))
        
        # Enhanced result handling
        if result.get('status') == 'success':
            cleaned_count = len(result.get('cleaned_targets', []))
            success_msg = f"Enhanced cleanup selesai: {cleaned_count} direktori dibersihkan dengan analysis"
            show_ui_success(ui_components, success_msg, button_manager)
            
            if logger:
                logger.success(f"✅ {success_msg}")
        else:
            handle_ui_error(ui_components, result.get('message', 'Enhanced cleanup failed'), button_manager)
            
    except Exception as e:
        handle_ui_error(ui_components, f"Error enhanced cleanup: {str(e)}", button_manager)

def _create_analysis_aware_progress_callback(ui_components: Dict[str, Any], cleanup_analysis: Dict[str, Any]):
    """Create progress callback yang aware dengan cleanup analysis"""
    original_callback = ui_components.get('progress_callback')
    logger = ui_components.get('logger')
    
    def enhanced_callback(step: str, current: int, total: int, message: str):
        # Call original callback
        if original_callback:
            original_callback(step, current, total, message)
        
        # Enhanced logging dengan analysis context
        if logger and step in ['cleanup', 'analyze', 'validate']:
            analysis_emoji = "🔍" if step == 'analyze' else "🗑️" if step == 'cleanup' else "✅"
            logger.info(f"{analysis_emoji} Analysis Step: {step} - {message}")
    
    return enhanced_callback

def _handle_enhanced_cleanup_confirm(confirmation_area, on_confirm_callback, logger):
    """Handle enhanced cleanup confirm"""
    try:
        if logger:
            logger.info("✅ User confirmed enhanced cleanup with analysis")
        
        from IPython.display import clear_output
        with confirmation_area:
            clear_output(wait=True)
        
        on_confirm_callback()
        
    except Exception as e:
        if logger:
            logger.error(f"❌ Error in enhanced cleanup confirm: {str(e)}")

def _handle_enhanced_cleanup_cancel(confirmation_area, ui_components: Dict[str, Any], logger):
    """Handle enhanced cleanup cancel"""
    try:
        if logger:
            logger.info("🚫 User cancelled enhanced cleanup analysis")
        
        from IPython.display import clear_output
        with confirmation_area:
            clear_output(wait=True)
        
        button_manager = get_button_manager(ui_components)
        button_manager.enable_buttons()
        
    except Exception as e:
        if logger:
            logger.error(f"❌ Error in enhanced cleanup cancel: {str(e)}")