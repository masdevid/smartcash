"""
File: smartcash/ui/dataset/downloader/utils/enhanced_confirmation.py
Deskripsi: Enhanced confirmation dialogs dengan cleanup behavior analysis
"""

from typing import Dict, Any, Callable
from IPython.display import display, clear_output
import ipywidgets as widgets

def show_enhanced_cleanup_confirmation(ui_components: Dict[str, Any], targets_result: Dict[str, Any],
                                     cleanup_analysis: Dict[str, Any], on_confirm_callback: Callable):
    """Show enhanced cleanup confirmation dengan behavior analysis"""
    confirmation_area = ui_components.get('confirmation_area')
    logger = ui_components.get('logger')
    
    if not confirmation_area:
        if logger:
            logger.warning("⚠️ No confirmation area, executing cleanup directly")
        on_confirm_callback()
        return
    
    try:
        # Force confirmation area visible
        _force_confirmation_area_ready(confirmation_area, logger)
        
        # Build enhanced message dengan cleanup analysis
        message = _build_enhanced_cleanup_message(targets_result, cleanup_analysis)
        
        # Create enhanced confirmation widget
        confirmation_widget = _create_enhanced_cleanup_widget(
            message,
            lambda: _handle_enhanced_confirm(confirmation_area, on_confirm_callback, logger),
            lambda: _handle_enhanced_cancel(confirmation_area, ui_components, logger)
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

def _force_confirmation_area_ready(confirmation_area, logger):
    """Force confirmation area ready untuk display"""
    if not hasattr(confirmation_area, 'layout'):
        if logger:
            logger.warning("⚠️ Confirmation area has no layout attribute")
        return
    
    # Force visibility settings
    layout_updates = {
        'display': 'block', 'visibility': 'visible', 'height': 'auto',
        'min_height': '200px', 'max_height': '600px', 'overflow': 'auto',
        'border': '1px solid #ddd', 'padding': '10px', 'background_color': 'white'
    }
    
    for attr, value in layout_updates.items():
        try:
            setattr(confirmation_area.layout, attr, value)
        except Exception as e:
            if logger:
                logger.debug(f"Could not set {attr}: {str(e)}")

def _build_enhanced_cleanup_message(targets_result: Dict[str, Any], cleanup_analysis: Dict[str, Any]) -> str:
    """Build enhanced cleanup message dengan behavior analysis"""
    summary = targets_result.get('summary', {})
    targets = targets_result.get('targets', {})
    
    total_files = summary.get('total_files', 0)
    size_formatted = summary.get('size_formatted', '0 B')
    
    lines = [
        f"🔍 Enhanced Cleanup Analysis: {total_files:,} file akan dihapus",
        f"📊 Total Size: {size_formatted}",
        "",
        "🔍 Cleanup Behavior Analysis:",
    ]
    
    # Add cleanup phases dari analysis
    cleanup_phases = cleanup_analysis.get('cleanup_phases', [])
    for phase in cleanup_phases[:3]:  # Show first 3 critical phases
        phase_desc = phase.get('description', 'Unknown phase')
        lines.append(f"  📋 {phase_desc}")
    
    lines.extend(["", "📂 Target Cleanup dengan Analysis:"])
    
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

def _create_enhanced_cleanup_widget(message: str, on_confirm: Callable, on_cancel: Callable) -> widgets.VBox:
    """Create enhanced cleanup confirmation widget dengan analysis styling"""
    
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
    
    # Button handlers dengan disable mechanism
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

def _handle_enhanced_confirm(confirmation_area, on_confirm_callback: Callable, logger):
    """Handle enhanced cleanup confirm dengan analysis context"""
    try:
        if logger:
            logger.info("✅ User confirmed enhanced cleanup with analysis")
        
        # Clear confirmation area
        with confirmation_area:
            clear_output(wait=True)
        
        # Execute callback
        on_confirm_callback()
        
    except Exception as e:
        if logger:
            logger.error(f"❌ Error in enhanced cleanup confirm: {str(e)}")

def _handle_enhanced_cancel(confirmation_area, ui_components: Dict[str, Any], logger):
    """Handle enhanced cleanup cancel dengan proper state reset"""
    try:
        if logger:
            logger.info("🚫 User cancelled enhanced cleanup analysis")
        
        # Clear confirmation area
        with confirmation_area:
            clear_output(wait=True)
        
        # Reset button states
        from smartcash.ui.dataset.downloader.utils.button_manager import get_button_manager
        button_manager = get_button_manager(ui_components)
        button_manager.enable_buttons()
        
    except Exception as e:
        if logger:
            logger.error(f"❌ Error in enhanced cleanup cancel: {str(e)}")