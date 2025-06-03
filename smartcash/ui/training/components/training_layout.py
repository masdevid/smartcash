"""
File: smartcash/ui/training/components/training_layout.py
Deskripsi: Enhanced responsive training layout dengan YAML config integration
"""

import ipywidgets as widgets
from typing import Dict, Any

def create_training_layout(form_components: Dict[str, Any]) -> Dict[str, Any]:
    """Create responsive training layout dengan YAML config awareness"""
    try:
        config = form_components.get('config', {})
        
        # Header dengan dynamic model info
        header = _create_dynamic_header(config)
        
        # Sections dengan enhanced components
        sections = [
            _create_config_section(form_components),
            _create_control_section(form_components),
            _create_progress_section(form_components),
            _create_metrics_section(form_components),
            _create_log_section(form_components)
        ]
        
        # Main container dengan responsive layout
        main_container = widgets.VBox([
            header,
            *[section for section in sections if section],  # Filter None sections
            _create_footer()
        ], layout=widgets.Layout(width='100%', padding='15px', max_width='1200px'))
        
        # Prevent duplicate display
        if not getattr(main_container, '_displayed', False):
            main_container._displayed = True
            form_components.update({
                'main_container': main_container,
                'ui': main_container,
                'header': header
            })
        
        return form_components
        
    except Exception as e:
        return _create_simple_fallback_layout(form_components, str(e))

def _create_dynamic_header(config: Dict[str, Any]) -> widgets.HTML:
    """Create dynamic header berdasarkan YAML config"""
    model_config = config.get('model', {})
    model_type = model_config.get('type', 'efficient_basic')
    backbone = model_config.get('backbone', 'efficientnet_b4')
    epochs = config.get('epochs', 100)
    
    # Dynamic title berdasarkan model type
    model_descriptions = {
        'efficient_basic': f'{backbone.upper()} Basic Detection',
        'efficient_optimized': f'{backbone.upper()} + Feature Optimization',
        'efficient_advanced': f'{backbone.upper()} + Full Optimization Suite',
        'yolov5s': 'YOLOv5s Baseline Model'
    }
    
    model_desc = model_descriptions.get(model_type, f'{backbone.upper()} Custom Model')
    
    return widgets.HTML(f"""
    <div style="text-align: center; padding: 20px; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                border-radius: 10px; margin-bottom: 20px; color: white;">
        <h2 style="margin: 0; font-size: 24px; font-weight: bold;">🚀 {model_desc}</h2>
        <p style="margin: 10px 0 0 0; opacity: 0.9; font-size: 14px;">
            Training untuk {epochs} epochs dengan optimasi currency detection
        </p>
        <div style="margin-top: 10px; opacity: 0.8; font-size: 12px;">
            <span>💡 {model_type.replace('_', ' ').title()}</span> • 
            <span>🔧 Mixed Precision: {'✅' if config.get('training_utils', {}).get('mixed_precision', True) else '❌'}</span> • 
            <span>📊 Layer Mode: {config.get('training_utils', {}).get('layer_mode', 'single').title()}</span>
        </div>
    </div>
    """)

def _create_config_section(form_components: Dict[str, Any]) -> widgets.VBox:
    """Create config section dengan YAML integration dan tombol refresh di bawah tabs"""
    from smartcash.ui.training.components.control_buttons import create_refresh_config_button
    
    config_tabs = form_components.get('config_tabs')
    if not config_tabs:
        return None
    
    # Buat refresh button
    refresh_components = create_refresh_config_button()
    refresh_button = refresh_components['refresh_button']
    
    # Styling untuk refresh button
    refresh_container = widgets.HBox(
        [widgets.HTML('<span style="font-size: 12px; color: #666;">Refresh dari YAML:</span>'), refresh_button],
        layout=widgets.Layout(justify_content='flex-end', width='100%', margin='5px 0')
    )
    
    # Simpan referensi refresh button untuk handler
    form_components['refresh_button'] = refresh_button
    
    return widgets.VBox([
        widgets.HTML("<h4 style='margin: 15px 0 10px 0; color: #333;'>ℹ️ YAML Configuration</h4>"),
        config_tabs,
        refresh_container
    ], layout=widgets.Layout(margin='10px 0'))

def _create_control_section(form_components: Dict[str, Any]) -> widgets.VBox:
    """Create enhanced control section"""
    button_container = form_components.get('button_container')
    if not button_container:
        return None
        
    return widgets.VBox([
        widgets.HTML("<h4 style='margin: 15px 0 10px 0; color: #333;'>⚙️ Training Controls</h4>"),
        button_container,
        form_components.get('status_panel', widgets.HTML(""))
    ], layout=widgets.Layout(margin='10px 0'))

def _create_progress_section(form_components: Dict[str, Any]) -> widgets.VBox:
    """Create progress section dengan default hidden state"""
    progress_container = form_components.get('progress_container')
    if not progress_container:
        return None
    
    # Set default hidden untuk progress
    hasattr(progress_container, 'layout') and setattr(progress_container.layout, 'display', 'none')
    
    return widgets.VBox([
        widgets.HTML("<h4 style='margin: 15px 0 10px 0; color: #333;'>📊 Training Progress</h4>"),
        progress_container
    ], layout=widgets.Layout(margin='10px 0'))

def _create_metrics_section(form_components: Dict[str, Any]) -> widgets.VBox:
    """Create metrics section dengan responsive chart layout"""
    chart_output = form_components.get('chart_output')
    metrics_output = form_components.get('metrics_output')
    
    if not chart_output and not metrics_output:
        return None
    
    # Create accordion untuk metrics
    from smartcash.ui.components.info_accordion import create_info_accordion
    
    # Chart container dengan proper sizing
    chart_container = widgets.VBox([
        chart_output or widgets.HTML("Chart akan muncul saat training dimulai")
    ], layout=widgets.Layout(width='100%', max_height='400px'))
    
    # Metrics container
    metrics_container = widgets.VBox([
        metrics_output or widgets.HTML("Metrics akan muncul saat training berjalan")
    ], layout=widgets.Layout(width='100%', max_height='150px'))
    
    # Combined container
    metrics_accordion = create_info_accordion(
        title="Training Metrics & Visualization",
        content=widgets.VBox([chart_container, metrics_container]),
        icon="chart",
        open_by_default=False
    )
    
    return widgets.VBox([
        metrics_accordion['container']
    ], layout=widgets.Layout(margin='10px 0'))

def _create_log_section(form_components: Dict[str, Any]) -> widgets.VBox:
    """Create log section dengan proper accordion"""
    log_accordion = form_components.get('log_accordion')
    log_output = form_components.get('log_output')
    
    if not log_accordion and not log_output:
        return None
        
    return widgets.VBox([
        widgets.HTML("<h4 style='margin: 15px 0 10px 0; color: #333;'>📋 Training Logs</h4>"),
        log_accordion or log_output
    ], layout=widgets.Layout(margin='10px 0'))

def _create_footer() -> widgets.HTML:
    """Create simple footer dengan training info"""
    return widgets.HTML("""
    <div style="text-align: center; padding: 15px; margin-top: 20px; 
                background: #f8f9fa; border-radius: 5px; color: #666; font-size: 12px;">
        <p style="margin: 0;">💰 <b>SmartCash Training Module</b> • 
        Powered by EfficientNet-B4 + YOLOv5 • 
        <span style="color: #007bff;">YAML Config Driven</span></p>
    </div>
    """)

def _create_simple_fallback_layout(form_components: Dict[str, Any], error_msg: str) -> Dict[str, Any]:
    """Simple fallback layout tanpa excessive error handling"""
    
    error_display = widgets.HTML(f"""
    <div style="padding: 20px; background: #ffeaea; border-radius: 8px; text-align: center; margin: 10px 0;">
        <h3 style="color: #d32f2f; margin: 0 0 10px 0;">❌ Layout Error</h3>
        <p style="margin: 0; color: #666;">{error_msg}</p>
    </div>
    """)
    
    # Simple vertical layout dengan available components
    available_components = [error_display]
    safe_components = ['button_container', 'status_panel', 'log_output', 'chart_output']
    [available_components.append(form_components[comp]) for comp in safe_components if comp in form_components]
    
    main_container = widgets.VBox(available_components, layout=widgets.Layout(width='100%', padding='15px'))
    
    form_components.update({
        'main_container': main_container,
        'ui': main_container,
        'layout_error': error_msg
    })
    
    return form_components

# One-liner utilities untuk layout management
create_section_divider = lambda: widgets.HTML('<div style="height: 1px; background: linear-gradient(to right, transparent, #ddd, transparent); margin: 15px 0;"></div>')
get_responsive_width = lambda: "100%"
create_section_header = lambda title, icon="📋": widgets.HTML(f"<h4 style='margin: 15px 0 10px 0; color: #333;'>{icon} {title}</h4>")
safe_get_layout_component = lambda form_components, key: form_components.get(key) if form_components.get(key) else None