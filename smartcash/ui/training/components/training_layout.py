"""
File: smartcash/ui/training/components/training_layout.py
Deskripsi: Fixed responsive training layout dengan refresh button di luar tabs
"""

import ipywidgets as widgets
from typing import Dict, Any

def create_training_layout(form_components: Dict[str, Any]) -> Dict[str, Any]:
    """Create responsive training layout dengan proper refresh button placement"""
    try:
        config = form_components.get('config', {})
        
        # Header dengan dynamic model info
        header = _create_dynamic_header(config)
        
        # Sections dengan enhanced components - refresh button di luar tabs
        sections = [
            _create_config_section_with_external_refresh(form_components),
            _create_control_section(form_components),
            _create_progress_section(form_components),
            _create_metrics_section(form_components),
            _create_log_section(form_components)
        ]
        
        # Filter None sections dan create main container
        valid_sections = [section for section in sections if section]
        main_container = widgets.VBox([
            header,
            *valid_sections,
            _create_footer()
        ], layout=widgets.Layout(width='100%', padding='15px', max_width='1200px'))
        
        # Prevent duplicate display dan update components
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
    
    # Model descriptions dengan one-liner mapping
    model_descriptions = {
        'efficient_basic': f'{backbone.upper()} Basic Detection',
        'efficient_optimized': f'{backbone.upper()} + Feature Optimization',
        'efficient_advanced': f'{backbone.upper()} + Full Optimization Suite',
        'yolov5s': 'YOLOv5s Baseline Model'
    }
    
    model_desc = model_descriptions.get(model_type, f'{backbone.upper()} Custom Model')
    mixed_precision = config.get('training_utils', {}).get('mixed_precision', True)
    layer_mode = config.get('training_utils', {}).get('layer_mode', 'single')
    
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
            <span>🔧 Mixed Precision: {'✅' if mixed_precision else '❌'}</span> • 
            <span>📊 Layer Mode: {layer_mode.title()}</span>
        </div>
    </div>
    """)

def _create_config_section_with_external_refresh(form_components: Dict[str, Any]) -> widgets.VBox:
    """Create config section dengan refresh button di luar tabs"""
    config_tabs = form_components.get('config_tabs')
    refresh_button = form_components.get('refresh_button')
    
    if not config_tabs:
        return None
    
    # Header dengan refresh button di samping
    section_header = widgets.HBox([
        widgets.HTML("<h4 style='margin: 0; color: #333; line-height: 30px;'>ℹ️ YAML Configuration</h4>"),
        widgets.HTML("<div style='flex: 1;'></div>"),  # Spacer
        refresh_button or widgets.HTML("")
    ], layout=widgets.Layout(width='100%', justify_content='space-between', align_items='center', margin='15px 0 10px 0'))
    
    return widgets.VBox([
        section_header,
        config_tabs
    ], layout=widgets.Layout(margin='10px 0'))

def _create_control_section(form_components: Dict[str, Any]) -> widgets.VBox:
    """Create enhanced control section"""
    button_container = form_components.get('button_container')
    status_panel = form_components.get('status_panel')
    
    if not button_container:
        return None
        
    return widgets.VBox([
        widgets.HTML("<h4 style='margin: 15px 0 10px 0; color: #333;'>⚙️ Training Controls</h4>"),
        button_container,
        status_panel or widgets.HTML("")
    ], layout=widgets.Layout(margin='10px 0'))

def _create_progress_section(form_components: Dict[str, Any]) -> widgets.VBox:
    """Create progress section dengan default hidden state"""
    progress_container = form_components.get('progress_container')
    if not progress_container:
        return None
    
    # Set default hidden state dengan one-liner
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
    
    # Create accordion untuk metrics dengan one-liner content creation
    chart_container = widgets.VBox([
        chart_output or widgets.HTML("<div style='text-align: center; padding: 20px; color: #666;'>📈 Chart akan muncul saat training dimulai</div>")
    ], layout=widgets.Layout(width='100%', max_height='400px'))
    
    metrics_container = widgets.VBox([
        metrics_output or widgets.HTML("<div style='text-align: center; padding: 10px; color: #666;'>📊 Metrics akan muncul saat training berjalan</div>")
    ], layout=widgets.Layout(width='100%', max_height='150px'))
    
    # Combined container dengan accordion-like behavior
    from smartcash.ui.components.info_accordion import create_info_accordion
    
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
    """Create footer dengan training info"""
    return widgets.HTML("""
    <div style="text-align: center; padding: 15px; margin-top: 20px; 
                background: #f8f9fa; border-radius: 5px; color: #666; font-size: 12px;">
        <p style="margin: 0;">💰 <b>SmartCash Training Module</b> • 
        Powered by EfficientNet-B4 + YOLOv5 • 
        <span style="color: #007bff;">YAML Config Driven</span></p>
    </div>
    """)

def _create_simple_fallback_layout(form_components: Dict[str, Any], error_msg: str) -> Dict[str, Any]:
    """Simple fallback layout dengan minimal error handling"""
    
    error_display = widgets.HTML(f"""
    <div style="padding: 20px; background: #ffeaea; border-radius: 8px; text-align: center; margin: 10px 0;">
        <h3 style="color: #d32f2f; margin: 0 0 10px 0;">❌ Layout Error</h3>
        <p style="margin: 0; color: #666;">{error_msg}</p>
        <p style="margin: 10px 0 0 0; font-size: 12px; color: #999;">Menggunakan fallback layout minimal</p>
    </div>
    """)
    
    # Simple vertical layout dengan available components
    safe_components = ['button_container', 'status_panel', 'log_output', 'chart_output', 'refresh_button']
    available_components = [error_display] + [form_components[comp] for comp in safe_components if comp in form_components]
    
    main_container = widgets.VBox(available_components, layout=widgets.Layout(width='100%', padding='15px'))
    
    form_components.update({
        'main_container': main_container,
        'ui': main_container,
        'layout_error': error_msg
    })
    
    return form_components

# One-liner utilities untuk layout management
create_section_header = lambda title, icon="📋": widgets.HTML(f"<h4 style='margin: 15px 0 10px 0; color: #333;'>{icon} {title}</h4>")
create_header_with_button = lambda title, button: widgets.HBox([widgets.HTML(f"<h4 style='margin: 0; color: #333; line-height: 30px;'>{title}</h4>"), widgets.HTML("<div style='flex: 1;'></div>"), button], layout=widgets.Layout(width='100%', justify_content='space-between', align_items='center'))
safe_get_component = lambda form_components, key, fallback=None: form_components.get(key, fallback)
set_hidden_state = lambda widget: hasattr(widget, 'layout') and setattr(widget.layout, 'display', 'none')