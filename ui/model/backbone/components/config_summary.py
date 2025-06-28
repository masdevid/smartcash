"""
File: smartcash/ui/model/backbone/components/config_summary.py
Deskripsi: Config summary component dengan enhanced visual design
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional, List

def create_config_summary(config: Optional[Dict[str, Any]] = None) -> widgets.VBox:
    """Create config summary display dengan modern styling
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        VBox containing the config summary
    """
    config = config or {}
    
    # Create HTML widget untuk summary display
    summary_html = widgets.HTML(
        value=_generate_summary_html(config),
        layout=widgets.Layout(width='100%')
    )
    
    # Header dengan styling
    header_html = widgets.HTML("""
        <h4 style="
            margin: 15px 0 10px 0;
            color: #2c3e50;
            font-weight: 600;
            border-bottom: 2px solid #27ae60;
            padding-bottom: 5px;
        ">📋 Ringkasan Konfigurasi</h4>
    """)
    
    # === SUMMARY CONTAINER ===
    
    summary = widgets.VBox([
        header_html,
        summary_html
    ], layout=widgets.Layout(
        width='100%',  # Take full width of parent
        padding='15px',
        border='1px solid #e0e0e0',
        border_radius='8px',
        background_color='#f8f9fa',
        box_shadow='0 1px 3px rgba(0,0,0,0.05)',
        margin='0',
        height='100%',
        box_sizing='border-box'
    ))
    
    # Store HTML widget untuk update
    summary.summary_html = summary_html
    
    return summary

def _generate_summary_html(config: Dict[str, Any]) -> str:
    """Generate HTML untuk configuration summary dengan enhanced styling"""
    model_config = config.get('model', {})
    
    # Extract values
    backbone = model_config.get('backbone', 'efficientnet_b4')
    detection_layers = model_config.get('detection_layers', ['banknote'])
    layer_mode = model_config.get('layer_mode', 'single')
    feature_opt = model_config.get('feature_optimization', {}).get('enabled', False)
    mixed_precision = model_config.get('mixed_precision', True)
    
    # Format values
    backbone_display = _format_backbone_name(backbone)
    layers_display = _format_detection_layers(detection_layers)
    mode_display = _format_layer_mode(layer_mode)
    
    # Build HTML dengan enhanced styling
    html = f"""
    <div style="font-size: 14px; line-height: 1.8; color: #2c3e50;">
        <!-- Architecture Section -->
        <div style="
            margin-bottom: 20px;
            padding: 15px;
            background: #ffffff;
            border-radius: 8px;
            border-left: 4px solid #3498db;
        ">
            <strong style="color: #3498db; font-size: 16px;">🏗️ Architecture</strong>
            <div style="margin-top: 8px; margin-left: 10px;">
                <div style="margin: 5px 0;">
                    <span style="color: #7f8c8d;">Backbone:</span>
                    <code style="
                        background: #ecf0f1;
                        padding: 2px 8px;
                        border-radius: 4px;
                        color: #2c3e50;
                        font-weight: 500;
                    ">{backbone_display}</code>
                </div>
                <div style="margin: 5px 0;">
                    <span style="color: #7f8c8d;">Input Size:</span>
                    <code style="
                        background: #ecf0f1;
                        padding: 2px 8px;
                        border-radius: 4px;
                        color: #2c3e50;
                    ">640×640</code>
                </div>
                <div style="margin: 5px 0;">
                    <span style="color: #7f8c8d;">Classes:</span>
                    <code style="
                        background: #ecf0f1;
                        padding: 2px 8px;
                        border-radius: 4px;
                        color: #2c3e50;
                    ">7 denominations</code>
                </div>
            </div>
        </div>
        
        <!-- Detection Section -->
        <div style="
            margin-bottom: 20px;
            padding: 15px;
            background: #ffffff;
            border-radius: 8px;
            border-left: 4px solid #27ae60;
        ">
            <strong style="color: #27ae60; font-size: 16px;">🎯 Detection</strong>
            <div style="margin-top: 8px; margin-left: 10px;">
                <div style="margin: 5px 0;">
                    <span style="color: #7f8c8d;">Layers:</span>
                    {layers_display}
                </div>
                <div style="margin: 5px 0;">
                    <span style="color: #7f8c8d;">Mode:</span>
                    <code style="
                        background: #ecf0f1;
                        padding: 2px 8px;
                        border-radius: 4px;
                        color: #2c3e50;
                    ">{mode_display}</code>
                </div>
            </div>
        </div>
        
        <!-- Optimization Section -->
        <div style="
            margin-bottom: 20px;
            padding: 15px;
            background: #ffffff;
            border-radius: 8px;
            border-left: 4px solid #f39c12;
        ">
            <strong style="color: #f39c12; font-size: 16px;">⚡ Optimization</strong>
            <div style="margin-top: 8px; margin-left: 10px;">
                <div style="margin: 5px 0;">
                    <span style="color: #7f8c8d;">Feature Opt:</span>
                    <span style="
                        background: {'#27ae60' if feature_opt else '#e74c3c'};
                        color: white;
                        padding: 2px 8px;
                        border-radius: 4px;
                        font-size: 12px;
                        font-weight: 500;
                    ">{'Enabled' if feature_opt else 'Disabled'}</span>
                </div>
                <div style="margin: 5px 0;">
                    <span style="color: #7f8c8d;">Mixed Precision:</span>
                    <span style="
                        background: {'#27ae60' if mixed_precision else '#95a5a6'};
                        color: white;
                        padding: 2px 8px;
                        border-radius: 4px;
                        font-size: 12px;
                        font-weight: 500;
                    ">{'FP16' if mixed_precision else 'FP32'}</span>
                </div>
            </div>
        </div>
        
        <!-- Config File Section -->
        <div style="
            padding: 12px;
            background: #ecf0f1;
            border-radius: 8px;
            text-align: center;
        ">
            <span style="color: #7f8c8d;">💾 Config File:</span>
            <code style="
                color: #2c3e50;
                font-weight: 500;
            ">model_config.yaml</code>
        </div>
    </div>
    """
    
    return html

def _format_backbone_name(backbone: str) -> str:
    """Format backbone name untuk display"""
    names = {
        'efficientnet_b4': 'EfficientNet-B4',
        'cspdarknet': 'CSPDarknet'
    }
    return names.get(backbone, backbone.upper())

def _format_detection_layers(layers: List[str]) -> str:
    """Format detection layers dengan badges"""
    layer_colors = {
        'banknote': '#3498db',
        'nominal': '#27ae60',
        'security': '#e74c3c'
    }
    
    badges = []
    for layer in layers:
        color = layer_colors.get(layer, '#95a5a6')
        badge = f"""<span style="
            background: {color};
            color: white;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 500;
            margin-right: 5px;
        ">{layer.capitalize()}</span>"""
        badges.append(badge)
    
    return ''.join(badges) if badges else '<span style="color: #e74c3c;">None selected</span>'

def _format_layer_mode(mode: str) -> str:
    """Format layer mode untuk display"""
    modes = {
        'single': 'Single Layer',
        'multilayer': 'Multi-layer'
    }
    return modes.get(mode, mode.capitalize())

def update_config_summary(summary_component: widgets.VBox, config: Dict[str, Any]) -> None:
    """Update config summary display dengan new configuration
    
    Args:
        summary_component: Summary widget
        config: New configuration dictionary
    """
    if hasattr(summary_component, 'summary_html'):
        summary_component.summary_html.value = _generate_summary_html(config)

def create_config_status_badge(status: str, message: str) -> widgets.HTML:
    """Create status badge untuk config validation
    
    Args:
        status: Status type ('success', 'warning', 'error')
        message: Status message
        
    Returns:
        HTML widget dengan status badge
    """
    colors = {
        'success': '#27ae60',
        'warning': '#f39c12',
        'error': '#e74c3c'
    }
    
    icons = {
        'success': '✅',
        'warning': '⚠️',
        'error': '❌'
    }
    
    color = colors.get(status, '#95a5a6')
    icon = icons.get(status, 'ℹ️')
    
    return widgets.HTML(f"""
        <div style="
            margin: 10px 0;
            padding: 10px;
            background: {color}22;
            border: 1px solid {color};
            border-radius: 6px;
            color: {color};
            font-size: 13px;
        ">
            <strong>{icon} {message}</strong>
        </div>
    """)