"""
File: smartcash/ui/backbone/components/ui_layout.py
Deskripsi: Layout responsif dengan model selection + features di kiri dan backbone summary di kanan
"""

from typing import Dict, Any
import ipywidgets as widgets
from smartcash.ui.utils.header_utils import create_header
from smartcash.ui.utils.layout_utils import create_responsive_container, create_responsive_two_column

def create_backbone_layout(form_components: Dict[str, Any]) -> Dict[str, Any]:
    """Create layout dengan model selection + features di kiri, backbone summary di kanan"""
    
    # Header dengan icon dan description
    header = create_header(
        title="Konfigurasi Backbone Model",
        description="Pilih arsitektur backbone dan optimasi untuk deteksi denominasi mata uang Rupiah",
        icon="🧠"
    )
    
    # Left column: Model Selection + Features
    model_selection = create_responsive_container([
        widgets.HTML("<h4 style='color: #2c3e50; margin: 0 0 15px 0;'>🔧 Seleksi Model</h4>"),
        form_components['backbone_dropdown'],
        form_components['model_type_dropdown'],
        widgets.HTML("<hr style='margin: 15px 0; border: 1px dashed #ddd;'>"),
        widgets.HTML("<h4 style='color: #2c3e50; margin: 0 0 15px 0;'>⚡ Optimasi Fitur</h4>"),
        form_components['use_attention_checkbox'],
        form_components['use_residual_checkbox'],
        form_components['use_ciou_checkbox']
    ], padding="15px", margin="5px")
    
    # Right column: Backbone Summary
    backbone_summary = create_responsive_container([
        widgets.HTML("<h4 style='color: #2c3e50; margin: 0 0 15px 0;'>📊 Ringkasan Backbone</h4>"),
        _create_backbone_summary_widget(form_components)
    ], padding="15px", margin="5px")
    
    # Two-column layout
    main_content = create_responsive_two_column(
        model_selection,
        backbone_summary,
        left_width="48%",
        right_width="48%",
        gap="4%"
    )
    
    # Controls section
    controls_section = create_responsive_container([
        form_components['save_reset_container']
    ], container_type="hbox", justify_content="flex-end", margin="15px 0")
    
    # Main container
    main_container = create_responsive_container([
        header,
        form_components['status_panel'],
        widgets.HTML("<hr style='margin: 15px 0; border: 1px dashed #ddd;'>"),
        main_content,
        widgets.HTML("<hr style='margin: 15px 0; border: 1px dashed #ddd;'>"),
        controls_section
    ], padding="20px", width="100%")
    
    return {
        'main_container': main_container,
        'ui': main_container,
        'header': header,
        'model_selection': model_selection,
        'backbone_summary': backbone_summary,
        'main_content': main_content,
        'controls_section': controls_section,
        **form_components
    }

def _create_backbone_summary_widget(form_components: Dict[str, Any]) -> widgets.HTML:
    """Create dynamic backbone summary widget"""
    summary_widget = widgets.HTML()
    _update_summary_content(form_components, summary_widget)
    _setup_summary_auto_update(form_components, summary_widget)
    return summary_widget

def _update_summary_content(form_components: Dict[str, Any], summary_widget: widgets.HTML) -> None:
    """Update summary content dengan current form values"""
    # Get current values dengan safe access
    backbone_val = getattr(form_components.get('backbone_dropdown'), 'value', 'efficientnet_b4')
    model_type_val = getattr(form_components.get('model_type_dropdown'), 'value', 'efficient_optimized')
    use_attention = getattr(form_components.get('use_attention_checkbox'), 'value', False)
    use_residual = getattr(form_components.get('use_residual_checkbox'), 'value', False)
    use_ciou = getattr(form_components.get('use_ciou_checkbox'), 'value', False)
    
    # Get config info
    from ..handlers.defaults import get_default_backbone_config
    config = get_default_backbone_config()
    backbone_info = config['backbones'].get(backbone_val, {})
    model_info = config['model_types'].get(model_type_val, {})
    
    # Feature status icons
    features = [
        ("FeatureAdapter", use_attention),
        ("ResidualAdapter", use_residual), 
        ("CIoU Loss", use_ciou)
    ]
    
    features_html = ''.join([
        f"{'✅' if active else '❌'} <span style='color: {'#28a745' if active else '#6c757d'}'>{name}</span><br>"
        for name, active in features
    ])
    
    summary_html = f"""
    <div style='background: #f8f9fa; padding: 12px; border-radius: 8px; border-left: 4px solid #007bff;'>
        <div style='margin-bottom: 10px;'>
            <strong style='color: #2c3e50;'>🧠 {backbone_val.upper()}</strong>
            <br><small style='color: #6c757d;'>{backbone_info.get('description', 'No description')[:80]}...</small>
        </div>
        
        <div style='margin-bottom: 10px;'>
            <strong style='color: #2c3e50;'>🔧 {model_type_val.replace('_', ' ').title()}</strong>
            <br><small style='color: #6c757d;'>{model_info.get('description', 'No description')[:80]}...</small>
        </div>
        
        <div style='background: white; padding: 8px; border-radius: 4px; margin-top: 8px;'>
            <strong style='color: #495057;'>Optimasi Aktif:</strong><br>
            <div style='font-size: 13px; line-height: 1.6; margin-top: 5px;'>
                {features_html}
            </div>
        </div>
        
        <div style='background: white; padding: 8px; border-radius: 4px; margin-top: 8px;'>
            <strong style='color: #495057;'>Spesifikasi:</strong><br>
            <div style='font-size: 12px; color: #6c757d; margin-top: 5px;'>
                Features: {backbone_info.get('features', 'N/A')}<br>
                Stride: {backbone_info.get('stride', 'N/A')}<br>
                Classes: {model_info.get('num_classes', 7)}<br>
                Input: {model_info.get('img_size', 640)}px
            </div>
        </div>
    </div>
    """
    
    summary_widget.value = summary_html

def _setup_summary_auto_update(form_components: Dict[str, Any], summary_widget: widgets.HTML) -> None:
    """Setup auto-update untuk summary widget"""
    def update_summary(*args):
        # Skip jika dalam suppression mode
        if form_components.get('_suppress_all_changes', False):
            return
        try:
            _update_summary_content(form_components, summary_widget)
        except Exception:
            pass  # Silent fail
    
    # Register observers
    observers = [
        ('backbone_dropdown', 'value'),
        ('model_type_dropdown', 'value'), 
        ('use_attention_checkbox', 'value'),
        ('use_residual_checkbox', 'value'),
        ('use_ciou_checkbox', 'value')
    ]
    
    [getattr(form_components.get(widget), 'observe', lambda: None)(update_summary, names=observe_name)
     for widget, observe_name in observers if widget in form_components]