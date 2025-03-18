"""
File: smartcash/ui/training_config/backbone_selection_handler.py
Deskripsi: Handler untuk pemilihan backbone dan konfigurasi layer dengan ui_helpers
"""

from IPython.display import display, HTML, clear_output
from typing import Dict, Any, Optional

def setup_backbone_selection_handlers(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """
    Setup handler untuk komponen UI backbone selection.
    
    Args:
        ui_components: Dictionary komponen UI
        env: Environment manager
        config: Konfigurasi aplikasi
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    from smartcash.ui.training_config.config_handler import save_config, reset_config
    
    # Import helper functions dari ui_helpers
    from smartcash.ui.utils.ui_helpers import create_status_indicator, update_output_area
    
    # Dapatkan logger jika tersedia
    logger = None
    try:
        from smartcash.common.logger import get_logger
        logger = get_logger(ui_components.get('module_name', 'backbone_selection'))
    except ImportError:
        pass
    
    # Dapatkan layer config manager jika tersedia
    layer_config_manager = None
    try:
        from smartcash.common.layer_config import get_layer_config_manager
        layer_config_manager = get_layer_config_manager()
    except ImportError:
        pass
    
    # Validasi config
    if config is None:
        config = {}
    
    # Default config
    default_config = {
        'model': {
            'backbone': 'efficientnet_b4',
            'framework': 'YOLOv5',
            'pretrained': True,
            'confidence': 0.25,
            'iou_threshold': 0.45,
            'freeze_backbone': True
        },
        'layers': {
            'banknote': {
                'enabled': True,
                'threshold': 0.25
            },
            'nominal': {
                'enabled': True,
                'threshold': 0.30
            },
            'security': {
                'enabled': True,
                'threshold': 0.35
            }
        }
    }
    
    # Fungsi untuk update config dari UI
    def update_config_from_ui(current_config=None):
        """Ambil nilai dari UI dan update config."""
        if current_config is None:
            current_config = config
            
        # Get backbone config
        backbone_options = ui_components['backbone_options']
        backbone_selection = backbone_options.children[0].value
        backbone_type = 'efficientnet_b4' if 'EfficientNet' in backbone_selection else 'cspdarknet'
        pretrained = backbone_options.children[1].value
        freeze_backbone = backbone_options.children[2].value
        
        # Get layer config
        layer_config = ui_components['layer_config']
        layers = {}
        layer_names = ['banknote', 'nominal', 'security']
        
        for i, layer_name in enumerate(layer_names):
            if i < len(layer_config.children):
                layer_row = layer_config.children[i]
                
                if len(layer_row.children) >= 2:
                    layers[layer_name] = {
                        'enabled': layer_row.children[0].value,
                        'threshold': layer_row.children[1].value
                    }
        
        # Ensure model config exists
        if 'model' not in current_config:
            current_config['model'] = {}
        
        # Update model settings
        current_config['model'].update({
            'backbone': backbone_type,
            'pretrained': pretrained,
            'freeze_backbone': freeze_backbone
        })
        
        # Update layer settings
        current_config['layers'] = layers
        
        # Update layer summary
        update_layer_summary()
        
        return current_config
    
    # Fungsi untuk update UI dari config
    def update_ui_from_config():
        """Update UI components dari config."""
        # Check if config exists
        if not config:
            return
            
        # Update backbone settings
        if 'model' in config:
            model_cfg = config['model']
            backbone_options = ui_components['backbone_options']
            
            # Set backbone type
            if 'backbone' in model_cfg:
                backbone_type = model_cfg['backbone']
                backbone_options.children[0].value = (
                    'EfficientNet-B4 (Recommended)' if backbone_type == 'efficientnet_b4' 
                    else 'CSPDarknet'
                )
            
            # Set pretrained option
            if 'pretrained' in model_cfg:
                backbone_options.children[1].value = model_cfg['pretrained']
            
            # Set freeze backbone option
            if 'freeze_backbone' in model_cfg:
                backbone_options.children[2].value = model_cfg['freeze_backbone']
        
        # Update layer settings
        if 'layers' in config:
            layer_cfg = config['layers']
            layer_config = ui_components['layer_config']
            layer_names = ['banknote', 'nominal', 'security']
            
            for i, layer_name in enumerate(layer_names):
                if layer_name in layer_cfg and i < len(layer_config.children):
                    layer_row = layer_config.children[i]
                    layer_settings = layer_cfg[layer_name]
                    
                    if len(layer_row.children) >= 2:
                        # Set enabled state
                        if 'enabled' in layer_settings:
                            layer_row.children[0].value = layer_settings['enabled']
                        
                        # Set threshold
                        if 'threshold' in layer_settings:
                            layer_row.children[1].value = layer_settings['threshold']
        
        # Update layer summary
        update_layer_summary()
    
    # Fungsi untuk update layer summary
    def update_layer_summary():
        """Update layer summary display."""
        layer_summary = ui_components.get('layer_summary')
        if not layer_summary:
            return
            
        with layer_summary:
            clear_output(wait=True)
            
            try:
                current_config = update_config_from_ui({})
                
                # Create tabular summary of layers
                html = "<h4 style='margin-top:0; color:#2c3e50'>📋 Layer Configuration Summary</h4>"
                html += "<table style='width:100%; border-collapse:collapse; margin-top:10px;'>"
                html += "<tr style='background:#f2f2f2'><th>Layer</th><th>Status</th><th>Threshold</th><th>Classes</th></tr>"
                
                for name in ['banknote', 'nominal', 'security']:
                    # Get settings from updated config
                    settings = current_config['layers'].get(name, {})
                    enabled = settings.get('enabled', False)
                    threshold = settings.get('threshold', 0.25)
                    
                    # Get class info if layer_config_manager available
                    class_names = ""
                    if layer_config_manager:
                        layer_cfg = layer_config_manager.get_layer_config(name)
                        class_names = ", ".join(layer_cfg.get('classes', [])[:3])
                        if len(layer_cfg.get('classes', [])) > 3:
                            class_names += ", ..."
                    
                    # Status color and icon
                    status_color = "green" if enabled else "gray"
                    status_icon = "✅" if enabled else "❌"
                    
                    html += f"<tr>"
                    html += f"<td style='padding:8px; border-bottom:1px solid #ddd;'>{name.capitalize()}</td>"
                    html += f"<td style='padding:8px; border-bottom:1px solid #ddd; color:{status_color}'>{status_icon} {'Aktif' if enabled else 'Nonaktif'}</td>"
                    html += f"<td style='padding:8px; border-bottom:1px solid #ddd;'>{threshold:.2f}</td>"
                    html += f"<td style='padding:8px; border-bottom:1px solid #ddd;'>{class_names}</td>"
                    html += f"</tr>"
                
                html += "</table>"
                
                # Add info about model and backbone
                backbone_type = current_config['model'].get('backbone', 'efficientnet_b4')
                pretrained = current_config['model'].get('pretrained', True)
                freeze = current_config['model'].get('freeze_backbone', True)
                
                html += f"""
                <div style="margin-top:15px; padding:10px; background:#f8f9fa; border-radius:5px; color:#2c3e50">
                <p><b>🧠 Backbone:</b> {backbone_type} {'(pretrained)' if pretrained else ''} {'(frozen)' if freeze else ''}</p>
                <p><b>Enabled Layers:</b> {', '.join([name.capitalize() for name, settings in current_config['layers'].items() if settings.get('enabled', False)])}</p>
                </div>
                """
                
                display(HTML(html))
                
            except Exception as e:
                if logger:
                    logger.error(f"❌ Error updating layer summary: {str(e)}")
                display(HTML(f"<p style='color:red'>❌ Error updating summary: {str(e)}</p>"))
    
    # Handler untuk save button
    def on_save_click(b):
        save_config(
            ui_components,
            config,
            "configs/model_config.yaml",
            update_config_from_ui,
            "Konfigurasi Model"
        )
    
    # Handler untuk reset button
    def on_reset_click(b):
        reset_config(
            ui_components,
            config,
            default_config,
            update_ui_from_config,
            "Konfigurasi Model"
        )
    
    # Listen untuk perubahan pada UI components untuk update summary
    def on_component_change(change):
        if change['name'] != 'value':
            return
        update_layer_summary()
    
    # Register callbacks
    ui_components['save_button'].on_click(on_save_click)
    ui_components['reset_button'].on_click(on_reset_click)
    
    # Register change observers for summary updates
    for child in ui_components['backbone_options'].children:
        child.observe(on_component_change, names='value')
    
    for layer_row in ui_components['layer_config'].children:
        for control in layer_row.children[:2]:  # Only first 2 controls (checkbox & threshold)
            control.observe(on_component_change, names='value')
    
    # Inisialisasi UI dari config
    update_ui_from_config()
    
    # Define cleanup function
    def cleanup():
        """Clean up resources."""
        # Hapus observe handlers untuk menghindari memory leak
        for child in ui_components['backbone_options'].children:
            child.unobserve(on_component_change, names='value')
        
        for layer_row in ui_components['layer_config'].children:
            for control in layer_row.children[:2]:
                control.unobserve(on_component_change, names='value')
                
        if logger:
            logger.info("✅ Backbone selection handlers cleaned up")
    
    # Tambahkan fungsi cleanup ke ui_components
    ui_components['cleanup'] = cleanup
    
    return ui_components