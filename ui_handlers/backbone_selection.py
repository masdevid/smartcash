"""
File: smartcash/ui_handlers/backbone_selection.py
Author: Refactor
Deskripsi: Handler untuk UI pemilihan backbone dan layer model SmartCash (optimized).
"""

from IPython.display import display, HTML, clear_output

from smartcash.utils.ui_utils import create_status_indicator

def setup_backbone_selection_handlers(ui_components, config=None):
    """Setup handler untuk UI pemilihan backbone dan konfigurasi layer."""
    # Inisialisasi dependencies
    logger = None
    layer_config_manager = None
    config_manager = None
    
    try:
        from smartcash.utils.logger import get_logger
        from smartcash.utils.layer_config_manager import get_layer_config
        from smartcash.utils.config_manager import get_config_manager
        
        logger = get_logger("backbone_selection")
        layer_config_manager = get_layer_config(logger=logger)
        config_manager = get_config_manager(logger=logger)
        
    except ImportError as e:
        with ui_components['status_output']:
            display(create_status_indicator("warning", f"⚠️ Beberapa modul tidak tersedia: {str(e)}"))
        return ui_components
    
    # Fungsi untuk update config dari UI
    def update_config_from_ui():
        """Ambil nilai dari UI dan update config."""
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
        
        # Update config
        if not config:
            return {
                'model': {
                    'backbone': backbone_type,
                    'pretrained': pretrained,
                    'freeze_backbone': freeze_backbone
                },
                'layers': layers
            }
            
        # Ensure model config exists
        if 'model' not in config:
            config['model'] = {}
        
        # Update model settings
        config['model'].update({
            'backbone': backbone_type,
            'pretrained': pretrained,
            'freeze_backbone': freeze_backbone
        })
        
        # Update layer settings
        config['layers'] = layers
        
        return config
    
    # Fungsi untuk update UI dari config
    def update_ui_from_config():
        """Update UI components dari config."""
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
        
        # Update layer summary display
        update_layer_summary()
    
    # Function to update layer configuration summary display
    def update_layer_summary():
        """Update layer summary display."""
        try:
            with ui_components['layer_summary']:
                clear_output(wait=True)
                
                cfg = update_config_from_ui()
                if not cfg:
                    return
                
                # Create tabular summary of layers
                html = "<h4 style='margin-top:0; color:#2c3e50'>📋 Layer Configuration Summary</h4>"
                html += "<table style='width:100%; border-collapse:collapse; margin-top:10px;'>"
                html += "<tr style='background:#f2f2f2'><th>Layer</th><th>Status</th><th>Threshold</th><th>Classes</th></tr>"
                
                for name in ['banknote', 'nominal', 'security']:
                    # Get settings from updated config
                    settings = cfg['layers'].get(name, {})
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
                backbone_type = cfg['model'].get('backbone', 'efficientnet_b4')
                pretrained = cfg['model'].get('pretrained', True)
                freeze = cfg['model'].get('freeze_backbone', True)
                
                html += f"""
                <div style="margin-top:15px; padding:10px; background:#f8f9fa; border-radius:5px; color:#2c3e50">
                <p><b>🧠 Backbone:</b> {backbone_type} {'(pretrained)' if pretrained else ''} {'(frozen)' if freeze else ''}</p>
                <p><b>Enabled Layers:</b> {', '.join([name.capitalize() for name, settings in cfg['layers'].items() if settings.get('enabled', False)])}</p>
                </div>
                """
                
                display(HTML(html))
                
        except Exception as e:
            if logger:
                logger.error(f"❌ Error updating layer summary: {str(e)}")
            display(HTML(f"<p style='color:red'>❌ Error updating summary: {str(e)}</p>"))
    
    # Handler untuk save button
    def on_save_click(b):
        with ui_components['status_output']:
            clear_output()
            display(create_status_indicator("info", "🔄 Menyimpan konfigurasi backbone dan layer..."))
            
            try:
                # Update config dari UI
                updated_config = update_config_from_ui()
                
                # Simpan ke file
                if config_manager:
                    success = config_manager.save_config(
                        updated_config, 
                        "configs/model_config.yaml",
                        backup=True
                    )
                    
                    if success:
                        display(create_status_indicator(
                            "success", 
                            "✅ Konfigurasi model berhasil disimpan ke configs/model_config.yaml"
                        ))
                    else:
                        display(create_status_indicator(
                            "warning", 
                            "⚠️ Konfigurasi diupdate dalam memori, tetapi gagal menyimpan ke file"
                        ))
                else:
                    # Just update in-memory if config_manager not available
                    display(create_status_indicator(
                        "success", 
                        "✅ Konfigurasi model diupdate dalam memori"
                    ))
                
                # Create summary of settings
                backbone_type = updated_config['model']['backbone']
                pretrained = updated_config['model']['pretrained']
                layers_enabled = sum(1 for layer, settings in updated_config['layers'].items() 
                                    if settings.get('enabled', False))
                
                summary_html = f"""
                <div style="background-color: #f8f9fa; padding: 10px; border-radius: 5px; margin-top: 10px; color:#2c3e50">
                    <h4 style="margin-top: 0;">📊 Model Configuration Summary</h4>
                    <ul>
                        <li><b>Backbone:</b> {backbone_type}</li>
                        <li><b>Pretrained:</b> {'Yes' if pretrained else 'No'}</li>
                        <li><b>Layers:</b> {layers_enabled} active layers</li>
                        <li><b>Default confidence:</b> {updated_config['model'].get('confidence', 0.25)}</li>
                    </ul>
                </div>
                """
                display(HTML(summary_html))
                
            except Exception as e:
                display(create_status_indicator("error", f"❌ Error saat menyimpan konfigurasi: {str(e)}"))
    
    # Handler untuk reset button
    def on_reset_click(b):
        with ui_components['status_output']:
            clear_output()
            display(create_status_indicator("info", "🔄 Reset konfigurasi ke default..."))
            
            try:
                # Default config values
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
                
                # Update UI from default
                if config:
                    config.update(default_config)
                
                update_ui_from_config()
                
                display(create_status_indicator("success", "✅ Konfigurasi berhasil direset ke default"))
                
            except Exception as e:
                display(create_status_indicator("error", f"❌ Error saat reset konfigurasi: {str(e)}"))
    
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
        if logger:
            logger.info("✅ Backbone selection handlers cleaned up")
    
    # Add cleanup to UI components
    ui_components['cleanup'] = cleanup
    
    return ui_components