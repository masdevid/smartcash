"""
File: smartcash/ui/training_config/backbone_selection_handler.py
Deskripsi: Handler yang dioptimalkan untuk pemilihan model dan konfigurasi layer yang sesuai dengan ModelManager
"""

from IPython.display import display, HTML, clear_output
from typing import Dict, Any, Optional

def setup_backbone_selection_handlers(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """Setup handler untuk komponen UI backbone selection."""
    # Import dengan penanganan error sederhana
    try:
        from smartcash.ui.training_config.config_handler import save_config, reset_config
        from smartcash.ui.utils.ui_helpers import create_status_indicator, update_output_area
        
        # Dapatkan logger jika tersedia
        logger = None
        try:
            from smartcash.common.logger import get_logger
            logger = get_logger(ui_components.get('module_name', 'backbone_selection'))
        except ImportError:
            pass
        
        # Coba dapatkan model manager untuk akses ke model types jika tersedia
        model_manager = None
        try:
            from smartcash.model.manager import ModelManager
            # Hanya untuk definisi tipe model, tidak perlu instance sebenarnya
            model_manager = ModelManager
        except ImportError:
            # Mode fallback jika model_manager tidak tersedia
            if logger:
                logger.warning("⚠️ ModelManager tidak tersedia, menggunakan definisi model tetap")
        
        # Validasi config
        if config is None:
            config = {}
        
        # Default config (sederhana)
        default_config = {
            'model': {
                'model_type': 'efficient_optimized',
                'backbone': 'efficientnet_b4',
                'pretrained': True,
                'freeze_backbone': True,
                'use_attention': True,
                'use_residual': False,
                'use_ciou': False,
                'num_repeats': 3
            },
            'layers': {
                'banknote': {'enabled': True, 'threshold': 0.25},
                'nominal': {'enabled': True, 'threshold': 0.30},
                'security': {'enabled': True, 'threshold': 0.35}
            }
        }
        
        # Pemetaan model_type ke konfigurasi backbone dan fitur
        MODEL_CONFIGS = {
            'efficient_basic': {
                'backbone': 'efficientnet_b4',
                'use_attention': False,
                'use_residual': False,
                'use_ciou': False,
                'num_repeats': 1
            },
            'efficient_optimized': {
                'backbone': 'efficientnet_b4',
                'use_attention': True,
                'use_residual': False,
                'use_ciou': False,
                'num_repeats': 1
            },
            'efficient_advanced': {
                'backbone': 'efficientnet_b4',
                'use_attention': True,
                'use_residual': True,
                'use_ciou': True,
                'num_repeats': 3
            },
            'yolov5s': {
                'backbone': 'cspdarknet_s',
                'use_attention': False,
                'use_residual': False,
                'use_ciou': False,
                'num_repeats': 1
            },
            'efficient_experiment': {
                'backbone': 'efficientnet_b4',
                'use_attention': True,
                'use_residual': True,
                'use_ciou': True,
                'num_repeats': 3
            }
        }
        
        # Fungsi update config & UI (untuk pembaruan model_type)
        def on_model_type_change(change):
            """Handler untuk perubahan model type."""
            if change['name'] != 'value':
                return
                
            # Parse model type dari pilihan dropdown
            model_option = change['new']
            model_type = model_option.split(' - ')[0].strip()
            
            # Dapatkan konfigurasi model
            model_config = MODEL_CONFIGS.get(model_type, MODEL_CONFIGS['efficient_optimized'])
            
            # Update backbone dropdown
            backbone_options = ui_components['backbone_options']
            if backbone_options and hasattr(backbone_options, 'children') and len(backbone_options.children) > 0:
                backbone_dropdown = backbone_options.children[0]
                
                for i, option in enumerate(backbone_dropdown.options):
                    if option.startswith(model_config['backbone']):
                        backbone_dropdown.index = i
                        break
            
            # Update fitur options
            features_options = ui_components['features_options']
            if features_options and hasattr(features_options, 'children') and len(features_options.children) >= 4:
                # Update attention, residual, ciou checkboxes dan num_repeats slider
                features_options.children[0].value = model_config['use_attention']
                features_options.children[1].value = model_config['use_residual']
                features_options.children[2].value = model_config['use_ciou']
                features_options.children[3].value = model_config['num_repeats']
                
                # Aktifkan/nonaktifkan num_repeats berdasarkan use_residual
                features_options.children[3].disabled = not model_config['use_residual']
            
            # Update konfigurasi
            if 'model' not in config:
                config['model'] = {}
                
            config['model']['model_type'] = model_type
            config['model'].update(model_config)
            
            # Update summary
            update_layer_summary()
            
            # Tampilkan info perubahan
            with ui_components['status']:
                clear_output(wait=True)
                display(create_status_indicator("info", 
                    f"ℹ️ Model diubah ke {model_type}. Backbone dan fitur diupdate otomatis."))
        
        def update_config_from_ui(current_config=None):
            if current_config is None:
                current_config = config
                
            # Get model type
            model_options = ui_components['model_options']
            model_dropdown = model_options.children[0]
            model_option = model_dropdown.value
            model_type = model_option.split(' - ')[0].strip()
            
            # Get backbone config
            backbone_options = ui_components['backbone_options']
            backbone_dropdown = backbone_options.children[0]
            backbone_option = backbone_dropdown.options[backbone_dropdown.index]
            backbone_type = backbone_option.split(' - ')[0].strip()
            
            pretrained = backbone_options.children[1].value
            freeze_backbone = backbone_options.children[2].value
            
            # Get features config
            features_options = ui_components['features_options']
            use_attention = features_options.children[0].value
            use_residual = features_options.children[1].value
            use_ciou = features_options.children[2].value
            num_repeats = features_options.children[3].value
            
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
            
            # Memastikan model config ada
            if 'model' not in current_config:
                current_config['model'] = {}
            
            # Update model & settings
            current_config['model'].update({
                'model_type': model_type,
                'backbone': backbone_type,
                'pretrained': pretrained,
                'freeze_backbone': freeze_backbone,
                'use_attention': use_attention,
                'use_residual': use_residual,
                'use_ciou': use_ciou,
                'num_repeats': num_repeats
            })
            
            # Update layer config
            current_config['layers'] = layers
            
            # Update layer summary
            update_layer_summary()
            
            return current_config
        
        def update_ui_from_config():
            """Update UI dari konfigurasi."""
            # Check if config exists
            if not config or not isinstance(config, dict):
                return
                
            model_cfg = config.get('model', {})
            model_type = model_cfg.get('model_type', 'efficient_optimized')
            
            # Update model type dropdown
            model_options = ui_components['model_options']
            model_dropdown = model_options.children[0]
            
            for i, option in enumerate(model_dropdown.options):
                if option.startswith(model_type):
                    model_dropdown.index = i
                    break
                    
            # Backbone settings sudah otomatis diupdate oleh model type change handler
                    
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
                            # Set enabled & threshold
                            layer_row.children[0].value = layer_settings.get('enabled', True)
                            layer_row.children[1].value = layer_settings.get('threshold', 0.25)
            
            # Update summary
            update_layer_summary()
        
        # Fungsi update layer summary (diringkas)
        def update_layer_summary():
            layer_summary = ui_components.get('layer_summary')
            if not layer_summary:
                return
                
            with layer_summary:
                clear_output(wait=True)
                
                try:
                    current_config = update_config_from_ui({})
                    
                    # Dapatkan model type dan backbone
                    model_type = current_config['model'].get('model_type', 'efficient_optimized')
                    backbone_type = current_config['model'].get('backbone', 'efficientnet_b4')
                    
                    # Dapatkan status fitur
                    use_attention = current_config['model'].get('use_attention', False)
                    use_residual = current_config['model'].get('use_residual', False)
                    use_ciou = current_config['model'].get('use_ciou', False)
                    num_repeats = current_config['model'].get('num_repeats', 1)
                    
                    # Buat HTML summary yang informatif
                    html = "<h4 style='margin-top:0; color:#2c3e50'>📋 Model & Layer Configuration</h4>"
                    
                    # Tampilkan model info
                    html += f"""
                    <div style="margin-bottom:15px; padding:10px; background:#f8f9fa; border-radius:5px; color:#2c3e50">
                        <p><b>🧠 Model:</b> {model_type}</p>
                        <p><b>⚙️ Backbone:</b> {backbone_type}</p>
                        <p><b>🔌 Features:</b> 
                            {'Attention' if use_attention else ''} 
                            {' + Residual' if use_residual else ''} 
                            {' + CIoU' if use_ciou else ''}
                            {' (None)' if not any([use_attention, use_residual, use_ciou]) else ''}
                        </p>
                        {f'<p><b>🔄 Residual Blocks:</b> {num_repeats}</p>' if use_residual else ''}
                    </div>
                    """
                    
                    # Tampilkan tabel untuk layer
                    html += "<table style='width:100%; border-collapse:collapse; margin-top:10px;'>"
                    html += "<tr style='background:#f2f2f2'><th>Layer</th><th>Status</th><th>Threshold</th></tr>"
                    
                    for name in ['banknote', 'nominal', 'security']:
                        settings = current_config['layers'].get(name, {})
                        enabled = settings.get('enabled', False)
                        threshold = settings.get('threshold', 0.25)
                        
                        status_color = "green" if enabled else "gray"
                        status_icon = "✅" if enabled else "❌"
                        
                        html += f"<tr>"
                        html += f"<td style='padding:8px; border-bottom:1px solid #ddd;'>{name.capitalize()}</td>"
                        html += f"<td style='padding:8px; border-bottom:1px solid #ddd; color:{status_color}'>{status_icon} {'Aktif' if enabled else 'Nonaktif'}</td>"
                        html += f"<td style='padding:8px; border-bottom:1px solid #ddd;'>{threshold:.2f}</td>"
                        html += f"</tr>"
                    
                    html += "</table>"
                    
                    # Tampilkan enabled layers
                    enabled_layers = [name.capitalize() for name, settings in current_config['layers'].items() 
                                     if settings.get('enabled', False)]
                    
                    html += f"""
                    <div style="margin-top:15px; padding:10px; background:#f8f9fa; border-radius:5px; color:#2c3e50">
                    <p><b>✓ Layer Aktif:</b> {', '.join(enabled_layers) if enabled_layers else 'Tidak ada layer aktif'}</p>
                    </div>
                    """
                    
                    display(HTML(html))
                    
                except Exception as e:
                    if logger:
                        logger.error(f"❌ Error updating layer summary: {str(e)}")
                    display(HTML(f"<p style='color:red'>❌ Error: {str(e)}</p>"))
        
        # Handler untuk save/reset buttons
        def on_save_click(b):
            save_config(ui_components, config, "configs/model_config.yaml", update_config_from_ui, "Konfigurasi Model")
        
        def on_reset_click(b):
            reset_config(ui_components, config, default_config, update_ui_from_config, "Konfigurasi Model")
        
        # Register event handlers
        model_dropdown = ui_components['model_options'].children[0]
        model_dropdown.observe(on_model_type_change, names='value')
            
        # Register callbacks untuk save/reset buttons
        ui_components['save_button'].on_click(on_save_click)
        ui_components['reset_button'].on_click(on_reset_click)
        
        # Register change listeners untuk layer updates
        for layer_row in ui_components['layer_config'].children:
            for control in layer_row.children[:2]:  # Cukup observe 2 kontrol pertama
                control.observe(lambda change: update_layer_summary() if change['name'] == 'value' else None, names='value')
        
        # Initialize UI dari config
        update_ui_from_config()
        
        # Fungsi cleanup yang sederhana
        def cleanup():
            model_dropdown = ui_components['model_options'].children[0]
            model_dropdown.unobserve(on_model_type_change, names='value')
                
            for layer_row in ui_components['layer_config'].children:
                for control in layer_row.children[:2]:
                    if hasattr(control, 'unobserve_all'):
                        control.unobserve_all()
                    
            if logger:
                logger.info("✅ Backbone handler cleaned up")
        
        # Assign cleanup function
        ui_components['cleanup'] = cleanup
        
    except Exception as e:
        # Fallback sederhana
        if 'status' in ui_components:
            with ui_components['status']:
                display(HTML(f"<p style='color:red'>❌ Error setup backbone handler: {str(e)}</p>"))
        else:
            print(f"❌ Error setup backbone handler: {str(e)}")
    
    return ui_components