"""
File: smartcash/ui/training_config/backbone_selection_handler.py
Deskripsi: Handler utama untuk pemilihan model dan konfigurasi layer
"""

from IPython.display import display, HTML, clear_output
from typing import Dict, Any, Optional

from smartcash.ui.utils.alert_utils import create_status_indicator
from smartcash.ui.training_config.model_config_definitions import get_default_config, get_model_config
from smartcash.ui.training_config.model_ui_updater import update_ui_for_model_type, update_ui_from_config, update_layer_summary
from smartcash.ui.training_config.model_config_extractor import extract_config_from_ui

def setup_backbone_selection_handlers(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """Setup handler untuk komponen UI backbone selection."""
    # Import dengan penanganan error sederhana
    try:
        from smartcash.ui.training_config.config_handler import save_config, reset_config
        
        # Dapatkan logger jika tersedia
        logger = None
        try:
            from smartcash.common.logger import get_logger
            logger = get_logger(ui_components.get('module_name', 'backbone_selection'))
        except ImportError:
            pass
        
        # Coba dapatkan model manager untuk akses ke model types jika tersedia
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
        
        # Default config dari model_config_definitions
        default_config = get_default_config()
        
        # Fungsi update config & UI (untuk pembaruan model_type)
        def on_model_type_change(change):
            """Handler untuk perubahan model type."""
            if change['name'] != 'value':
                return
                
            # Parse model type dari pilihan dropdown
            model_option = change['new']
            model_type = model_option.split(' - ')[0].strip()
            
            # Update UI untuk model type ini
            update_ui_for_model_type(ui_components, model_type, config)
            
            # Update summary
            update_layer_summary(ui_components, config)
            
            # Tampilkan info perubahan
            with ui_components['status']:
                clear_output(wait=True)
                display(create_status_indicator("info", 
                    f"ℹ️ Model diubah ke {model_type}. Backbone dan fitur diupdate otomatis."))
        
        # Handler untuk save/reset buttons
        def on_save_click(b):
            save_config(ui_components, config, "configs/model_config.yaml", extract_config_from_ui, "Konfigurasi Model")
        
        def on_reset_click(b):
            reset_config(ui_components, config, default_config, 
                         lambda: update_ui_from_config(ui_components, config), 
                         "Konfigurasi Model")
        
        # Register event handlers
        model_dropdown = ui_components.get('model_options', {}).children[0] if (
            'model_options' in ui_components and 
            hasattr(ui_components['model_options'], 'children') and 
            len(ui_components['model_options'].children) > 0
        ) else None
            
        if model_dropdown:
            model_dropdown.observe(on_model_type_change, names='value')
            
        # Register callbacks untuk save/reset buttons
        if 'save_button' in ui_components:
            ui_components['save_button'].on_click(on_save_click)
        if 'reset_button' in ui_components:
            ui_components['reset_button'].on_click(on_reset_click)
        
        # Register change listeners untuk layer updates
        layer_config = ui_components.get('layer_config')
        if layer_config and hasattr(layer_config, 'children'):
            for layer_row in layer_config.children:
                if hasattr(layer_row, 'children'):
                    for control in layer_row.children[:2]:  # Cukup observe 2 kontrol pertama
                        control.observe(
                            lambda change: update_layer_summary(ui_components, config) if change['name'] == 'value' else None, 
                            names='value'
                        )
        
        # Initialize UI dari config
        update_ui_from_config(ui_components, config)
        
        # Fungsi cleanup yang sederhana
        def cleanup():
            if 'model_options' in ui_components and hasattr(ui_components['model_options'], 'children'):
                model_dropdown = ui_components['model_options'].children[0]
                if model_dropdown:
                    model_dropdown.unobserve(on_model_type_change, names='value')
                    
            layer_config = ui_components.get('layer_config')
            if layer_config and hasattr(layer_config, 'children'):
                for layer_row in layer_config.children:
                    if hasattr(layer_row, 'children'):
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