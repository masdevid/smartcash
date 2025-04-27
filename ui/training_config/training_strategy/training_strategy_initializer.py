"""
File: smartcash/ui/training_config/training_strategy/training_strategy_initializer.py
Deskripsi: Initializer untuk UI konfigurasi strategi pelatihan model
"""

from typing import Dict, Any, Optional
from IPython.display import display, HTML
import os
import yaml
import copy

def initialize_training_strategy_ui(env: Any = None, config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Inisialisasi UI untuk konfigurasi strategi pelatihan model.
    
    Args:
        env: Environment manager
        config: Konfigurasi untuk model
        
    Returns:
        Dict berisi komponen UI
    """
    ui_components = {'module_name': 'training_strategy'}
    
    try:
        # Import dependency
        from smartcash.common.environment import get_environment_manager
        from smartcash.common.config import get_config_manager
        
        # Dapatkan environment dan config jika belum tersedia
        env = env or get_environment_manager()
        config_manager = get_config_manager()
        
        # Load konfigurasi dari file jika belum tersedia
        config_path = "configs/training_config.yaml"
        if config is None:
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r') as f:
                        config = yaml.safe_load(f)
                except Exception as e:
                    config = config_manager.config
            else:
                config = config_manager.config
        
        # Buat komponen UI
        from smartcash.ui.training_config.training_strategy.components.training_strategy_components import create_training_strategy_ui
        ui_components.update(create_training_strategy_ui(config))
        
        # Tambahkan tombol konfigurasi dari komponen standar
        from smartcash.ui.components.config_buttons import create_config_buttons
        config_buttons = create_config_buttons()
        ui_components.update({
            'save_button': config_buttons['save_button'],
            'reset_button': config_buttons['reset_button'],
            'config_buttons': config_buttons['container']
        })
        
        # Setup multi-progress tracking
        from smartcash.ui.handlers.multi_progress import setup_multi_progress_tracking
        setup_multi_progress_tracking(ui_components, "training_strategy", "training_strategy_step")
        
        # Setup handlers untuk tombol save dan reset
        def on_save_config(b):
            from smartcash.ui.utils.alert_utils import create_status_indicator
            
            # Update config dari UI
            updated_config = update_config_from_ui(ui_components, copy.deepcopy(config))
            
            # Simpan konfigurasi ke file
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            try:
                with open(config_path, 'w') as f:
                    yaml.dump(updated_config, f, default_flow_style=False)
                success = True
                message = "✅ Konfigurasi strategi training berhasil disimpan"
            except Exception as e:
                success = False
                message = f"❌ Gagal menyimpan konfigurasi: {str(e)}"
            
            # Update status
            status_type = 'success' if success else 'error'
            if 'status' in ui_components:
                with ui_components['status']:
                    display(create_status_indicator(status_type, message))
        
        def on_reset_config(b):
            # Reset ke default config
            default_config = config_manager.get_default_config().get('training', {})
            ui_components = update_ui_from_config(ui_components, default_config)
            
            # Tampilkan status
            if 'status' in ui_components:
                from smartcash.ui.utils.alert_utils import create_status_indicator
                with ui_components['status']:
                    display(create_status_indicator('info', "🔄 Konfigurasi direset ke default"))
        
        # Register handler untuk tombol
        ui_components['save_button'].on_click(on_save_config)
        ui_components['reset_button'].on_click(on_reset_config)
        
        # Tambahkan fungsi helper ke ui_components
        ui_components.update({
            'update_config_from_ui': update_config_from_ui,
            'update_ui_from_config': update_ui_from_config,
            'config': config
        })
        
        # Setup handlers lainnya
        from smartcash.ui.training_config.training_strategy.handlers.button_handlers import setup_training_strategy_button_handlers
        from smartcash.ui.training_config.training_strategy.handlers.form_handlers import setup_training_strategy_form_handlers
        
        ui_components = setup_training_strategy_button_handlers(ui_components, env, config)
        ui_components = setup_training_strategy_form_handlers(ui_components, env, config)
        
        # Update UI dari config yang tersimpan
        ui_components = update_ui_from_config(ui_components, config)
        
        # Tampilkan container utama
        if 'main_container' in ui_components:
            # Ganti placeholder tombol dengan config_buttons
            if 'buttons_placeholder' in ui_components and hasattr(ui_components['main_container'], 'children'):
                # Dapatkan indeks buttons_placeholder dalam children
                children_list = list(ui_components['main_container'].children)
                placeholder_idx = next((i for i, child in enumerate(children_list) 
                                    if child is ui_components['buttons_placeholder']), -1)
                
                if placeholder_idx >= 0:
                    # Ganti placeholder dengan config_buttons
                    children_list[placeholder_idx] = config_buttons['container']
                    ui_components['main_container'].children = tuple(children_list)
                else:
                    # Jika placeholder tidak ditemukan, tambahkan sebelum status
                    if 'status' in ui_components:
                        status_idx = next((i for i, child in enumerate(children_list) 
                                        if child is ui_components['status']), len(children_list))
                        children_list.insert(status_idx, config_buttons['container'])
                        ui_components['main_container'].children = tuple(children_list)
            
            display(ui_components['main_container'])
        else:
            # Fallback jika main_container tidak ada
            from smartcash.ui.utils.alert_utils import create_alert_html
            display(HTML(create_alert_html(
                "Container utama tidak ditemukan. Mencoba menampilkan komponen yang tersedia.",
                "warning"
            )))
            # Coba tampilkan komponen yang tersedia
            if 'form' in ui_components:
                display(ui_components['form'])
            display(config_buttons['container'])
            if 'status' in ui_components:
                display(ui_components['status'])
        
    except Exception as e:
        # Gunakan utilitas fallback yang ada
        from smartcash.ui.utils.fallback_utils import create_fallback_ui
        ui_components = create_fallback_ui(ui_components, str(e), "error")
    
    return ui_components

# Fungsi untuk mengupdate konfigurasi dari UI
def update_config_from_ui(ui_components: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update konfigurasi dari nilai UI.
    
    Args:
        ui_components: Dictionary komponen UI
        config: Konfigurasi training strategy
        
    Returns:
        Konfigurasi yang diupdate
    """
    # Pastikan struktur config ada
    if 'training' not in config:
        config['training'] = {}
    
    # Ekstrak nilai dari form training strategy
    if 'early_stopping_checkbox' in ui_components:
        config['training']['early_stopping'] = ui_components['early_stopping_checkbox'].value
    
    if 'patience_slider' in ui_components:
        config['training']['patience'] = ui_components['patience_slider'].value
    
    if 'validation_split_slider' in ui_components:
        config['training']['validation_split'] = ui_components['validation_split_slider'].value
    
    if 'augmentation_checkbox' in ui_components:
        config['training']['use_augmentation'] = ui_components['augmentation_checkbox'].value
    
    # Simpan konfigurasi di ui_components
    ui_components['config'] = config
    
    return config

# Fungsi untuk mengupdate UI dari konfigurasi
def update_ui_from_config(ui_components: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update UI dari konfigurasi.
    
    Args:
        ui_components: Dictionary komponen UI
        config: Konfigurasi training strategy
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    # Dapatkan konfigurasi training
    training_config = config.get('training', {})
    
    # Update checkbox early stopping
    if 'early_stopping_checkbox' in ui_components:
        ui_components['early_stopping_checkbox'].value = training_config.get('early_stopping', True)
    
    # Update slider patience
    if 'patience_slider' in ui_components:
        ui_components['patience_slider'].value = training_config.get('patience', 10)
    
    # Update slider validation split
    if 'validation_split_slider' in ui_components:
        ui_components['validation_split_slider'].value = training_config.get('validation_split', 0.2)
    
    # Update checkbox augmentation
    if 'augmentation_checkbox' in ui_components:
        ui_components['augmentation_checkbox'].value = training_config.get('use_augmentation', True)
    
    # Simpan referensi config di ui_components
    ui_components['config'] = config
    
    return ui_components
