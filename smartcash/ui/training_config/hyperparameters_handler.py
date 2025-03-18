"""
File: smartcash/ui/training_config/hyperparameters_handler.py
Deskripsi: Handler untuk konfigurasi hyperparameter model dengan ui_helpers
"""

import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
from typing import Dict, Any, Optional

def setup_hyperparameters_handlers(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """
    Setup handler untuk komponen UI hyperparameters.
    
    Args:
        ui_components: Dictionary komponen UI
        env: Environment manager
        config: Konfigurasi aplikasi
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    from smartcash.ui.training_config.config_handler import (
        save_config, reset_config, get_config_manager
    )
    
    # Import helper functions dari ui_helpers
    from smartcash.ui.utils.ui_helpers import create_status_indicator, update_output_area
    
    # Dapatkan logger jika tersedia
    logger = None
    try:
        from smartcash.common.logger import get_logger
        logger = get_logger(ui_components.get('module_name', 'hyperparameters'))
    except ImportError:
        pass
    
    # Validasi config
    if config is None:
        config = {}
    
    # Try to load config from file if training section not present
    if 'training' not in config:
        config_manager = get_config_manager(logger)
        if config_manager:
            try:
                loaded_config = config_manager.load_config("configs/training_config.yaml")
                if loaded_config and 'training' in loaded_config:
                    config.update(loaded_config)
            except Exception as e:
                if logger:
                    logger.warning(f"⚠️ Gagal memuat konfigurasi: {e}")
    
    # Default config values
    default_config = {
        'training': {
            'epochs': 50,
            'batch_size': 16,
            'lr0': 0.01,
            'lrf': 0.01,
            'optimizer': 'Adam',
            'scheduler': 'cosine',
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'early_stopping_patience': 10,
            'early_stopping_enabled': True,
            'early_stopping_monitor': 'val_mAP',
            'save_best_only': True,
            'save_period': 5,
            'box_loss_weight': 0.05,
            'obj_loss_weight': 0.5,
            'cls_loss_weight': 0.5,
            'use_ema': False,
            'use_swa': False,
            'mixed_precision': True
        }
    }
    
    # Handler untuk scheduler changes
    def on_scheduler_change(change):
        if change['name'] != 'value': return
        
        scheduler_type = change['new']
        scheduler_params = ui_components.get('scheduler_params')
        if not scheduler_params or not hasattr(scheduler_params, 'children'): return
        
        children = scheduler_params.children
        if len(children) < 4: return
        
        # Enable/disable komponen berdasarkan jenis scheduler
        children[2].disabled = scheduler_type != 'step'
        children[3].disabled = scheduler_type != 'step'
        
        # Update visualization
        update_lr_visualization(scheduler_type)
    
    # Handler untuk early stopping changes
    def on_early_stopping_change(change):
        if change['name'] != 'value': return
        
        enabled = change['new']
        early_stopping_params = ui_components.get('early_stopping_params')
        if not early_stopping_params or not hasattr(early_stopping_params, 'children'): return
        
        children = early_stopping_params.children
        if len(children) < 3: return
        
        # Enable/disable komponen
        children[1].disabled = not enabled
        children[2].disabled = not enabled
    
    # Update config dari UI
    def update_config_from_ui(current_config=None):
        if current_config is None:
            current_config = config
            
        # Ensure training section exists
        if 'training' not in current_config:
            current_config['training'] = {}
        
        # Helper untuk safely get widget value
        def get_value(container_name, idx, default, attr='value'):
            container = ui_components.get(container_name)
            if not container or not hasattr(container, 'children'): return default
            
            children = container.children
            if idx < 0 or idx >= len(children): return default
            
            return getattr(children[idx], attr, default)
        
        # Update dari komponen dasar
        current_config['training'].update({
            'epochs': int(get_value('basic_params', 0, 50)),
            'batch_size': int(get_value('basic_params', 1, 16)),
            'lr0': float(get_value('basic_params', 2, 0.01)),
            'optimizer': get_value('basic_params', 3, 'Adam')
        })
        
        # Update dari scheduler params
        current_config['training'].update({
            'scheduler': get_value('scheduler_params', 0, 'cosine'),
            'lrf': float(get_value('scheduler_params', 1, 0.01))
        })
        
        # Update scheduler-specific params jika tidak disabled
        if get_value('scheduler_params', 2, True, 'disabled') == False:
            current_config['training'].update({
                'scheduler_patience': int(get_value('scheduler_params', 2, 5)),
                'scheduler_factor': float(get_value('scheduler_params', 3, 0.1))
            })
        
        # Update dari early stopping params
        current_config['training'].update({
            'early_stopping_enabled': get_value('early_stopping_params', 0, True),
            'early_stopping_patience': int(get_value('early_stopping_params', 1, 10)),
            'early_stopping_monitor': get_value('early_stopping_params', 2, 'val_mAP'),
            'save_best_only': get_value('early_stopping_params', 3, True),
            'save_period': int(get_value('early_stopping_params', 4, 5))
        })
        
        # Update dari advanced params
        current_config['training'].update({
            'momentum': float(get_value('advanced_params', 0, 0.937)),
            'weight_decay': float(get_value('advanced_params', 1, 0.0005)),
            'mixed_precision': get_value('advanced_params', 2, True),
            'use_ema': get_value('advanced_params', 3, False),
            'use_swa': get_value('advanced_params', 4, False)
        })
        
        # Update dari loss params
        current_config['training'].update({
            'box_loss_weight': float(get_value('loss_params', 0, 0.05)),
            'obj_loss_weight': float(get_value('loss_params', 1, 0.5)),
            'cls_loss_weight': float(get_value('loss_params', 2, 0.5))
        })
        
        return current_config
        
    # Update UI dari config
    def update_ui_from_config():
        """Update komponen UI dari konfigurasi."""
        if not config or 'training' not in config:
            return
        
        training = config['training']
        
        # Helper untuk update widget dengan aman
        def set_value(container_name, idx, value, min_val=None, max_val=None, options=None):
            container = ui_components.get(container_name)
            if not container or not hasattr(container, 'children'): 
                return
                
            children = container.children
            if idx < 0 or idx >= len(children): 
                return
                
            widget = children[idx]
            
            # Validasi options untuk dropdown
            if options and hasattr(widget, 'options') and value not in widget.options:
                return
                
            # Validasi range untuk sliders
            if min_val is not None and max_val is not None and hasattr(widget, 'min') and hasattr(widget, 'max'):
                value = min(max(value, min_val), max_val)
                
            try:
                widget.value = value
            except Exception as e:
                if logger:
                    logger.debug(f"⚠️ Tidak dapat update widget {container_name}[{idx}]: {e}")
        
        # Update basic params
        set_value('basic_params', 0, training.get('epochs', 50))
        set_value('basic_params', 1, training.get('batch_size', 16))
        set_value('basic_params', 2, training.get('lr0', 0.01))
        set_value('basic_params', 3, training.get('optimizer', 'Adam'))
        
        # Update scheduler params
        set_value('scheduler_params', 0, training.get('scheduler', 'cosine'))
        set_value('scheduler_params', 1, training.get('lrf', 0.01))
        set_value('scheduler_params', 2, training.get('scheduler_patience', 5))
        set_value('scheduler_params', 3, training.get('scheduler_factor', 0.1))
        
        # Update early stopping params
        set_value('early_stopping_params', 0, training.get('early_stopping_enabled', True))
        set_value('early_stopping_params', 1, training.get('early_stopping_patience', 10))
        set_value('early_stopping_params', 2, training.get('early_stopping_monitor', 'val_mAP'))
        set_value('early_stopping_params', 3, training.get('save_best_only', True))
        set_value('early_stopping_params', 4, training.get('save_period', 5))
        
        # Update advanced params
        set_value('advanced_params', 0, training.get('momentum', 0.937))
        set_value('advanced_params', 1, training.get('weight_decay', 0.0005))
        set_value('advanced_params', 2, training.get('mixed_precision', True))
        set_value('advanced_params', 3, training.get('use_ema', False))
        set_value('advanced_params', 4, training.get('use_swa', False))
        
        # Update loss params
        set_value('loss_params', 0, training.get('box_loss_weight', 0.05))
        set_value('loss_params', 1, training.get('obj_loss_weight', 0.5))
        set_value('loss_params', 2, training.get('cls_loss_weight', 0.5))
        
        # Update visualisasi Learning Rate
        scheduler_type = training.get('scheduler', 'cosine')
        update_lr_visualization(scheduler_type)
    
    # Visualisasi Learning Rate
    def update_lr_visualization(scheduler_type):
        """Update visualisasi untuk learning rate schedule."""
        visualization_output = ui_components.get('visualization_output')
        if not visualization_output:
            return
            
        with visualization_output:
            clear_output(wait=True)
            
            # Dapatkan nilai lr dari config
            lr0 = config.get('training', {}).get('lr0', 0.01)
            lrf = config.get('training', {}).get('lrf', 0.01)
            final_lr = lr0 * lrf
            
            # Deskripsi per jenis scheduler
            descriptions = {
                'cosine': f"Cosine scheduler: Learning rate menurun halus dari {lr0} ke {final_lr} mengikuti kurva cosine.",
                'linear': f"Linear scheduler: Learning rate menurun linear dari {lr0} ke {final_lr}.",
                'step': "Step scheduler: Learning rate diturunkan dengan faktor tertentu setiap n epoch.",
                'OneCycleLR': "OneCycle: Learning rate meningkat lalu menurun kembali dengan pola tertentu.",
                'none': "Tidak menggunakan scheduler: Learning rate tetap konstan selama training."
            }
            
            description = descriptions.get(scheduler_type, f"Learning rate scheduler: {scheduler_type}")
            
            # Menggunakan create_info_alert dari ui_helpers jika tersedia
            try:
                from smartcash.ui.utils.ui_helpers import create_info_alert
                display(create_info_alert(
                    f"📝 Learning Rate Schedule: {scheduler_type}\n\n{description}",
                    "info"
                ))
            except ImportError:
                # Fallback ke implementasi manual
                display(HTML(f"""
                <div style="padding: 10px; background-color: #e3f2fd; border-left: 4px solid #2196F3; margin: 10px 0;">
                    <h4 style="margin-top: 0;">📝 Learning Rate Schedule: {scheduler_type}</h4>
                    <p>{description}</p>
                </div>
                """))
    
    # Handler untuk save button
    def on_save_click(b):
        save_config(
            ui_components,
            config,
            "configs/training_config.yaml",
            update_config_from_ui,
            "Hyperparameters"
        )
    
    # Handler untuk reset button
    def on_reset_click(b):
        reset_config(
            ui_components,
            config,
            default_config,
            update_ui_from_config,
            "Hyperparameters"
        )
    
    # Setup event handlers
    try:
        # Register callbacks
        ui_components['save_button'].on_click(on_save_click)
        ui_components['reset_button'].on_click(on_reset_click)
        
        # Register change handlers untuk komponen dinamis
        scheduler_params = ui_components.get('scheduler_params')
        if scheduler_params and len(scheduler_params.children) > 0:
            scheduler_params.children[0].observe(on_scheduler_change, names='value')
            
        early_stop_params = ui_components.get('early_stopping_params')
        if early_stop_params and len(early_stop_params.children) > 0:
            early_stop_params.children[0].observe(on_early_stopping_change, names='value')
    except Exception as e:
        if logger:
            logger.warning(f"⚠️ Error saat setup handlers: {e}")
    
    # Initialize UI dari config
    update_ui_from_config()
    
    # Cleanup function
    def cleanup():
        """Cleanup resources."""
        try:
            # Unobserve handlers
            scheduler_params = ui_components.get('scheduler_params')
            if scheduler_params and len(scheduler_params.children) > 0:
                scheduler_params.children[0].unobserve(on_scheduler_change, names='value')
                
            early_stop_params = ui_components.get('early_stopping_params')
            if early_stop_params and len(early_stop_params.children) > 0:
                early_stop_params.children[0].unobserve(on_early_stopping_change, names='value')
                
            if logger:
                logger.info("✅ Hyperparameters handler resources cleaned up")
        except Exception as e:
            if logger:
                logger.warning(f"⚠️ Error saat cleanup: {e}")
    
    # Add cleanup function
    ui_components['cleanup'] = cleanup
    
    return ui_components