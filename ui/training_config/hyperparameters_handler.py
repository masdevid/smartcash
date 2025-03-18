"""
File: smartcash/ui/training_config/hyperparameters_handler.py
Deskripsi: Handler yang dioptimalkan untuk konfigurasi hyperparameter model
"""

import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
from typing import Dict, Any, Optional

def setup_hyperparameters_handlers(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """Setup handler untuk komponen UI hyperparameters."""
    try:
        # Import dasar dengan penanganan error minimal
        from smartcash.ui.training_config.config_handler import save_config, reset_config, get_config_manager
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
        
        # Load config dari file jika training section tidak ada
        if 'training' not in config:
            config_manager = get_config_manager()
            if config_manager:
                try:
                    loaded_config = config_manager.load_config("configs/training_config.yaml")
                    if loaded_config and 'training' in loaded_config:
                        config.update(loaded_config)
                except Exception as e:
                    if logger:
                        logger.warning(f"⚠️ Gagal memuat konfigurasi: {e}")
        
        # Default config (lebih sederhana)
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
                'mixed_precision': True
            }
        }
        
        # Handler untuk scheduler changes (diringkas)
        def on_scheduler_change(change):
            if change['name'] != 'value': return
            
            scheduler_type = change['new']
            scheduler_params = ui_components.get('scheduler_params')
            if not scheduler_params or not hasattr(scheduler_params, 'children') or len(scheduler_params.children) < 4: 
                return
            
            # Enable/disable komponen berdasarkan jenis scheduler
            scheduler_params.children[2].disabled = scheduler_type != 'step'
            scheduler_params.children[3].disabled = scheduler_type != 'step'
            
            # Update visualization
            update_lr_visualization(scheduler_type)
        
        # Handler untuk early stopping changes (diringkas)
        def on_early_stopping_change(change):
            if change['name'] != 'value': return
            
            enabled = change['new']
            early_stopping_params = ui_components.get('early_stopping_params')
            if not early_stopping_params or not hasattr(early_stopping_params, 'children') or len(early_stopping_params.children) < 3: 
                return
            
            # Enable/disable komponen
            early_stopping_params.children[1].disabled = not enabled
            early_stopping_params.children[2].disabled = not enabled
        
        # Fungsi update config dari UI (diringkas)
        def update_config_from_ui(current_config=None):
            if current_config is None:
                current_config = config
                
            # Ensure training section exists
            if 'training' not in current_config:
                current_config['training'] = {}
            
            # Helper untuk safely get widget value
            def get_value(container_name, idx, default, attr='value'):
                container = ui_components.get(container_name)
                if not container or not hasattr(container, 'children') or idx >= len(container.children): 
                    return default
                return getattr(container.children[idx], attr, default)
            
            # Update konfigurasi dari semua section
            current_config['training'].update({
                # Basic params
                'epochs': int(get_value('basic_params', 0, 50)),
                'batch_size': int(get_value('basic_params', 1, 16)),
                'lr0': float(get_value('basic_params', 2, 0.01)),
                'optimizer': get_value('basic_params', 3, 'Adam'),
                
                # Scheduler params
                'scheduler': get_value('scheduler_params', 0, 'cosine'),
                'lrf': float(get_value('scheduler_params', 1, 0.01)),
                
                # Early stopping params
                'early_stopping_enabled': get_value('early_stopping_params', 0, True),
                'early_stopping_patience': int(get_value('early_stopping_params', 1, 10)),
                'early_stopping_monitor': get_value('early_stopping_params', 2, 'val_mAP'),
                'save_best_only': get_value('early_stopping_params', 3, True),
                'save_period': int(get_value('early_stopping_params', 4, 5)),
                
                # Advanced params
                'momentum': float(get_value('advanced_params', 0, 0.937)),
                'weight_decay': float(get_value('advanced_params', 1, 0.0005)),
                'mixed_precision': get_value('advanced_params', 2, True),
                
                # Loss params
                'box_loss_weight': float(get_value('loss_params', 0, 0.05)),
                'obj_loss_weight': float(get_value('loss_params', 1, 0.5)),
                'cls_loss_weight': float(get_value('loss_params', 2, 0.5))
            })
            
            # Add scheduler-specific params jika aktif
            if not get_value('scheduler_params', 2, True, 'disabled'):
                current_config['training'].update({
                    'scheduler_patience': int(get_value('scheduler_params', 2, 5)),
                    'scheduler_factor': float(get_value('scheduler_params', 3, 0.1))
                })
            
            return current_config
            
        # Fungsi update UI dari config (diringkas)
        def update_ui_from_config():
            """Update komponen UI dari konfigurasi."""
            if not config or 'training' not in config:
                return
            
            training = config['training']
            
            # Helper untuk update widget dengan aman
            def set_value(container_name, idx, value):
                container = ui_components.get(container_name)
                if not container or not hasattr(container, 'children') or idx >= len(container.children): 
                    return
                    
                try:
                    container.children[idx].value = value
                except Exception as e:
                    if logger:
                        logger.debug(f"⚠️ Error update widget {container_name}[{idx}]: {e}")
            
            # Update semua parameters secara efisien
            # Basic
            set_value('basic_params', 0, training.get('epochs', 50))
            set_value('basic_params', 1, training.get('batch_size', 16))
            set_value('basic_params', 2, training.get('lr0', 0.01))
            set_value('basic_params', 3, training.get('optimizer', 'Adam'))
            
            # Scheduler
            set_value('scheduler_params', 0, training.get('scheduler', 'cosine'))
            set_value('scheduler_params', 1, training.get('lrf', 0.01))
            set_value('scheduler_params', 2, training.get('scheduler_patience', 5))
            set_value('scheduler_params', 3, training.get('scheduler_factor', 0.1))
            
            # Early stopping
            set_value('early_stopping_params', 0, training.get('early_stopping_enabled', True))
            set_value('early_stopping_params', 1, training.get('early_stopping_patience', 10))
            set_value('early_stopping_params', 2, training.get('early_stopping_monitor', 'val_mAP'))
            set_value('early_stopping_params', 3, training.get('save_best_only', True))
            set_value('early_stopping_params', 4, training.get('save_period', 5))
            
            # Advanced
            set_value('advanced_params', 0, training.get('momentum', 0.937))
            set_value('advanced_params', 1, training.get('weight_decay', 0.0005))
            set_value('advanced_params', 2, training.get('mixed_precision', True))
            
            # Loss
            set_value('loss_params', 0, training.get('box_loss_weight', 0.05))
            set_value('loss_params', 1, training.get('obj_loss_weight', 0.5))
            set_value('loss_params', 2, training.get('cls_loss_weight', 0.5))
            
            # Update visualisasi
            update_lr_visualization(training.get('scheduler', 'cosine'))
        
        # Visualisasi Learning Rate (diringkas)
        def update_lr_visualization(scheduler_type):
            visualization_output = ui_components.get('visualization_output')
            if not visualization_output:
                return
                
            with visualization_output:
                clear_output(wait=True)
                
                # Dapatkan nilai learning rate
                lr0 = config.get('training', {}).get('lr0', 0.01)
                lrf = config.get('training', {}).get('lrf', 0.01)
                final_lr = lr0 * lrf
                
                # Deskripsi per jenis scheduler
                descriptions = {
                    'cosine': f"Cosine scheduler: LR menurun halus dari {lr0} ke {final_lr} mengikuti kurva cosine.",
                    'linear': f"Linear scheduler: LR menurun linear dari {lr0} ke {final_lr}.",
                    'step': "Step scheduler: LR diturunkan dengan faktor tertentu setiap n epoch.",
                    'OneCycleLR': "OneCycle: LR meningkat lalu menurun kembali dengan pola tertentu.",
                    'none': "No scheduler: LR tetap konstan selama training."
                }
                
                description = descriptions.get(scheduler_type, f"Learning rate scheduler: {scheduler_type}")
                
                # Tampilkan info
                try:
                    from smartcash.ui.utils.ui_helpers import create_info_alert
                    display(create_info_alert(f"📝 Learning Rate Schedule: {scheduler_type}\n\n{description}", "info"))
                except ImportError:
                    # Fallback sederhana
                    display(HTML(f"""
                    <div style="padding:10px; background:#e3f2fd; border-left:4px solid #2196F3; margin:10px 0;">
                        <h4 style="margin-top:0">📝 Learning Rate Schedule: {scheduler_type}</h4>
                        <p>{description}</p>
                    </div>
                    """))
        
        # Handler untuk save/reset buttons
        def on_save_click(b):
            save_config(ui_components, config, "configs/training_config.yaml", update_config_from_ui, "Hyperparameters")
        
        def on_reset_click(b):
            reset_config(ui_components, config, default_config, update_ui_from_config, "Hyperparameters")
        
        # Register semua event handlers
        ui_components['save_button'].on_click(on_save_click)
        ui_components['reset_button'].on_click(on_reset_click)
        
        # Register handler untuk komponen dinamis
        ui_components.get('scheduler_params', widgets.VBox()).children[0].observe(on_scheduler_change, names='value')
        ui_components.get('early_stopping_params', widgets.VBox()).children[0].observe(on_early_stopping_change, names='value')
        
        # Initialize UI dari config
        update_ui_from_config()
        
        # Cleanup function (diringkas)
        def cleanup():
            try:
                scheduler_params = ui_components.get('scheduler_params', widgets.VBox())
                early_stop_params = ui_components.get('early_stopping_params', widgets.VBox())
                
                # Unobserve handlers secara efisien
                if hasattr(scheduler_params, 'children') and len(scheduler_params.children) > 0:
                    scheduler_params.children[0].unobserve(on_scheduler_change, names='value')
                    
                if hasattr(early_stop_params, 'children') and len(early_stop_params.children) > 0:
                    early_stop_params.children[0].unobserve(on_early_stopping_change, names='value')
                    
                if logger:
                    logger.info("✅ Hyperparameters handler cleaned up")
            except Exception as e:
                if logger:
                    logger.warning(f"⚠️ Error cleanup: {e}")
        
        # Assign cleanup function
        ui_components['cleanup'] = cleanup
        
    except Exception as e:
        # Fallback sederhana jika terjadi error
        if 'status' in ui_components:
            with ui_components['status']:
                display(HTML(f"<p style='color:red'>❌ Error setup hyperparameters handler: {str(e)}</p>"))
        else:
            print(f"❌ Error setup hyperparameters handler: {str(e)}")
    
    return ui_components