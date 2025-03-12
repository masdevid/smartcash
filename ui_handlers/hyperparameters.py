"""
File: smartcash/ui_handlers/hyperparameters.py
Author: Generated
Deskripsi: Handler untuk komponen UI konfigurasi hyperparameter model SmartCash.
"""

from IPython.display import display, clear_output, HTML
import threading
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from smartcash.utils.ui_utils import create_status_indicator

def setup_hyperparameters_handlers(ui_components, config=None):
    """Setup handlers untuk komponen UI konfigurasi hyperparameter."""
    # Inisialisasi dependencies
    deps = {}
    
    try:
        # Import needed modules
        from smartcash.utils.logger import get_logger
        from smartcash.utils.config_manager import get_config_manager
        from smartcash.utils.observer import EventTopics, EventDispatcher
        from smartcash.utils.observer.observer_manager import ObserverManager
        from smartcash.utils.environment_manager import EnvironmentManager
        from smartcash.utils.early_stopping import EarlyStopping
        
        # Setup dependencies
        deps['logger'] = get_logger("hyperparameters")
        deps['config_manager'] = get_config_manager(logger=deps['logger'])
        deps['observer_manager'] = ObserverManager(auto_register=True)
        deps['env_manager'] = EnvironmentManager(logger=deps['logger'])
        deps['observer_manager'].unregister_group("hyperparameters_ui")
        
        try:
            from smartcash.handlers.model import ModelManager
            if config: deps['model_manager'] = ModelManager(config, logger=deps['logger'])
        except ImportError:
            deps['logger'].warning("⚠️ ModelManager tidak tersedia")
            
    except ImportError as e:
        print(f"⚠️ Error saat import dependencies: {e}")
    
    # Load config jika belum ada
    if not config or 'training' not in config:
        if deps.get('config_manager'):
            try:
                config = deps['config_manager'].load_config("configs/training_config.yaml")
            except:
                pass
                
        if not config or 'training' not in config:
            config = {'training': {
                'epochs': 50, 'batch_size': 16, 'lr0': 0.01, 'lrf': 0.01,
                'optimizer': 'Adam', 'scheduler': 'cosine', 'momentum': 0.937,
                'weight_decay': 0.0005, 'early_stopping_patience': 10,
                'early_stopping_enabled': True, 'early_stopping_monitor': 'val_mAP',
                'save_best_only': True, 'save_period': 5,
                'box_loss_weight': 0.05, 'obj_loss_weight': 0.5, 'cls_loss_weight': 0.5,
                'use_ema': False, 'use_swa': False, 'mixed_precision': True
            }}
    
    # Handler helper untuk safely get components
    def get_component(name, default=None): 
        return ui_components.get(name, default)
    
    def get_widget_children(name):
        widget = get_component(name)
        return widget.children if widget and hasattr(widget, 'children') else []
    
    # Handler untuk scheduler changes
    def on_scheduler_change(change):
        if change['name'] != 'value': return
        
        scheduler_type = change['new']
        children = get_widget_children('scheduler_params')
        if len(children) < 4: return
        
        # Enable/disable komponen berdasarkan jenis scheduler
        children[2].disabled = scheduler_type != 'step'
        children[3].disabled = scheduler_type != 'step'
        
        # Update visualization
        update_lr_visualization()
    
    # Handler untuk early stopping changes
    def on_early_stopping_change(change):
        if change['name'] != 'value': return
        
        enabled = change['new']
        children = get_widget_children('early_stopping_params')
        if len(children) < 3: return
        
        # Enable/disable komponen
        children[1].disabled = not enabled
        children[2].disabled = not enabled
    
    # Update config dari UI
    def update_config_from_ui():
        if not config or 'training' not in config: 
            config.update({'training': {}})
        
        # Get widgets
        basic = get_widget_children('basic_params')
        scheduler = get_widget_children('scheduler_params')
        early_stop = get_widget_children('early_stopping_params')
        advanced = get_widget_children('advanced_params')
        loss = get_widget_children('loss_params')
        
        # Update training config with safe access
        training = {}
        
        # Helper to safely get widget value with default
        def get_value(widgets, idx, default, attr='value'):
            try:
                return getattr(widgets[idx], attr, default) if 0 <= idx < len(widgets) else default
            except:
                return default
        
        # Basic params
        if basic:
            training.update({
                'epochs': int(get_value(basic, 0, 50)),
                'batch_size': int(get_value(basic, 1, 16)),
                'lr0': float(get_value(basic, 2, 0.01)),
                'optimizer': get_value(basic, 3, 'Adam')
            })
        
        # Scheduler params
        if scheduler:
            training.update({
                'scheduler': get_value(scheduler, 0, 'cosine'),
                'lrf': float(get_value(scheduler, 1, 0.01))
            })
            
            if get_value(scheduler, 2, True, 'disabled') == False:
                training.update({
                    'scheduler_patience': int(get_value(scheduler, 2, 5)),
                    'scheduler_factor': float(get_value(scheduler, 3, 0.1))
                })
        
        # Early stopping params
        if early_stop:
            training.update({
                'early_stopping_enabled': get_value(early_stop, 0, True),
                'early_stopping_patience': int(get_value(early_stop, 1, 10)),
                'early_stopping_monitor': get_value(early_stop, 2, 'val_mAP'),
                'save_best_only': get_value(early_stop, 3, True),
                'save_period': int(get_value(early_stop, 4, 5))
            })
        
        # Advanced params
        if advanced:
            training.update({
                'momentum': float(get_value(advanced, 0, 0.937)),
                'weight_decay': float(get_value(advanced, 1, 0.0005)),
                'use_ema': get_value(advanced, 2, False),
                'use_swa': get_value(advanced, 3, False),
                'mixed_precision': get_value(advanced, 4, True)
            })
        
        # Loss weights
        if loss:
            training.update({
                'box_loss_weight': float(get_value(loss, 0, 0.05)),
                'obj_loss_weight': float(get_value(loss, 1, 0.5)),
                'cls_loss_weight': float(get_value(loss, 2, 0.5))
            })
        
        # Update config
        config['training'].update(training)
        
        # Notify if observer available
        observer_manager = deps.get('observer_manager')
        if observer_manager:
            try:
                from smartcash.utils.observer import EventTopics, EventDispatcher
                EventDispatcher.notify(
                    event_type=EventTopics.CONFIG_UPDATED,
                    sender="hyperparameters_handler",
                    config_type="training",
                    update_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                )
            except: pass
                    
        return config
    
    # Update UI dari config
    def update_ui_from_config():
        if not config or 'training' not in config: return
        training = config['training']
        
        # Helper untuk update widget dengan aman
        def set_value(widgets, idx, attr, value, min_val=None, max_val=None, options=None):
            try:
                if idx < 0 or idx >= len(widgets): return
                widget = widgets[idx]
                
                if options and value not in options: return
                if min_val is not None and max_val is not None:
                    value = min(max(value, min_val), max_val)
                
                setattr(widget, attr, value)
            except Exception:
                pass
        
        # Update widgets
        basic = get_widget_children('basic_params')
        scheduler = get_widget_children('scheduler_params')
        early_stop = get_widget_children('early_stopping_params')
        advanced = get_widget_children('advanced_params')
        loss = get_widget_children('loss_params')
        
        # Update basic params
        if basic:
            set_value(basic, 0, 'value', training.get('epochs', 50), 
                    min_val=getattr(basic[0], 'min', 10), max_val=getattr(basic[0], 'max', 200))
            set_value(basic, 1, 'value', training.get('batch_size', 16), 
                    min_val=getattr(basic[1], 'min', 4), max_val=getattr(basic[1], 'max', 64))
            set_value(basic, 2, 'value', training.get('lr0', 0.01))
            set_value(basic, 3, 'value', training.get('optimizer', 'Adam'), options=getattr(basic[3], 'options', None))
        
        # Update scheduler params
        if scheduler:
            set_value(scheduler, 0, 'value', training.get('scheduler', 'cosine'), options=getattr(scheduler[0], 'options', None))
            set_value(scheduler, 1, 'value', training.get('lrf', 0.01), 
                    min_val=getattr(scheduler[1], 'min', 0.001), max_val=getattr(scheduler[1], 'max', 0.1))
            set_value(scheduler, 2, 'value', training.get('scheduler_patience', 5),
                    min_val=getattr(scheduler[2], 'min', 1), max_val=getattr(scheduler[2], 'max', 10))
            set_value(scheduler, 3, 'value', training.get('scheduler_factor', 0.1),
                    min_val=getattr(scheduler[3], 'min', 0.01), max_val=getattr(scheduler[3], 'max', 0.5))
        
        # Update early stopping params
        if early_stop:
            set_value(early_stop, 0, 'value', training.get('early_stopping_enabled', True))
            set_value(early_stop, 1, 'value', training.get('early_stopping_patience', 10),
                    min_val=getattr(early_stop[1], 'min', 1), max_val=getattr(early_stop[1], 'max', 50))
            set_value(early_stop, 2, 'value', training.get('early_stopping_monitor', 'val_mAP'), 
                    options=getattr(early_stop[2], 'options', None))
            set_value(early_stop, 3, 'value', training.get('save_best_only', True))
            set_value(early_stop, 4, 'value', training.get('save_period', 5),
                    min_val=getattr(early_stop[4], 'min', 1), max_val=getattr(early_stop[4], 'max', 10))
        
        # Update advanced params
        if advanced:
            set_value(advanced, 0, 'value', training.get('momentum', 0.937),
                    min_val=getattr(advanced[0], 'min', 0.8), max_val=getattr(advanced[0], 'max', 0.999))
            set_value(advanced, 1, 'value', training.get('weight_decay', 0.0005))
            set_value(advanced, 2, 'value', training.get('use_ema', False))
            set_value(advanced, 3, 'value', training.get('use_swa', False))
            set_value(advanced, 4, 'value', training.get('mixed_precision', True))
        
        # Update loss params
        if loss:
            set_value(loss, 0, 'value', training.get('box_loss_weight', 0.05),
                    min_val=getattr(loss[0], 'min', 0.01), max_val=getattr(loss[0], 'max', 0.2))
            set_value(loss, 1, 'value', training.get('obj_loss_weight', 0.5),
                    min_val=getattr(loss[1], 'min', 0.1), max_val=getattr(loss[1], 'max', 1.0))
            set_value(loss, 2, 'value', training.get('cls_loss_weight', 0.5),
                    min_val=getattr(loss[2], 'min', 0.1), max_val=getattr(loss[2], 'max', 1.0))
    
    # Visualisasi learning rate
    def update_lr_visualization():
        visualization = get_component('visualization_output')
        if not visualization: return
        
        # Get widget values safely
        basic = get_widget_children('basic_params')
        scheduler = get_widget_children('scheduler_params')
        
        try:
            # Extract params
            epochs = int(basic[0].value) if len(basic) > 0 else 50
            initial_lr = float(basic[2].value) if len(basic) > 2 else 0.01
            final_lr = float(scheduler[1].value) if len(scheduler) > 1 else 0.01
            scheduler_type = scheduler[0].value if len(scheduler) > 0 else 'cosine'
            
            with visualization:
                clear_output()
                
                plt.figure(figsize=(10, 6))
                xs = np.linspace(0, epochs-1, epochs)
                
                # Calculate learning rates based on scheduler
                try:
                    if scheduler_type == 'cosine':
                        lrs = [initial_lr * (1 + np.cos(np.pi * x / epochs)) / 2 for x in xs]
                    elif scheduler_type == 'linear':
                        lrs = [initial_lr - (initial_lr - final_lr) * (x / (epochs-1)) for x in xs]
                    elif scheduler_type == 'step':
                        patience = int(scheduler[2].value) if len(scheduler) > 2 and not scheduler[2].disabled else 5
                        factor = float(scheduler[3].value) if len(scheduler) > 3 and not scheduler[3].disabled else 0.1
                        lrs = []
                        current_lr = initial_lr
                        for i in range(epochs):
                            if i > 0 and i % patience == 0:
                                current_lr *= factor
                            lrs.append(current_lr)
                    elif scheduler_type == 'exp':
                        gamma = np.exp(np.log(final_lr / initial_lr) / epochs)
                        lrs = [initial_lr * (gamma ** x) for x in xs]
                    elif scheduler_type == 'OneCycleLR':
                        half_epochs = epochs // 2
                        first_half = [initial_lr + (10*initial_lr - initial_lr) * (x / half_epochs) for x in range(half_epochs)]
                        second_half = [10*initial_lr * (1 - x / half_epochs) ** 1.5 for x in range(half_epochs + epochs % 2)]
                        lrs = first_half + second_half
                    else:
                        lrs = [initial_lr] * epochs
                except Exception:
                    lrs = [initial_lr] * epochs
                
                # Plot
                plt.plot(xs, lrs, 'b-', linewidth=2)
                plt.xlabel('Epoch')
                plt.ylabel('Learning Rate')
                plt.title(f'Learning Rate Schedule: {scheduler_type}')
                plt.grid(True, linestyle='--', alpha=0.6)
                plt.tight_layout()
                plt.show()
                
                # Display info
                html = f"""
                <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-top: 10px;">
                    <h4>📝 Learning Rate Schedule Analysis</h4>
                    <ul>
                        <li><b>Initial LR:</b> {initial_lr}</li>
                        <li><b>Final LR:</b> {final_lr}</li>
                        <li><b>Scheduler:</b> {scheduler_type}</li>
                        <li><b>Epochs:</b> {epochs}</li>
                    </ul>
                </div>
                """
                display(HTML(html))
            
            # Show visualization area
            visualization.layout.display = 'block'
        except Exception as e:
            logger = deps.get('logger')
            if logger: logger.warning(f"⚠️ Error visualisasi: {e}")
    
    # Handler untuk save config
    def on_save_configuration(b):
        output = get_component('status_output') or get_component('visualization_output')
        if not output: return
        
        with output:
            clear_output()
            display(create_status_indicator("info", "🔄 Menyimpan konfigurasi..."))
            
            try:
                # Update config dari UI
                update_config_from_ui()
                
                # Simpan config dengan config_manager
                success = False
                if deps.get('config_manager'):
                    env_manager = deps.get('env_manager')
                    is_drive_mounted = env_manager and getattr(env_manager, 'is_drive_mounted', False)
                    
                    success = deps['config_manager'].save_config(
                        config, 
                        "configs/training_config.yaml",
                        backup=True,
                        sync_to_drive=is_drive_mounted
                    )
                    
                    msg = "✅ Konfigurasi berhasil disimpan ke configs/training_config.yaml"
                    if success and is_drive_mounted:
                        msg += "\n☁️ Konfigurasi telah disync ke Google Drive"
                    
                    display(create_status_indicator("success", msg))
                    
                    # Demonstrasi EarlyStopping
                    early_stop = get_widget_children('early_stopping_params')
                    if 'EarlyStopping' in globals() and len(early_stop) > 2:
                        monitor = early_stop[2].value
                        patience = int(early_stop[1].value)
                        logger = deps.get('logger')
                        
                        # Inisialisasi EarlyStopping
                        early_stopping = EarlyStopping(
                            monitor=monitor,
                            patience=patience,
                            mode='max' if monitor != 'val_loss' else 'min',
                            logger=logger
                        )
                        
                        if logger:
                            logger.info(f"🔧 EarlyStopping diinisialisasi: monitor={monitor}, patience={patience}")
                
                # Fallback dengan yaml manual
                if not success:
                    try:
                        import yaml
                        from pathlib import Path
                        Path("configs").mkdir(exist_ok=True)
                        with open("configs/training_config.yaml", "w") as f:
                            yaml.dump(config, f, default_flow_style=False)
                        display(create_status_indicator("success", "✅ Konfigurasi berhasil disimpan"))
                    except Exception as e:
                        display(create_status_indicator("error", f"❌ Gagal menyimpan konfigurasi: {e}"))
            except Exception as e:
                display(create_status_indicator("error", f"❌ Error: {e}"))
    
    # Handler untuk reset config
    def on_reset_configuration(b):
        output = get_component('status_output') or get_component('visualization_output')
        if not output: return
        
        with output:
            clear_output()
            display(create_status_indicator("info", "🔄 Mereset hyperparameter ke default..."))
            
            try:
                # Default values
                default_training = {
                    'epochs': 50, 'batch_size': 16, 'lr0': 0.01, 'lrf': 0.01,
                    'optimizer': 'Adam', 'scheduler': 'cosine', 'momentum': 0.937,
                    'weight_decay': 0.0005, 'early_stopping_patience': 10,
                    'early_stopping_enabled': True, 'early_stopping_monitor': 'val_mAP',
                    'save_best_only': True, 'save_period': 5,
                    'box_loss_weight': 0.05, 'obj_loss_weight': 0.5, 'cls_loss_weight': 0.5,
                    'use_ema': False, 'use_swa': False, 'mixed_precision': True
                }
                
                # Update config
                if config: config['training'] = default_training
                else: config = {'training': default_training}
                
                # Update UI
                update_ui_from_config()
                update_lr_visualization()
                
                display(create_status_indicator("success", "✅ Hyperparameter berhasil direset ke default"))
            except Exception as e:
                display(create_status_indicator("error", f"❌ Error: {e}"))
    
    # Setup event handlers
    try:
        # Scheduler changes
        scheduler = get_component('scheduler_params')
        if scheduler and len(scheduler.children) > 0:
            scheduler.children[0].observe(on_scheduler_change, names='value')
        
        # Early stopping changes
        early_stop = get_component('early_stopping_params')
        if early_stop and len(early_stop.children) > 0:
            early_stop.children[0].observe(on_early_stopping_change, names='value')
        
        # Button handlers
        save_button = get_component('save_button')
        if save_button: save_button.on_click(on_save_configuration)
        
        reset_button = get_component('reset_button')
        if reset_button: reset_button.on_click(on_reset_configuration)
    except Exception as e:
        logger = deps.get('logger')
        if logger: logger.warning(f"⚠️ Error setup handlers: {e}")
    
    # Initialize UI
    try:
        update_ui_from_config()
        update_lr_visualization()
    except Exception as e:
        logger = deps.get('logger')
        if logger: logger.warning(f"⚠️ Error initialize UI: {e}")
    
    # Cleanup function
    def cleanup():
        """Cleanup resources."""
        observer_manager = deps.get('observer_manager')
        if observer_manager:
            observer_manager.unregister_group("hyperparameters_ui")
            
            # Unobserve handlers
            try:
                scheduler = get_component('scheduler_params')
                if scheduler and len(scheduler.children) > 0:
                    scheduler.children[0].unobserve(on_scheduler_change, names='value')
                
                early_stop = get_component('early_stopping_params')
                if early_stop and len(early_stop.children) > 0:
                    early_stop.children[0].unobserve(on_early_stopping_change, names='value')
            except:
                pass
            
            logger = deps.get('logger')
            if logger: logger.info("✅ Hyperparameters handler resources cleaned up")
    
    # Add cleanup
    ui_components['cleanup'] = cleanup
    
    return ui_components