"""
File: smartcash/ui_handlers/hyperparameters.py
Author: Refactor
Deskripsi: Handler untuk komponen UI konfigurasi hyperparameter model SmartCash.
           Memperbaiki chart visualization error dengan menghapus fitur chart.
"""

from IPython.display import display, clear_output, HTML

def setup_hyperparameters_handlers(ui_components, config=None):
    """Setup handlers untuk komponen UI konfigurasi hyperparameter."""
    # Defensive programming - return early if ui_components is None
    if ui_components is None:
        print("⚠️ UI components tidak tersedia")
        return None
        
    # Define simple create_status_indicator function in case utils are not available
    def create_status_indicator(status, message):
        icon = "✅" if status == "success" else "⚠️" if status == "warning" else "❌" if status == "error" else "ℹ️"
        return HTML(f"<div style='margin: 5px 0'>{icon} {message}</div>")
    
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
        
        # Try importing ModelManager silently - don't log warnings
        try:
            from smartcash.handlers.model import ModelManager
            if config: 
                deps['model_manager'] = ModelManager(config, logger=None)
                if deps['logger']:
                    deps['logger'].info("✅ ModelManager tersedia")
        except ImportError:
            # ModelManager is optional, so we don't need to log this
            pass
            
    except ImportError as e:
        if 'status_output' in ui_components:
            with ui_components['status_output']:
                display(create_status_indicator("warning", f"⚠️ Error saat import dependencies: {e}"))
        else:
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
        
        # Helper to safely get widget value with default
        def get_value(widgets, idx, default, attr='value'):
            try:
                return getattr(widgets[idx], attr, default) if 0 <= idx < len(widgets) else default
            except:
                return default
        
        # Basic params
        if basic:
            config['training'].update({
                'epochs': int(get_value(basic, 0, 50)),
                'batch_size': int(get_value(basic, 1, 16)),
                'lr0': float(get_value(basic, 2, 0.01)),
                'optimizer': get_value(basic, 3, 'Adam')
            })
        
        # Scheduler params
        if scheduler:
            config['training'].update({
                'scheduler': get_value(scheduler, 0, 'cosine'),
                'lrf': float(get_value(scheduler, 1, 0.01))
            })
            
            if get_value(scheduler, 2, True, 'disabled') == False:
                config['training'].update({
                    'scheduler_patience': int(get_value(scheduler, 2, 5)),
                    'scheduler_factor': float(get_value(scheduler, 3, 0.1))
                })
        
        # Early stopping params
        if early_stop:
            config['training'].update({
                'early_stopping_enabled': get_value(early_stop, 0, True),
                'early_stopping_patience': int(get_value(early_stop, 1, 10)),
                'early_stopping_monitor': get_value(early_stop, 2, 'val_mAP'),
                'save_best_only': get_value(early_stop, 3, True),
                'save_period': int(get_value(early_stop, 4, 5))
            })
        
        # Advanced params
        if advanced:
            config['training'].update({
                'momentum': float(get_value(advanced, 0, 0.937)),
                'weight_decay': float(get_value(advanced, 1, 0.0005)),
                'use_ema': get_value(advanced, 2, False),
                'use_swa': get_value(advanced, 3, False),
                'mixed_precision': get_value(advanced, 4, True)
            })
        
        # Loss weights
        if loss:
            config['training'].update({
                'box_loss_weight': float(get_value(loss, 0, 0.05)),
                'obj_loss_weight': float(get_value(loss, 1, 0.5)),
                'cls_loss_weight': float(get_value(loss, 2, 0.5))
            })
        
        # Notify if observer available
        observer_manager = deps.get('observer_manager')
        if observer_manager:
            try:
                from smartcash.utils.observer import EventTopics, EventDispatcher
                EventDispatcher.notify(
                    event_type=EventTopics.CONFIG_UPDATED,
                    sender="hyperparameters_handler",
                    config_type="training"
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
                
                # Show success message
                display(create_status_indicator("success", "✅ Hyperparameter berhasil direset ke default"))
                
                # Display information about Learning Rate Schedule text instead of visualization
                display(HTML("""
                <div style="padding: 10px; background-color: #e3f2fd; border-left: 4px solid #2196F3; margin: 10px 0;">
                    <h4 style="margin-top: 0;">📝 Learning Rate Schedule Info</h4>
                    <p>Cosine scheduler memberikan penurunan learning rate yang halus dari nilai awal ke nilai akhir.</p>
                    <p>Nilai learning rate akan menurun mengikuti kurva cosine dari <b>0.01</b> ke <b>0.01 * lrf</b>.</p>
                </div>
                """))
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
        
        # Display text description of LR scheduler instead of visualization
        visualization_output = get_component('visualization_output')
        if visualization_output:
            visualization_output.layout.display = 'block'
            with visualization_output:
                clear_output()
                
                # Get scheduler type from config
                scheduler_type = config.get('training', {}).get('scheduler', 'cosine')
                lr0 = config.get('training', {}).get('lr0', 0.01)
                lrf = config.get('training', {}).get('lrf', 0.01)
                
                scheduler_descriptions = {
                    'cosine': f"Cosine scheduler: Learning rate menurun halus dari {lr0} ke {lr0 * lrf} mengikuti kurva cosine.",
                    'linear': f"Linear scheduler: Learning rate menurun linear dari {lr0} ke {lr0 * lrf}.",
                    'step': "Step scheduler: Learning rate diturunkan dengan faktor tertentu setiap n epoch.",
                    'OneCycleLR': "OneCycle: Learning rate meningkat lalu menurun kembali dengan pola tertentu."
                }
                
                description = scheduler_descriptions.get(scheduler_type, f"Learning rate scheduler: {scheduler_type}")
                
                display(HTML(f"""
                <div style="padding: 10px; background-color: #e3f2fd; border-left: 4px solid #2196F3; margin: 10px 0;">
                    <h4 style="margin-top: 0;">📝 Learning Rate Schedule: {scheduler_type}</h4>
                    <p>{description}</p>
                </div>
                """))
                
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