"""
File: smartcash/ui_handlers/training_strategy.py
Author: Refactor
Deskripsi: Handler untuk UI konfigurasi strategi training model SmartCash (optimized).
           Includes defensive programming to handle missing UI components.
"""

from IPython.display import display, HTML, clear_output

def setup_training_strategy_handlers(ui_components, config=None):
    """Setup handlers untuk UI konfigurasi strategi training model."""
    # Early return if ui_components is None
    if ui_components is None:
        print("⚠️ UI components tidak tersedia")
        return None
        
    # Import necessities
    try:
        from smartcash.utils.logger import get_logger
        from smartcash.utils.observer.observer_manager import ObserverManager
        from smartcash.utils.config_manager import get_config_manager
        from smartcash.utils.ui_utils import create_status_indicator
        
        logger = get_logger("training_strategy")
        observer_manager = ObserverManager(auto_register=True)
        config_manager = get_config_manager(logger=logger)
        
        # Ensure observer cleanup from previous sessions
        observer_group = "training_strategy_observers"
        observer_manager.unregister_group(observer_group)
        
    except ImportError as e:
        # Fallback to simple status indicator if utils not available
        def create_status_indicator(status, message):
            icon = "✅" if status == "success" else "⚠️" if status == "warning" else "❌" if status == "error" else "ℹ️"
            return HTML(f"<div style='margin: 5px 0'>{icon} {message}</div>")
            
        # Find and use status_output if available
        if ui_components and 'status_output' in ui_components:
            with ui_components['status_output']:
                display(create_status_indicator("warning", f"⚠️ Beberapa modul tidak tersedia: {str(e)}"))
        else:
            print(f"⚠️ Beberapa modul tidak tersedia: {str(e)}")
        
    # Check if the required components exist in ui_components
    required_components = ['augmentation_options', 'optimization_options', 'policy_options', 
                          'strategy_summary', 'save_button', 'reset_button', 'status_output']
    missing_components = [comp for comp in required_components if comp not in ui_components]
    
    if missing_components:
        if 'status_output' in ui_components:
            with ui_components['status_output']:
                display(create_status_indicator("error", 
                                              f"❌ Komponen UI tidak lengkap: {', '.join(missing_components)}"))
        else:
            print(f"❌ Komponen UI tidak lengkap: {', '.join(missing_components)}")
        return ui_components
    
    # Initialize config if not provided
    if config is None:
        config = {}
    
    if 'training' not in config:
        config['training'] = {}
        
    # Set default values if not in config
    if 'augmentation' not in config['training']:
        config['training']['augmentation'] = {
            'enabled': True,
            'mosaic': 0.5,
            'fliplr': 0.5,
            'scale': 0.3,
            'mixup': 0.0
        }
        
    if 'optimization' not in config['training']:
        config['training']['optimization'] = {
            'mixed_precision': True,
            'ema': True,
            'swa': False,
            'lr_schedule': 'cosine',
            'weight_decay': 0.01
        }
        
    if 'policy' not in config['training']:
        config['training']['policy'] = {
            'save_best': True,
            'save_period': 5,
            'early_stopping_patience': 15,
            'validate_every_epoch': True,
            'log_tensorboard': True
        }
    
    # Fungsi untuk update config dari UI
    def update_config_from_ui():
        """Ambil nilai dari UI dan update config."""
        # Get augmentation config
        aug_opts = ui_components['augmentation_options']
        if aug_opts and hasattr(aug_opts, 'children') and len(aug_opts.children) >= 5:
            aug_enabled = aug_opts.children[0].value
            mosaic_prob = aug_opts.children[1].value
            flip_prob = aug_opts.children[2].value
            scale_jitter = aug_opts.children[3].value
            mixup_enabled = aug_opts.children[4].value
            
            config['training']['augmentation'].update({
                'enabled': aug_enabled,
                'mosaic': mosaic_prob,
                'fliplr': flip_prob,
                'scale': scale_jitter,
                'mixup': 1.0 if mixup_enabled else 0.0
            })
        
        # Get optimization config
        opt_opts = ui_components['optimization_options']
        if opt_opts and hasattr(opt_opts, 'children') and len(opt_opts.children) >= 5:
            mixed_precision = opt_opts.children[0].value
            use_ema = opt_opts.children[1].value
            use_swa = opt_opts.children[2].value
            lr_schedule = opt_opts.children[3].value
            weight_decay = opt_opts.children[4].value
            
            config['training']['optimization'].update({
                'mixed_precision': mixed_precision,
                'ema': use_ema,
                'swa': use_swa,
                'lr_schedule': lr_schedule,
                'weight_decay': weight_decay
            })
        
        # Get policy config
        policy_opts = ui_components['policy_options']
        if policy_opts and hasattr(policy_opts, 'children') and len(policy_opts.children) >= 5:
            save_best = policy_opts.children[0].value
            save_period = policy_opts.children[1].value
            early_stopping = policy_opts.children[2].value
            validate_every_epoch = policy_opts.children[3].value
            log_tensorboard = policy_opts.children[4].value
            
            config['training']['policy'].update({
                'save_best': save_best,
                'save_period': save_period,
                'early_stopping_patience': early_stopping,
                'validate_every_epoch': validate_every_epoch,
                'log_tensorboard': log_tensorboard
            })
        
        # Update strategy summary
        update_strategy_summary()
        
        return config
    
    # Fungsi untuk update UI dari config
    def update_ui_from_config():
        """Update UI components dari config."""
        if not config or 'training' not in config:
            return
            
        # Get config sections
        aug_config = config['training'].get('augmentation', {})
        opt_config = config['training'].get('optimization', {})
        policy_config = config['training'].get('policy', {})
        
        # Helper function to safely set widget value
        def safe_set_value(widget, key, config_dict, default_value):
            if widget and hasattr(widget, 'value') and key in config_dict:
                widget.value = config_dict[key]
            elif widget and hasattr(widget, 'value'):
                widget.value = default_value
        
        # Update augmentation UI
        aug_opts = ui_components['augmentation_options']
        if aug_opts and hasattr(aug_opts, 'children') and len(aug_opts.children) >= 5:
            safe_set_value(aug_opts.children[0], 'enabled', aug_config, True)
            safe_set_value(aug_opts.children[1], 'mosaic', aug_config, 0.5)
            safe_set_value(aug_opts.children[2], 'fliplr', aug_config, 0.5)
            safe_set_value(aug_opts.children[3], 'scale', aug_config, 0.3)
            safe_set_value(aug_opts.children[4], 'mixup', aug_config, False)
            # Special case for mixup which is stored as float but displayed as boolean
            if 'mixup' in aug_config and aug_opts.children[4]:
                aug_opts.children[4].value = aug_config['mixup'] > 0
            
        # Update optimization UI
        opt_opts = ui_components['optimization_options']
        if opt_opts and hasattr(opt_opts, 'children') and len(opt_opts.children) >= 5:
            safe_set_value(opt_opts.children[0], 'mixed_precision', opt_config, True)
            safe_set_value(opt_opts.children[1], 'ema', opt_config, True)
            safe_set_value(opt_opts.children[2], 'swa', opt_config, False)
            safe_set_value(opt_opts.children[3], 'lr_schedule', opt_config, 'cosine')
            safe_set_value(opt_opts.children[4], 'weight_decay', opt_config, 0.01)
            
        # Update policy UI
        policy_opts = ui_components['policy_options']
        if policy_opts and hasattr(policy_opts, 'children') and len(policy_opts.children) >= 5:
            safe_set_value(policy_opts.children[0], 'save_best', policy_config, True)
            safe_set_value(policy_opts.children[1], 'save_period', policy_config, 5)
            safe_set_value(policy_opts.children[2], 'early_stopping_patience', policy_config, 15)
            safe_set_value(policy_opts.children[3], 'validate_every_epoch', policy_config, True)
            safe_set_value(policy_opts.children[4], 'log_tensorboard', policy_config, True)
            
        # Update strategy summary
        update_strategy_summary()
    
    # Update strategy summary
    def update_strategy_summary():
        """Create a summary of the current strategy settings."""
        if not ui_components or 'strategy_summary' not in ui_components:
            return
            
        with ui_components['strategy_summary']:
            clear_output(wait=True)
            
            try:
                if 'training' not in config:
                    config['training'] = {}
                    
                aug_config = config['training'].get('augmentation', {})
                opt_config = config['training'].get('optimization', {})
                policy_config = config['training'].get('policy', {})
                
                # Create summary HTML
                html = f"""
                <div style="background-color: #f8f9fa; padding: 10px; border-radius: 5px; color: #2c3e50">
                    <h4 style="margin-top: 0; color: #2c3e50">📊 Training Strategy Overview</h4>
                    
                    <div style="display: flex; flex-wrap: wrap; gap: 15px;">
                        <div style="flex: 1; min-width: 200px; border: 1px solid #ddd; border-radius: 5px; padding: 10px;">
                            <h5 style="margin-top: 0; color: #2c3e50">🔄 Augmentation</h5>
                            <p><b>Status:</b> {'Enabled' if aug_config.get('enabled', True) else 'Disabled'}</p>
                            <p><b>Mosaic:</b> {aug_config.get('mosaic', 0.5) * 100:.0f}%</p>
                            <p><b>Flip Rate:</b> {aug_config.get('fliplr', 0.5) * 100:.0f}%</p>
                            <p><b>Mixup:</b> {'Enabled' if aug_config.get('mixup', 0) > 0 else 'Disabled'}</p>
                        </div>
                        
                        <div style="flex: 1; min-width: 200px; border: 1px solid #ddd; border-radius: 5px; padding: 10px;">
                            <h5 style="margin-top: 0; color: #2c3e50">⚙️ Optimization</h5>
                            <p><b>Precision:</b> {'Mixed (FP16)' if opt_config.get('mixed_precision', True) else 'Full (FP32)'}</p>
                            <p><b>Scheduler:</b> {opt_config.get('lr_schedule', 'cosine').capitalize()}</p>
                            <p><b>EMA:</b> {'Enabled' if opt_config.get('ema', True) else 'Disabled'}</p>
                            <p><b>SWA:</b> {'Enabled' if opt_config.get('swa', False) else 'Disabled'}</p>
                        </div>
                        
                        <div style="flex: 1; min-width: 200px; border: 1px solid #ddd; border-radius: 5px; padding: 10px;">
                            <h5 style="margin-top: 0; color: #2c3e50">📋 Training Policy</h5>
                            <p><b>Save Best:</b> {'Yes' if policy_config.get('save_best', True) else 'No'}</p>
                            <p><b>Save Every:</b> {policy_config.get('save_period', 5)} epochs</p>
                            <p><b>Early Stop After:</b> {policy_config.get('early_stopping_patience', 15)} epochs</p>
                            <p><b>TensorBoard:</b> {'Enabled' if policy_config.get('log_tensorboard', True) else 'Disabled'}</p>
                        </div>
                    </div>
                    
                    <div style="margin-top: 15px; padding: 10px; background-color: #d1ecf1; border-radius: 5px; color: #0c5460;">
                        <p style="margin: 0;"><b>💡 Tip:</b> Cosine scheduler dengan EMA dan mixed precision adalah kombinasi optimal untuk sebagian besar kasus.</p>
                    </div>
                </div>
                """
                
                display(HTML(html))
                
            except Exception as e:
                if logger:
                    logger.error(f"❌ Error updating strategy summary: {str(e)}")
                display(HTML(f"<p style='color:red'>❌ Error updating summary: {str(e)}</p>"))
    
    # Handler untuk save button
    def on_save_click(b):
        status_output = ui_components.get('status_output')
        if not status_output:
            print("⚠️ Status output tidak tersedia")
            return
            
        with status_output:
            clear_output()
            display(create_status_indicator("info", "🔄 Menyimpan konfigurasi strategi training..."))
            
            try:
                # Update config dari UI
                updated_config = update_config_from_ui()
                
                # Simpan ke file
                if config_manager:
                    success = config_manager.save_config(
                        updated_config, 
                        "configs/training_config.yaml",
                        backup=True
                    )
                    
                    if success:
                        display(create_status_indicator(
                            "success", 
                            "✅ Konfigurasi strategi training berhasil disimpan ke configs/training_config.yaml"
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
                        "✅ Konfigurasi strategi training diupdate dalam memori"
                    ))
                
            except Exception as e:
                display(create_status_indicator("error", f"❌ Error saat menyimpan konfigurasi: {str(e)}"))
    
    # Handler untuk reset button
    def on_reset_click(b):
        status_output = ui_components.get('status_output')
        if not status_output:
            print("⚠️ Status output tidak tersedia")
            return
            
        with status_output:
            clear_output()
            display(create_status_indicator("info", "🔄 Reset konfigurasi ke default..."))
            
            try:
                # Default config values
                default_config = {
                    'training': {
                        'augmentation': {
                            'enabled': True,
                            'mosaic': 0.5,
                            'fliplr': 0.5,
                            'scale': 0.3,
                            'mixup': 0.0
                        },
                        'optimization': {
                            'mixed_precision': True,
                            'ema': True,
                            'swa': False,
                            'lr_schedule': 'cosine',
                            'weight_decay': 0.01
                        },
                        'policy': {
                            'save_best': True,
                            'save_period': 5,
                            'early_stopping_patience': 15,
                            'validate_every_epoch': True,
                            'log_tensorboard': True
                        }
                    }
                }
                
                # Update global config and UI
                if 'training' in config:
                    config['training'] = default_config['training']
                else:
                    config.update(default_config)
                    
                update_ui_from_config()
                
                display(create_status_indicator("success", "✅ Konfigurasi berhasil direset ke default"))
                
            except Exception as e:
                display(create_status_indicator("error", f"❌ Error saat reset konfigurasi: {str(e)}"))
    
    # Listen for changes
    def on_component_change(change):
        if change['name'] != 'value':
            return
        update_config_from_ui()
    
    # Register callbacks if components exist
    if ui_components.get('save_button'):
        ui_components['save_button'].on_click(on_save_click)
    
    if ui_components.get('reset_button'):
        ui_components['reset_button'].on_click(on_reset_click)
    
    # Register change listeners for all components
    for section_name in ['augmentation_options', 'optimization_options', 'policy_options']:
        section = ui_components.get(section_name)
        if section and hasattr(section, 'children'):
            for child in section.children:
                child.observe(on_component_change, names='value')
    
    # Initialize UI from config
    update_ui_from_config()
    
    # Define cleanup function
    def cleanup():
        """Clean up resources."""
        if observer_manager:
            observer_manager.unregister_group(observer_group)
        if logger:
            logger.info("✅ Training strategy handlers cleaned up")
    
    # Add cleanup to UI components
    ui_components['cleanup'] = cleanup
    
    return ui_components