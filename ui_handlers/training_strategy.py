"""
File: smartcash/ui_handlers/training_strategy.py
Author: Refactor
Deskripsi: Handler untuk UI konfigurasi strategi training model SmartCash.
"""

from IPython.display import display, HTML, clear_output

def setup_training_strategy_handlers(ui_components, config=None):
    """Setup handlers untuk UI konfigurasi strategi training model."""
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
        # Ensure ui_components has the necessary components
        if ui_components and 'status_output' in ui_components:
            with ui_components['status_output']:
                display(create_status_indicator("warning", f"⚠️ Beberapa modul tidak tersedia: {str(e)}"))
        return ui_components
    
    # Initialize config if not provided
    if config is None:
        config = {}
    
    # Fungsi untuk update config dari UI
    def update_config_from_ui():
        """Ambil nilai dari UI dan update config."""
        # Get augmentation config
        aug_opts = ui_components['augmentation_options']
        aug_enabled = aug_opts.children[0].value
        mosaic_prob = aug_opts.children[1].value
        flip_prob = aug_opts.children[2].value
        scale_jitter = aug_opts.children[3].value
        mixup_enabled = aug_opts.children[4].value
        
        # Get optimization config
        opt_opts = ui_components['optimization_options']
        mixed_precision = opt_opts.children[0].value
        use_ema = opt_opts.children[1].value
        use_swa = opt_opts.children[2].value
        lr_schedule = opt_opts.children[3].value
        weight_decay = opt_opts.children[4].value
        
        # Get policy config
        policy_opts = ui_components['policy_options']
        save_best = policy_opts.children[0].value
        save_period = policy_opts.children[1].value
        early_stopping = policy_opts.children[2].value
        validate_every_epoch = policy_opts.children[3].value
        log_tensorboard = policy_opts.children[4].value
        
        # Ensure training config exists
        if 'training' not in config:
            config['training'] = {}
            
        # Update config dictionary
        config['training']['augmentation'] = {
            'enabled': aug_enabled,
            'mosaic': mosaic_prob,
            'fliplr': flip_prob,
            'scale': scale_jitter,
            'mixup': 1.0 if mixup_enabled else 0.0
        }
        
        config['training']['optimization'] = {
            'mixed_precision': mixed_precision,
            'ema': use_ema,
            'swa': use_swa,
            'lr_schedule': lr_schedule,
            'weight_decay': weight_decay
        }
        
        config['training']['policy'] = {
            'save_best': save_best,
            'save_period': save_period,
            'early_stopping_patience': early_stopping,
            'validate_every_epoch': validate_every_epoch,
            'log_tensorboard': log_tensorboard
        }
        
        # Update strategy summary
        update_strategy_summary()
        
        return config
    
    # Fungsi untuk update UI dari config
    def update_ui_from_config():
        """Update UI components dari config."""
        if not config or 'training' not in config:
            return
            
        # Get config sections with safe defaults
        aug_config = config['training'].get('augmentation', {})
        opt_config = config['training'].get('optimization', {})
        policy_config = config['training'].get('policy', {})
        
        # Helper for safely updating widget values
        def safe_update_value(widget, value):
            if widget is None:
                return
            try:
                widget.value = value
            except Exception as e:
                if logger:
                    logger.debug(f"⚠️ Tidak dapat mengupdate widget: {e}")
        
        # Update augmentation UI safely
        aug_opts = ui_components['augmentation_options']
        if aug_opts is not None and hasattr(aug_opts, 'children') and len(aug_opts.children) >= 5:
            safe_update_value(aug_opts.children[0], aug_config.get('enabled', True))
            safe_update_value(aug_opts.children[1], aug_config.get('mosaic', 0.5))
            safe_update_value(aug_opts.children[2], aug_config.get('fliplr', 0.5))
            safe_update_value(aug_opts.children[3], aug_config.get('scale', 0.3))
            safe_update_value(aug_opts.children[4], aug_config.get('mixup', 0) > 0)
            
        # Update optimization UI safely
        opt_opts = ui_components['optimization_options']
        if opt_opts is not None and hasattr(opt_opts, 'children') and len(opt_opts.children) >= 5:
            safe_update_value(opt_opts.children[0], opt_config.get('mixed_precision', True))
            safe_update_value(opt_opts.children[1], opt_config.get('ema', True))
            safe_update_value(opt_opts.children[2], opt_config.get('swa', False))
            safe_update_value(opt_opts.children[3], opt_config.get('lr_schedule', 'cosine'))
            safe_update_value(opt_opts.children[4], opt_config.get('weight_decay', 0.01))
            
        # Update policy UI safely
        policy_opts = ui_components['policy_options']
        if policy_opts is not None and hasattr(policy_opts, 'children') and len(policy_opts.children) >= 5:
            safe_update_value(policy_opts.children[0], policy_config.get('save_best', True))
            safe_update_value(policy_opts.children[1], policy_config.get('save_period', 5))
            safe_update_value(policy_opts.children[2], policy_config.get('early_stopping_patience', 15))
            safe_update_value(policy_opts.children[3], policy_config.get('validate_every_epoch', True))
            safe_update_value(policy_opts.children[4], policy_config.get('log_tensorboard', True))
            
        # Update strategy summary
        update_strategy_summary()
    
    # Update strategy summary
    def update_strategy_summary():
        """Create a summary of the current strategy settings."""
        if 'strategy_summary' not in ui_components or ui_components['strategy_summary'] is None:
            return
            
        with ui_components['strategy_summary']:
            clear_output(wait=True)
            
            try:
                # Ensure training config exists
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
        with ui_components['status_output']:
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
        with ui_components['status_output']:
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
    
    # Register change listeners for all components
    def on_component_change(change):
        if change['name'] != 'value':
            return
        update_config_from_ui()
    
    # Register callbacks
    ui_components['save_button'].on_click(on_save_click)
    ui_components['reset_button'].on_click(on_reset_click)
    
    # Register change listeners for all components
    for section in ['augmentation_options', 'optimization_options', 'policy_options']:
        component = ui_components[section]
        for child in component.children:
            child.observe(on_component_change, names='value')
    
    # Initialize UI from config
    update_ui_from_config()
    
    # Define cleanup function
    def cleanup():
        """Clean up resources."""
        if observer_manager:
            observer_manager.unregister_group("training_strategy_observers")
        if logger:
            logger.info("✅ Training strategy handlers cleaned up")
    
    # Add cleanup to UI components
    ui_components['cleanup'] = cleanup
    
    return ui_components