"""
File: smartcash/ui/training_config/training_strategy_handler.py
Deskripsi: Handler untuk konfigurasi strategi training model dengan ui_helpers
"""

from IPython.display import display, HTML, clear_output
from typing import Dict, Any, Optional

def setup_training_strategy_handlers(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """
    Setup handler untuk komponen UI strategi training.
    
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
    from smartcash.ui.utils.ui_helpers import create_status_indicator, create_info_alert
    
    # Dapatkan logger jika tersedia
    logger = None
    try:
        from smartcash.common.logger import get_logger
        logger = get_logger(ui_components.get('module_name', 'training_strategy'))
    except ImportError:
        pass
    
    # Validasi config
    if config is None:
        config = {}
    
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
    
    # Update config dari UI
    def update_config_from_ui(current_config=None):
        if current_config is None:
            current_config = config
            
        # Ensure training section exists
        if 'training' not in current_config:
            current_config['training'] = {}
            
        # Get augmentation config
        aug_opts = ui_components.get('augmentation_options')
        if aug_opts and hasattr(aug_opts, 'children') and len(aug_opts.children) >= 5:
            current_config['training']['augmentation'] = {
                'enabled': aug_opts.children[0].value,
                'mosaic': aug_opts.children[1].value,
                'fliplr': aug_opts.children[2].value,
                'scale': aug_opts.children[3].value,
                'mixup': 1.0 if aug_opts.children[4].value else 0.0
            }
        
        # Get optimization config
        opt_opts = ui_components.get('optimization_options')
        if opt_opts and hasattr(opt_opts, 'children') and len(opt_opts.children) >= 5:
            current_config['training']['optimization'] = {
                'mixed_precision': opt_opts.children[0].value,
                'ema': opt_opts.children[1].value,
                'swa': opt_opts.children[2].value,
                'lr_schedule': opt_opts.children[3].value,
                'weight_decay': opt_opts.children[4].value
            }
        
        # Get policy config
        policy_opts = ui_components.get('policy_options')
        if policy_opts and hasattr(policy_opts, 'children') and len(policy_opts.children) >= 5:
            current_config['training']['policy'] = {
                'save_best': policy_opts.children[0].value,
                'save_period': policy_opts.children[1].value,
                'early_stopping_patience': policy_opts.children[2].value,
                'validate_every_epoch': policy_opts.children[3].value,
                'log_tensorboard': policy_opts.children[4].value
            }
        
        # Update strategy summary
        update_strategy_summary()
        
        return current_config
    
    # Update UI dari config
    def update_ui_from_config():
        """Update komponen UI dari konfigurasi."""
        if not config or 'training' not in config:
            return
        
        try:
            # Get config sections dengan default yang aman
            aug_config = config['training'].get('augmentation', {})
            opt_config = config['training'].get('optimization', {})
            policy_config = config['training'].get('policy', {})
            
            # Helper untuk update widget value dengan aman
            def safe_update(widget, value):
                try:
                    widget.value = value
                except Exception as e:
                    if logger:
                        logger.debug(f"⚠️ Tidak dapat update widget: {e}")
            
            # Update augmentation options
            aug_opts = ui_components.get('augmentation_options')
            if aug_opts and hasattr(aug_opts, 'children') and len(aug_opts.children) >= 5:
                safe_update(aug_opts.children[0], aug_config.get('enabled', True))
                safe_update(aug_opts.children[1], aug_config.get('mosaic', 0.5))
                safe_update(aug_opts.children[2], aug_config.get('fliplr', 0.5))
                safe_update(aug_opts.children[3], aug_config.get('scale', 0.3))
                safe_update(aug_opts.children[4], aug_config.get('mixup', 0) > 0)
                
            # Update optimization options
            opt_opts = ui_components.get('optimization_options')
            if opt_opts and hasattr(opt_opts, 'children') and len(opt_opts.children) >= 5:
                safe_update(opt_opts.children[0], opt_config.get('mixed_precision', True))
                safe_update(opt_opts.children[1], opt_config.get('ema', True))
                safe_update(opt_opts.children[2], opt_config.get('swa', False))
                safe_update(opt_opts.children[3], opt_config.get('lr_schedule', 'cosine'))
                safe_update(opt_opts.children[4], opt_config.get('weight_decay', 0.01))
                
            # Update policy options
            policy_opts = ui_components.get('policy_options')
            if policy_opts and hasattr(policy_opts, 'children') and len(policy_opts.children) >= 5:
                safe_update(policy_opts.children[0], policy_config.get('save_best', True))
                safe_update(policy_opts.children[1], policy_config.get('save_period', 5))
                safe_update(policy_opts.children[2], policy_config.get('early_stopping_patience', 15))
                safe_update(policy_opts.children[3], policy_config.get('validate_every_epoch', True))
                safe_update(policy_opts.children[4], policy_config.get('log_tensorboard', True))
                
            # Update strategy summary
            update_strategy_summary()
            
        except Exception as e:
            if logger:
                logger.warning(f"⚠️ Error saat update UI: {e}")
    
    # Update strategy summary
    def update_strategy_summary():
        """Create a summary of the current strategy settings."""
        strategy_summary = ui_components.get('strategy_summary')
        if not strategy_summary:
            return
            
        with strategy_summary:
            clear_output(wait=True)
            
            try:
                # Ensure training config exists
                if 'training' not in config:
                    config['training'] = {}
                    
                aug_config = config['training'].get('augmentation', {})
                opt_config = config['training'].get('optimization', {})
                policy_config = config['training'].get('policy', {})
                
                # Gunakan create_info_alert dari ui_helpers
                try:
                    from smartcash.ui.utils.ui_helpers import create_info_alert
                    
                    # Summary untuk augmentation
                    aug_summary = f"""
                    <h5 style="margin-top: 0; color: #2c3e50">🔄 Augmentation</h5>
                    <p><b>Status:</b> {'Enabled' if aug_config.get('enabled', True) else 'Disabled'}</p>
                    <p><b>Mosaic:</b> {aug_config.get('mosaic', 0.5) * 100:.0f}%</p>
                    <p><b>Flip Rate:</b> {aug_config.get('fliplr', 0.5) * 100:.0f}%</p>
                    <p><b>Mixup:</b> {'Enabled' if aug_config.get('mixup', 0) > 0 else 'Disabled'}</p>
                    """
                    
                    # Summary untuk optimization
                    opt_summary = f"""
                    <h5 style="margin-top: 0; color: #2c3e50">⚙️ Optimization</h5>
                    <p><b>Precision:</b> {'Mixed (FP16)' if opt_config.get('mixed_precision', True) else 'Full (FP32)'}</p>
                    <p><b>Scheduler:</b> {opt_config.get('lr_schedule', 'cosine').capitalize()}</p>
                    <p><b>EMA:</b> {'Enabled' if opt_config.get('ema', True) else 'Disabled'}</p>
                    <p><b>SWA:</b> {'Enabled' if opt_config.get('swa', False) else 'Disabled'}</p>
                    """
                    
                    # Summary untuk policy
                    policy_summary = f"""
                    <h5 style="margin-top: 0; color: #2c3e50">📋 Training Policy</h5>
                    <p><b>Save Best:</b> {'Yes' if policy_config.get('save_best', True) else 'No'}</p>
                    <p><b>Save Every:</b> {policy_config.get('save_period', 5)} epochs</p>
                    <p><b>Early Stop After:</b> {policy_config.get('early_stopping_patience', 15)} epochs</p>
                    <p><b>TensorBoard:</b> {'Enabled' if policy_config.get('log_tensorboard', True) else 'Disabled'}</p>
                    """
                    
                    # Tampilkan summary dengan flex layout
                    display(HTML(f"""
                    <div style="background-color: #f8f9fa; padding: 10px; border-radius: 5px; color: #2c3e50">
                        <h4 style="margin-top: 0; color: #2c3e50">📊 Training Strategy Overview</h4>
                        
                        <div style="display: flex; flex-wrap: wrap; gap: 15px;">
                            <div style="flex: 1; min-width: 200px; border: 1px solid #ddd; border-radius: 5px; padding: 10px;">
                                {aug_summary}
                            </div>
                            
                            <div style="flex: 1; min-width: 200px; border: 1px solid #ddd; border-radius: 5px; padding: 10px;">
                                {opt_summary}
                            </div>
                            
                            <div style="flex: 1; min-width: 200px; border: 1px solid #ddd; border-radius: 5px; padding: 10px;">
                                {policy_summary}
                            </div>
                        </div>
                    </div>
                    """))
                    
                    # Tambahkan tip menggunakan create_info_alert
                    display(create_info_alert(
                        "💡 Tip: Cosine scheduler dengan EMA dan mixed precision adalah kombinasi optimal untuk sebagian besar kasus.",
                        "info"
                    ))
                    
                except ImportError:
                    # Fallback manual HTML rendering
                    display(HTML(f"""
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
                    """))
                
            except Exception as e:
                if logger:
                    logger.error(f"❌ Error updating strategy summary: {str(e)}")
                display(HTML(f"<p style='color:red'>❌ Error updating summary: {str(e)}</p>"))
    
    # Handler untuk save button
    def on_save_click(b):
        save_config(
            ui_components,
            config,
            "configs/training_config.yaml",
            update_config_from_ui,
            "Strategi Training"
        )
    
    # Handler untuk reset button
    def on_reset_click(b):
        reset_config(
            ui_components,
            config,
            default_config,
            update_ui_from_config,
            "Strategi Training"
        )
    
    # Register change listeners untuk update summary
    def on_component_change(change):
        if change['name'] != 'value':
            return
        update_config_from_ui()
    
    # Setup event handlers
    try:
        # Register button callbacks
        ui_components['save_button'].on_click(on_save_click)
        ui_components['reset_button'].on_click(on_reset_click)
        
        # Register change listeners untuk semua komponen
        for section in ['augmentation_options', 'optimization_options', 'policy_options']:
            component = ui_components.get(section)
            if component and hasattr(component, 'children'):
                for child in component.children:
                    child.observe(on_component_change, names='value')
    except Exception as e:
        if logger:
            logger.warning(f"⚠️ Error setup handlers: {e}")
    
    # Initialize UI dari config
    update_ui_from_config()
    
    # Cleanup function
    def cleanup():
        """Cleanup resources."""
        try:
            # Unobserve all handlers
            for section in ['augmentation_options', 'optimization_options', 'policy_options']:
                component = ui_components.get(section)
                if component and hasattr(component, 'children'):
                    for child in component.children:
                        child.unobserve(on_component_change, names='value')
                        
            if logger:
                logger.info("✅ Training strategy handlers cleaned up")
        except Exception as e:
            if logger:
                logger.warning(f"⚠️ Error saat cleanup: {e}")
    
    # Add cleanup function
    ui_components['cleanup'] = cleanup
    
    return ui_components