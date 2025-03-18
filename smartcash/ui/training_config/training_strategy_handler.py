"""
File: smartcash/ui/training_config/training_strategy_handler.py
Deskripsi: Handler yang dioptimalkan untuk konfigurasi strategi training model
"""

from IPython.display import display, HTML, clear_output
from typing import Dict, Any, Optional

def setup_training_strategy_handlers(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """Setup handler untuk komponen UI strategi training."""
    try:
        # Import dengan penanganan error minimal
        from smartcash.ui.training_config.config_handler import save_config, reset_config, get_config_manager
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
        
        # Default config (lebih ringkas)
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
        
        # Update config dari UI (diringkas)
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
        
        # Update UI dari config (diringkas)
        def update_ui_from_config():
            """Update komponen UI dari konfigurasi."""
            if not config or 'training' not in config:
                return
            
            try:
                # Get config sections
                aug_config = config['training'].get('augmentation', {})
                opt_config = config['training'].get('optimization', {})
                policy_config = config['training'].get('policy', {})
                
                # Helper untuk update widget value dengan aman
                def safe_update(widget, value):
                    try:
                        widget.value = value
                    except Exception as e:
                        if logger:
                            logger.debug(f"⚠️ Error update widget: {e}")
                
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
                    logger.warning(f"⚠️ Error update UI: {e}")
        
        # Update strategy summary (lebih ringkas)
        def update_strategy_summary():
            """Buat rangkuman strategi yang ringkas."""
            strategy_summary = ui_components.get('strategy_summary')
            if not strategy_summary:
                return
                
            with strategy_summary:
                clear_output(wait=True)
                
                try:
                    # Config sections
                    aug_config = config.get('training', {}).get('augmentation', {})
                    opt_config = config.get('training', {}).get('optimization', {})
                    policy_config = config.get('training', {}).get('policy', {})
                    
                    # Gunakan grid layout yang sederhana
                    html = """
                    <div style="background-color:#f8f9fa; padding:10px; border-radius:5px; color:#2c3e50">
                        <h4 style="margin-top:0">📊 Training Strategy Overview</h4>
                        <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(200px, 1fr)); gap:10px">
                    """
                    
                    # Augmentation
                    html += """
                        <div style="border:1px solid #ddd; border-radius:5px; padding:10px">
                            <h5 style="margin-top:0">🔄 Augmentation</h5>
                            <ul style="margin:0; padding-left:20px">
                    """
                    html += f"<li><b>Status:</b> {'Enabled' if aug_config.get('enabled', True) else 'Disabled'}</li>"
                    html += f"<li><b>Mosaic:</b> {aug_config.get('mosaic', 0.5) * 100:.0f}%</li>"
                    html += f"<li><b>Flip Rate:</b> {aug_config.get('fliplr', 0.5) * 100:.0f}%</li>"
                    html += f"<li><b>Mixup:</b> {'Enabled' if aug_config.get('mixup', 0) > 0 else 'Disabled'}</li>"
                    html += "</ul></div>"
                    
                    # Optimization
                    html += """
                        <div style="border:1px solid #ddd; border-radius:5px; padding:10px">
                            <h5 style="margin-top:0">⚙️ Optimization</h5>
                            <ul style="margin:0; padding-left:20px">
                    """
                    html += f"<li><b>Precision:</b> {'Mixed (FP16)' if opt_config.get('mixed_precision', True) else 'Full (FP32)'}</li>"
                    html += f"<li><b>Scheduler:</b> {opt_config.get('lr_schedule', 'cosine').capitalize()}</li>"
                    html += f"<li><b>EMA:</b> {'Enabled' if opt_config.get('ema', True) else 'Disabled'}</li>"
                    html += f"<li><b>SWA:</b> {'Enabled' if opt_config.get('swa', False) else 'Disabled'}</li>"
                    html += "</ul></div>"
                    
                    # Policy
                    html += """
                        <div style="border:1px solid #ddd; border-radius:5px; padding:10px">
                            <h5 style="margin-top:0">📋 Training Policy</h5>
                            <ul style="margin:0; padding-left:20px">
                    """
                    html += f"<li><b>Save Best:</b> {'Yes' if policy_config.get('save_best', True) else 'No'}</li>"
                    html += f"<li><b>Save Every:</b> {policy_config.get('save_period', 5)} epochs</li>"
                    html += f"<li><b>Early Stop:</b> {policy_config.get('early_stopping_patience', 15)} epochs</li>"
                    html += f"<li><b>TensorBoard:</b> {'Enabled' if policy_config.get('log_tensorboard', True) else 'Disabled'}</li>"
                    html += "</ul></div>"
                    
                    # Tip box
                    html += """
                        </div>
                        <div style="margin-top:10px; padding:8px; background-color:#d1ecf1; border-radius:4px; color:#0c5460">
                            <p style="margin:0"><b>💡 Tip:</b> Cosine scheduler dengan EMA dan mixed precision optimal untuk sebagian besar kasus.</p>
                        </div>
                    </div>
                    """
                    
                    display(HTML(html))
                    
                except Exception as e:
                    if logger:
                        logger.error(f"❌ Error update summary: {e}")
                    display(HTML(f"<p style='color:red'>❌ Error update summary: {str(e)}</p>"))
        
        # Handler buttons sederhana
        def on_save_click(b):
            save_config(ui_components, config, "configs/training_config.yaml", update_config_from_ui, "Strategi Training")
        
        def on_reset_click(b):
            reset_config(ui_components, config, default_config, update_ui_from_config, "Strategi Training")
        
        # Handler untuk perubahan komponen
        def on_component_change(change):
            if change['name'] == 'value':
                update_config_from_ui()
        
        # Register handlers
        ui_components['save_button'].on_click(on_save_click)
        ui_components['reset_button'].on_click(on_reset_click)
        
        # Register observer untuk semua komponen UI
        for section in ['augmentation_options', 'optimization_options', 'policy_options']:
            component = ui_components.get(section)
            if component and hasattr(component, 'children'):
                for child in component.children:
                    child.observe(on_component_change, names='value')
        
        # Inisialisasi UI dari config
        update_ui_from_config()
        
        # Cleanup function yang ringkas
        def cleanup():
            try:
                for section in ['augmentation_options', 'optimization_options', 'policy_options']:
                    component = ui_components.get(section)
                    if component and hasattr(component, 'children'):
                        for child in component.children:
                            child.unobserve(on_component_change, names='value')
                            
                if logger:
                    logger.info("✅ Training strategy handlers cleaned up")
            except Exception as e:
                if logger:
                    logger.warning(f"⚠️ Error cleanup: {e}")
        
        # Tambahkan cleanup function
        ui_components['cleanup'] = cleanup
        
    except Exception as e:
        # Fallback sederhana jika terjadi error
        if 'status' in ui_components:
            with ui_components['status']:
                display(HTML(f"<p style='color:red'>❌ Error setup strategy handler: {str(e)}</p>"))
        else:
            print(f"❌ Error setup strategy handler: {str(e)}")
    
    return ui_components