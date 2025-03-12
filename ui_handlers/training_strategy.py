"""
File: smartcash/ui_handlers/training_strategy.py
Author: Refactored
Deskripsi: Handler untuk UI konfigurasi strategi training model SmartCash.
"""

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display, clear_output, HTML
import yaml
import json
from pathlib import Path

from smartcash.utils.ui_utils import create_status_indicator

def setup_training_strategy_handlers(ui_components, config=None):
    """Setup handlers untuk UI konfigurasi strategi training model."""
    # Inisialisasi dependencies
    logger = None
    observer_manager = None
    config_manager = None
    
    try:
        from smartcash.utils.logger import get_logger
        from smartcash.utils.observer.observer_manager import ObserverManager
        from smartcash.utils.config_manager import get_config_manager
        
        logger = get_logger("training_strategy")
        observer_manager = ObserverManager(auto_register=True)
        config_manager = get_config_manager(logger=logger)
        
        # Load config jika belum ada
        if not config or not isinstance(config, dict):
            config = config_manager.load_config(
                filename="configs/training_config.yaml",
                fallback_to_pickle=True
            ) or {
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
            
            # Simpan config baru jika belum ada
            if config_manager:
                config_manager.save_config(config, "configs/training_config.yaml")
        
    except ImportError as e:
        if logger:
            logger.warning(f"⚠️ Beberapa modul tidak tersedia: {str(e)}")
        else:
            print(f"⚠️ Beberapa modul tidak tersedia: {str(e)}")
    
    # Kelompok observer
    observer_group = "training_strategy_observers"
    
    # Reset observer group
    if observer_manager:
        observer_manager.unregister_group(observer_group)
    
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
        
        # Update config
        if 'training' not in config:
            config['training'] = {}
            
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
        
        # Update augmentation UI
        aug_opts = ui_components['augmentation_options']
        if 'enabled' in aug_config:
            aug_opts.children[0].value = aug_config['enabled']
        if 'mosaic' in aug_config:
            aug_opts.children[1].value = aug_config['mosaic']
        if 'fliplr' in aug_config:
            aug_opts.children[2].value = aug_config['fliplr']
        if 'scale' in aug_config:
            aug_opts.children[3].value = aug_config['scale']
        if 'mixup' in aug_config:
            aug_opts.children[4].value = aug_config['mixup'] > 0
            
        # Update optimization UI
        opt_opts = ui_components['optimization_options']
        if 'mixed_precision' in opt_config:
            opt_opts.children[0].value = opt_config['mixed_precision']
        if 'ema' in opt_config:
            opt_opts.children[1].value = opt_config['ema']
        if 'swa' in opt_config:
            opt_opts.children[2].value = opt_config['swa']
        if 'lr_schedule' in opt_config:
            opt_opts.children[3].value = opt_config['lr_schedule']
        if 'weight_decay' in opt_config:
            opt_opts.children[4].value = opt_config['weight_decay']
            
        # Update policy UI
        policy_opts = ui_components['policy_options']
        if 'save_best' in policy_config:
            policy_opts.children[0].value = policy_config['save_best']
        if 'save_period' in policy_config:
            policy_opts.children[1].value = policy_config['save_period']
        if 'early_stopping_patience' in policy_config:
            policy_opts.children[2].value = policy_config['early_stopping_patience']
        if 'validate_every_epoch' in policy_config:
            policy_opts.children[3].value = policy_config['validate_every_epoch']
        if 'log_tensorboard' in policy_config:
            policy_opts.children[4].value = policy_config['log_tensorboard']
    
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
                
                # Visualize strategy
                visualize_strategy()
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
                
                # Update UI from default
                if 'training' in config:
                    config['training'] = default_config['training']
                else:
                    config.update(default_config)
                    
                update_ui_from_config()
                
                display(create_status_indicator("success", "✅ Konfigurasi berhasil direset ke default"))
                
                # Visualize default strategy
                visualize_strategy()
                
            except Exception as e:
                display(create_status_indicator("error", f"❌ Error saat reset konfigurasi: {str(e)}"))
    
    # Visualize training strategy
    def visualize_strategy():
        """Visualisasi strategi training untuk membantu pemahaman."""
        try:
            with ui_components['visualization_output']:
                clear_output(wait=True)
                
                # Get updated config from UI
                cfg = update_config_from_ui()
                if not cfg or 'training' not in cfg:
                    return
                
                # Creating a 2x2 visualization layout
                fig, axs = plt.subplots(2, 2, figsize=(14, 10))
                
                # 1. Learning Rate Schedule
                ax1 = axs[0, 0]
                epochs = np.arange(0, 50)
                initial_lr = 0.01
                
                lr_schedule = cfg['training']['optimization']['lr_schedule']
                if lr_schedule == 'cosine':
                    lrs = initial_lr * (1 + np.cos(np.pi * epochs / 50)) / 2
                elif lr_schedule == 'step':
                    lrs = np.array([initial_lr * (0.1 ** (e // 15)) for e in epochs])
                elif lr_schedule == 'linear':
                    lrs = initial_lr * (1 - epochs / 50)
                else:  # constant
                    lrs = np.ones_like(epochs) * initial_lr
                
                ax1.plot(epochs, lrs)
                ax1.set_title('Learning Rate Schedule')
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('Learning Rate')
                ax1.grid(True, linestyle='--', alpha=0.7)
                
                # 2. Augmentation Stats
                ax2 = axs[0, 1]
                aug_cfg = cfg['training']['augmentation']
                aug_types = ['Mosaic', 'Flip', 'Scale', 'Mixup']
                aug_values = [
                    aug_cfg['mosaic'] if aug_cfg['enabled'] else 0,
                    aug_cfg['fliplr'] if aug_cfg['enabled'] else 0,
                    aug_cfg['scale'] if aug_cfg['enabled'] else 0,
                    aug_cfg['mixup'] if aug_cfg['enabled'] else 0
                ]
                bars = ax2.bar(aug_types, aug_values, color='skyblue')
                
                # Add values on top of bars
                for bar in bars:
                    height = bar.get_height()
                    ax2.annotate(f'{height:.2f}',
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 3),
                                textcoords="offset points",
                                ha='center', va='bottom')
                
                ax2.set_title('Augmentation Strategies')
                ax2.set_ylim(0, 1.1)
                ax2.grid(True, axis='y', linestyle='--', alpha=0.7)
                
                # 3. Optimization Techniques
                ax3 = axs[1, 0]
                opt_cfg = cfg['training']['optimization']
                tech_names = ['Mixed Precision', 'EMA', 'SWA']
                tech_values = [
                    1 if opt_cfg['mixed_precision'] else 0,
                    1 if opt_cfg['ema'] else 0,
                    1 if opt_cfg['swa'] else 0
                ]
                tech_colors = ['green' if v else 'red' for v in tech_values]
                
                bars = ax3.bar(tech_names, tech_values, color=tech_colors)
                ax3.set_title('Optimization Techniques')
                ax3.set_ylim(0, 1.1)
                ax3.set_yticks([0, 1])
                ax3.set_yticklabels(['Disabled', 'Enabled'])
                
                # Add text on top of bars
                for bar in bars:
                    height = bar.get_height()
                    text = "Enabled" if height > 0.5 else "Disabled"
                    ax3.annotate(text,
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 3),
                                textcoords="offset points",
                                ha='center', va='bottom')
                
                # 4. Training Policy
                ax4 = axs[1, 1]
                policy_cfg = cfg['training']['policy']
                
                # Create text summary instead of chart
                policy_text = "\n".join([
                    f"Save Best Model: {'✅' if policy_cfg['save_best'] else '❌'}",
                    f"Save Every: {policy_cfg['save_period']} epochs",
                    f"Early Stopping: {policy_cfg['early_stopping_patience']} epochs",
                    f"Validation: {'Every epoch' if policy_cfg['validate_every_epoch'] else 'Periodic'}",
                    f"TensorBoard: {'✅' if policy_cfg['log_tensorboard'] else '❌'}"
                ])
                
                # Remove all axes elements
                ax4.axis('off')
                # Add text box
                ax4.text(0.5, 0.5, policy_text, 
                        ha='center', va='center', 
                        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5),
                        fontsize=12)
                ax4.set_title('Training Policy')
                
                plt.tight_layout()
                display(plt.gcf())
                plt.close()
                
                # Display overall summary
                aug_enabled = aug_cfg['enabled']
                opt_summary = ", ".join([tech for tech, enabled in zip(
                    ["Mixed Precision", "EMA", "SWA"], 
                    [opt_cfg['mixed_precision'], opt_cfg['ema'], opt_cfg['swa']]
                ) if enabled])
                
                summary_html = f"""
                <div style="background-color: #f8f9fa; padding: 10px; border-radius: 5px; margin-top: 10px;">
                    <h4>📊 Training Strategy Summary</h4>
                    <ul>
                        <li><b>Augmentation:</b> {"Enabled" if aug_enabled else "Disabled"}</li>
                        <li><b>Optimizer Enhancements:</b> {opt_summary}</li>
                        <li><b>Learning Rate Schedule:</b> {lr_schedule.capitalize()}</li>
                        <li><b>Early Stopping Patience:</b> {policy_cfg['early_stopping_patience']} epochs</li>
                        <li><b>Weight Decay:</b> {opt_cfg['weight_decay']}</li>
                    </ul>
                </div>
                """
                display(HTML(summary_html))
                
        except Exception as e:
            if logger:
                logger.error(f"❌ Error saat visualisasi strategi: {str(e)}")
            display(HTML(f"<p style='color:red'>❌ Error saat visualisasi: {str(e)}</p>"))
    
    # Register callbacks
    ui_components['save_button'].on_click(on_save_click)
    ui_components['reset_button'].on_click(on_reset_click)
    
    # Update UI from config on init
    update_ui_from_config()
    
    # Initial visualization
    visualize_strategy()
    
    # Cleanup function
    def cleanup():
        """Bersihkan resources saat keluar dari scope."""
        if observer_manager:
            observer_manager.unregister_group(observer_group)
    
    # Add cleanup to UI components
    ui_components['cleanup'] = cleanup
    
    return ui_components