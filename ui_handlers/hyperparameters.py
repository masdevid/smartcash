"""
File: smartcash/ui_handlers/hyperparameters.py
Author: Generated
Deskripsi: Handler untuk komponen UI konfigurasi hyperparameter model SmartCash.
"""

from IPython.display import display, clear_output, HTML
import threading
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

from smartcash.utils.ui_utils import create_status_indicator
from smartcash.utils.early_stopping import EarlyStopping

def setup_hyperparameters_handlers(ui_components, config=None):
    """
    Setup handlers untuk komponen UI konfigurasi hyperparameter.
    
    Args:
        ui_components: Dictionary berisi komponen UI
        config: Konfigurasi yang akan digunakan (optional)
        
    Returns:
        Dictionary UI components yang sudah diupdate dengan handler
    """
    # Inisialisasi dependencies dengan penanganan error yang baik
    logger = None
    config_manager = None
    observer_manager = None
    env_manager = None
    model_manager = None
    
    try:
        from smartcash.utils.logging_factory import LoggingFactory
        from smartcash.utils.config_manager import get_config_manager
        from smartcash.utils.observer.observer_manager import ObserverManager
        from smartcash.utils.environment_manager import EnvironmentManager
        from smartcash.handlers.model import ModelManager
        from smartcash.utils.observer import EventTopics, EventDispatcher
        
        logger = LoggingFactory.get_logger("hyperparameters")
        config_manager = get_config_manager(logger=logger)
        observer_manager = ObserverManager(auto_register=True)
        env_manager = EnvironmentManager(logger=logger)
        
        # Cleanup existing observers
        observer_manager.unregister_group("hyperparameters_ui")
        
    except ImportError as e:
        with ui_components['status_output']:
            display(create_status_indicator(
                "warning", f"⚠️ Beberapa modul tidak tersedia - fallback ke mode dasar: {str(e)}"
            ))
    
    # Load config jika belum ada
    if not config or not isinstance(config, dict):
        try:
            if config_manager:
                config = config_manager.load_config("configs/training_config.yaml")
                
                if not config or 'training' not in config:
                    # Buat default config jika tidak ada
                    config = {
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
            else:
                # Fallback config
                config = {
                    'training': {
                        'epochs': 50,
                        'batch_size': 16,
                        'lr0': 0.01
                    }
                }
        except Exception as e:
            with ui_components['status_output']:
                display(create_status_indicator(
                    "error", f"❌ Error saat memuat konfigurasi: {str(e)}"
                ))
    
    # Fungsi untuk update UI berdasarkan config
    def update_ui_from_config():
        """Update semua komponen UI dari konfigurasi yang ada."""
        if not config or 'training' not in config:
            return
        
        training_config = config['training']
        basic_params = ui_components['basic_params'].children
        scheduler_params = ui_components['scheduler_params'].children
        early_stop_params = ui_components['early_stopping_params'].children
        advanced_params = ui_components['advanced_params'].children
        loss_params = ui_components['loss_params'].children
        
        # Update basic params
        try:
            # Epochs
            if 'epochs' in training_config:
                basic_params[0].value = min(max(training_config['epochs'], basic_params[0].min), basic_params[0].max)
            
            # Batch size
            if 'batch_size' in training_config:
                basic_params[1].value = min(max(training_config['batch_size'], basic_params[1].min), basic_params[1].max)
            
            # Learning rate
            if 'lr0' in training_config:
                lr = training_config['lr0']
                # Convert to log slider value
                basic_params[2].value = lr
            
            # Optimizer
            if 'optimizer' in training_config and training_config['optimizer'] in basic_params[3].options:
                basic_params[3].value = training_config['optimizer']
            
            # Scheduler
            if 'scheduler' in training_config and training_config['scheduler'] in scheduler_params[0].options:
                scheduler_params[0].value = training_config['scheduler']
            
            # Final LR
            if 'lrf' in training_config:
                scheduler_params[1].value = min(max(training_config['lrf'], scheduler_params[1].min), scheduler_params[1].max)
            
            # Early stopping
            if 'early_stopping_enabled' in training_config:
                early_stop_params[0].value = training_config['early_stopping_enabled']
            
            if 'early_stopping_patience' in training_config:
                early_stop_params[1].value = min(max(training_config['early_stopping_patience'], 
                                                  early_stop_params[1].min), early_stop_params[1].max)
            
            if 'early_stopping_monitor' in training_config and training_config['early_stopping_monitor'] in early_stop_params[2].options:
                early_stop_params[2].value = training_config['early_stopping_monitor']
            
            if 'save_best_only' in training_config:
                early_stop_params[3].value = training_config['save_best_only']
            
            if 'save_period' in training_config:
                early_stop_params[4].value = min(max(training_config['save_period'], 
                                                early_stop_params[4].min), early_stop_params[4].max)
            
            # Advanced params
            if 'momentum' in training_config:
                advanced_params[0].value = min(max(training_config['momentum'], 
                                            advanced_params[0].min), advanced_params[0].max)
            
            if 'weight_decay' in training_config:
                advanced_params[1].value = training_config['weight_decay']
            
            if 'use_ema' in training_config:
                advanced_params[2].value = training_config['use_ema']
            
            if 'use_swa' in training_config:
                advanced_params[3].value = training_config['use_swa']
            
            if 'mixed_precision' in training_config:
                advanced_params[4].value = training_config['mixed_precision']
            
            # Loss weights
            if 'box_loss_weight' in training_config:
                loss_params[0].value = min(max(training_config['box_loss_weight'], 
                                         loss_params[0].min), loss_params[0].max)
            
            if 'obj_loss_weight' in training_config:
                loss_params[1].value = min(max(training_config['obj_loss_weight'], 
                                         loss_params[1].min), loss_params[1].max)
            
            if 'cls_loss_weight' in training_config:
                loss_params[2].value = min(max(training_config['cls_loss_weight'], 
                                         loss_params[2].min), loss_params[2].max)
            
        except Exception as e:
            if logger:
                logger.error(f"❌ Error saat update UI: {str(e)}")
            else:
                print(f"❌ Error saat update UI: {str(e)}")
    
    # Handler untuk toggle komponen scheduler berdasarkan pilihan
    def on_scheduler_change(change):
        """Toggle komponen scheduler berdasarkan tipe yang dipilih."""
        if change['name'] != 'value':
            return
            
        scheduler_type = change['new']
        scheduler_params = ui_components['scheduler_params'].children
        
        # Enable/disable komponen berdasarkan jenis scheduler
        if scheduler_type == 'step':
            scheduler_params[2].disabled = False  # Patience
            scheduler_params[3].disabled = False  # Factor
        else:
            scheduler_params[2].disabled = True
            scheduler_params[3].disabled = True
            
        # Update visualization
        update_lr_visualization()
    
    # Handler untuk toggle early stopping komponen
    def on_early_stopping_change(change):
        """Toggle early stopping komponen."""
        if change['name'] != 'value':
            return
            
        enabled = change['new']
        early_stop_params = ui_components['early_stopping_params'].children
        
        # Enable/disable komponen berdasarkan checkbox
        for i in range(1, 3):  # Patience dan monitor
            early_stop_params[i].disabled = not enabled
    
    # Fungsi untuk update config dari UI
    def update_config_from_ui():
        """Update konfigurasi dari nilai komponen UI."""
        if not config:
            return {}
            
        if 'training' not in config:
            config['training'] = {}
            
        # Extract parameter values from UI components
        basic_params = ui_components['basic_params'].children
        scheduler_params = ui_components['scheduler_params'].children
        early_stop_params = ui_components['early_stopping_params'].children
        advanced_params = ui_components['advanced_params'].children
        loss_params = ui_components['loss_params'].children
        
        # Update training config from UI
        training_config = {
            # Basic params
            'epochs': int(basic_params[0].value),
            'batch_size': int(basic_params[1].value),
            'lr0': float(basic_params[2].value),
            'optimizer': basic_params[3].value,
            
            # Scheduler params
            'scheduler': scheduler_params[0].value,
            'lrf': float(scheduler_params[1].value),
            'scheduler_patience': int(scheduler_params[2].value) if not scheduler_params[2].disabled else 5,
            'scheduler_factor': float(scheduler_params[3].value) if not scheduler_params[3].disabled else 0.1,
            
            # Early stopping params
            'early_stopping_enabled': early_stop_params[0].value,
            'early_stopping_patience': int(early_stop_params[1].value),
            'early_stopping_monitor': early_stop_params[2].value,
            'save_best_only': early_stop_params[3].value,
            'save_period': int(early_stop_params[4].value),
            
            # Advanced params
            'momentum': float(advanced_params[0].value),
            'weight_decay': float(advanced_params[1].value),
            'use_ema': advanced_params[2].value,
            'use_swa': advanced_params[3].value, 
            'mixed_precision': advanced_params[4].value,
            
            # Loss weights
            'box_loss_weight': float(loss_params[0].value),
            'obj_loss_weight': float(loss_params[1].value),
            'cls_loss_weight': float(loss_params[2].value)
        }
        
        # Update config
        config['training'].update(training_config)
        
        # Notify about config update if observer manager available
        if observer_manager:
            try:
                EventDispatcher.notify(
                    event_type=EventTopics.CONFIG_UPDATED,
                    sender="hyperparameters_handler",
                    config_type="training",
                    update_time=str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                )
            except Exception as e:
                if logger:
                    logger.debug(f"Event notification error: {str(e)}")
                    
        return config
    
    # Simpan konfigurasi
    def save_configuration():
        """Simpan konfigurasi hyperparameter ke file."""
        with ui_components['status_output']:
            clear_output()
            display(create_status_indicator("info", "🔄 Menyimpan konfigurasi..."))
            
            try:
                # Update config dari UI
                update_config_from_ui()
                
                # Simpan config
                if config_manager:
                    success = config_manager.save_config(
                        config, 
                        "configs/training_config.yaml",
                        backup=True,
                        sync_to_drive=(env_manager and env_manager.is_drive_mounted)
                    )
                    
                    if success:
                        display(create_status_indicator(
                            "success", "✅ Konfigurasi berhasil disimpan ke configs/training_config.yaml"
                        ))
                        
                        if env_manager and env_manager.is_drive_mounted:
                            display(create_status_indicator(
                                "info", "☁️ Konfigurasi telah disync ke Google Drive"
                            ))
                    else:
                        display(create_status_indicator(
                            "error", "❌ Gagal menyimpan konfigurasi"
                        ))
                else:
                    # Fallback manual save
                    try:
                        import yaml
                        from pathlib import Path
                        
                        # Create configs directory if not exists
                        Path("configs").mkdir(exist_ok=True)
                        
                        with open("configs/training_config.yaml", "w") as f:
                            yaml.dump(config, f, default_flow_style=False)
                        
                        display(create_status_indicator(
                            "success", "✅ Konfigurasi berhasil disimpan manual ke configs/training_config.yaml"
                        ))
                    except Exception as e:
                        display(create_status_indicator(
                            "error", f"❌ Gagal menyimpan konfigurasi manual: {str(e)}"
                        ))
            
            except Exception as e:
                display(create_status_indicator(
                    "error", f"❌ Error: {str(e)}"
                ))
    
    # Reset ke default
    def reset_to_default():
        """Reset semua parameter ke nilai default."""
        with ui_components['status_output']:
            clear_output()
            display(create_status_indicator("info", "🔄 Mereset hyperparameter ke default..."))
            
            try:
                # Default config
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
                
                # Update config
                if config and 'training' in config:
                    config['training'] = default_config['training']
                else:
                    config = default_config
                
                # Update UI
                update_ui_from_config()
                
                # Visualize
                update_lr_visualization()
                
                display(create_status_indicator(
                    "success", "✅ Hyperparameter berhasil direset ke default"
                ))
            
            except Exception as e:
                display(create_status_indicator(
                    "error", f"❌ Error: {str(e)}"
                ))
    
    # Visualisasi learning rate
    def update_lr_visualization():
        """Visualisasikan learning rate schedule."""
        basic_params = ui_components['basic_params'].children
        scheduler_params = ui_components['scheduler_params'].children
        
        # Extract params
        epochs = int(basic_params[0].value)
        initial_lr = float(basic_params[2].value)
        final_lr = float(scheduler_params[1].value)
        scheduler_type = scheduler_params[0].value
        
        with ui_components['visualization_output']:
            clear_output()
            
            plt.figure(figsize=(10, 6))
            xs = np.linspace(0, epochs-1, epochs)
            
            if scheduler_type == 'cosine':
                # Cosine annealing
                lrs = [initial_lr * (1 + np.cos(np.pi * x / epochs)) / 2 for x in xs]
            elif scheduler_type == 'linear':
                # Linear decay
                lrs = [initial_lr - (initial_lr - final_lr) * (x / (epochs-1)) for x in xs]
            elif scheduler_type == 'step':
                # Step decay
                patience = int(scheduler_params[2].value)
                factor = float(scheduler_params[3].value)
                lrs = []
                current_lr = initial_lr
                
                for i in range(epochs):
                    if i > 0 and i % patience == 0:
                        current_lr *= factor
                    lrs.append(current_lr)
            elif scheduler_type == 'exp':
                # Exponential decay
                gamma = np.exp(np.log(final_lr / initial_lr) / epochs)
                lrs = [initial_lr * (gamma ** x) for x in xs]
            elif scheduler_type == 'OneCycleLR':
                # One Cycle LR
                half_epochs = epochs // 2
                # First half: increase LR
                first_half = [initial_lr + (10*initial_lr - initial_lr) * (x / half_epochs) for x in range(half_epochs)]
                # Second half: decrease LR to very low
                second_half = [10*initial_lr * (1 - x / half_epochs) ** 1.5 for x in range(half_epochs + epochs % 2)]
                lrs = first_half + second_half
            else:
                # Constant LR
                lrs = [initial_lr] * epochs
            
            plt.plot(xs, lrs, 'b-', linewidth=2)
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.title(f'Learning Rate Schedule: {scheduler_type}')
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.show()
            
            html = f"""
            <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-top: 10px;">
                <h4>📝 Learning Rate Schedule Analysis</h4>
                <ul>
                    <li><b>Initial LR:</b> {initial_lr}</li>
                    <li><b>Final LR:</b> {final_lr}</li>
                    <li><b>Scheduler:</b> {scheduler_type}</li>
                    <li><b>Epochs:</b> {epochs}</li>
                </ul>
                <p><em>Grafik menunjukkan perubahan learning rate selama training untuk membantu
                model konvergen ke minimum global.</em></p>
            </div>
            """
            display(HTML(html))
        
        # Show visualization area
        ui_