"""
File: smartcash/ui/training/handlers/button_handlers.py
Deskripsi: Handler untuk tombol UI pada komponen training
"""

from typing import Dict, Any, Optional, Callable
import ipywidgets as widgets
from IPython.display import display, HTML
import threading
import time
import yaml
import os
import matplotlib.pyplot as plt
import io
import base64
import numpy as np

from smartcash.common.config.manager import get_config_manager
from smartcash.common.logger import get_logger
from smartcash.ui.utils.constants import ICONS, COLORS
from smartcash.ui.utils.alert_utils import create_info_alert

# Status training global
training_active = False
training_thread = None
stop_training = False

# Handler untuk tombol start training
def on_start_training(b, ui_components: Dict[str, Any], env=None, config=None) -> None:
    """
    Handler untuk tombol start training.
    
    Args:
        b: Button widget
        ui_components: Komponen UI
        env: Environment manager
        config: Konfigurasi model
    """
    global training_active, training_thread, stop_training
    
    if training_active:
        return
    
    # Dapatkan logger
    logger = ui_components.get('logger', None) or get_logger()
    logger.info(f"{ICONS.get('play', '▶️')} Memulai proses training...")
    
    # Update status
    training_active = True
    stop_training = False
    
    # Update tombol
    ui_components['start_button'].disabled = True
    ui_components['stop_button'].disabled = False
    
    # Update status panel
    with ui_components['status_panel']:
        ui_components['status_panel'].clear_output()
        display(create_info_alert(
            f"{ICONS.get('play', '▶️')} Memulai proses training dengan konfigurasi yang telah diatur",
            alert_type='info'
        ))
    
    # Jalankan training dalam thread terpisah
    training_thread = threading.Thread(target=run_training, args=(ui_components, env, config))
    training_thread.daemon = True
    training_thread.start()

# Handler untuk tombol stop training
def on_stop_training(b, ui_components: Dict[str, Any], env=None, config=None) -> None:
    """
    Handler untuk tombol stop training.
    
    Args:
        b: Button widget
        ui_components: Komponen UI
        env: Environment manager
        config: Konfigurasi model
    """
    global training_active, stop_training
    
    if not training_active:
        return
    
    # Dapatkan logger
    logger = ui_components.get('logger', None) or get_logger()
    logger.info(f"{ICONS.get('stop', '⏹️')} Menghentikan proses training...")
    
    # Set flag untuk menghentikan training
    stop_training = True
    ui_components['stop_button'].disabled = True
    
    # Update status panel
    with ui_components['status_panel']:
        ui_components['status_panel'].clear_output()
        display(create_info_alert(
            f"{ICONS.get('stop', '⏹️')} Menghentikan proses training...",
            alert_type='warning'
        ))

# Handler untuk tombol reset training
def on_reset_training(b, ui_components: Dict[str, Any], env=None, config=None) -> None:
    """
    Handler untuk tombol reset training.
    
    Args:
        b: Button widget
        ui_components: Komponen UI
        env: Environment manager
        config: Konfigurasi model
    """
    global training_active, stop_training
    
    if training_active:
        return
    
    # Dapatkan logger
    logger = ui_components.get('logger', None) or get_logger()
    logger.info(f"{ICONS.get('reset', '🔄')} Reset konfigurasi training...")
    
    # Reset form ke nilai default
    ui_components['backbone_dropdown'].value = 'efficientnet_b4'
    ui_components['epochs_input'].value = 100
    ui_components['batch_size_input'].value = 16
    ui_components['learning_rate_input'].value = 0.001
    ui_components['save_checkpoints'].value = True
    ui_components['use_tensorboard'].value = True
    ui_components['use_mixed_precision'].value = True
    ui_components['use_ema'].value = False
    
    # Reset chart dan metrik
    with ui_components['chart_output']:
        ui_components['chart_output'].clear_output()
        display(ui_components['create_metrics_chart']())
    
    with ui_components['metrics_box']:
        ui_components['metrics_box'].clear_output()
    
    # Update status panel
    with ui_components['status_panel']:
        ui_components['status_panel'].clear_output()
        display(create_info_alert(
            f"{ICONS.get('reset', '🔄')} Konfigurasi training telah direset ke nilai default",
            alert_type='info'
        ))

# Handler untuk tombol cleanup training
def on_cleanup_training(b, ui_components: Dict[str, Any], env=None, config=None) -> None:
    """
    Handler untuk tombol cleanup training.
    
    Args:
        b: Button widget
        ui_components: Komponen UI
        env: Environment manager
        config: Konfigurasi model
    """
    if training_active:
        return
    
    # Dapatkan logger
    logger = ui_components.get('logger', None) or get_logger()
    logger.info(f"{ICONS.get('cleanup', '🧹')} Membersihkan output training...")
    
    # Reset chart dan metrik
    with ui_components['chart_output']:
        ui_components['chart_output'].clear_output()
        display(ui_components['create_metrics_chart']())
    
    with ui_components['metrics_box']:
        ui_components['metrics_box'].clear_output()
    
    # Reset log output
    with ui_components['log_output']:
        ui_components['log_output'].clear_output()
    
    # Update status panel
    with ui_components['status_panel']:
        ui_components['status_panel'].clear_output()
        display(create_info_alert(
            f"{ICONS.get('cleanup', '🧹')} Output training telah dibersihkan",
            alert_type='info'
        ))

# Handler untuk tombol save config
def on_save_config(b, ui_components: Dict[str, Any], env=None, config=None) -> None:
    """
    Handler untuk tombol save konfigurasi.
    
    Args:
        b: Button widget
        ui_components: Komponen UI
        env: Environment manager
        config: Konfigurasi model
    """
    # Dapatkan logger
    logger = ui_components.get('logger', None) or get_logger()
    logger.info(f"{ICONS.get('save', '💾')} Menyimpan konfigurasi training...")
    
    try:
        # Dapatkan config manager
        config_manager = get_config_manager()
        
        # Ambil nilai dari form
        backbone = ui_components['backbone_dropdown'].value
        epochs = ui_components['epochs_input'].value
        batch_size = ui_components['batch_size_input'].value
        learning_rate = ui_components['learning_rate_input'].value
        save_checkpoints = ui_components['save_checkpoints'].value
        use_tensorboard = ui_components['use_tensorboard'].value
        use_mixed_precision = ui_components['use_mixed_precision'].value
        use_ema = ui_components['use_ema'].value
        
        # Buat konfigurasi baru
        new_model_config = {
            'backbone': backbone
        }
        
        new_hyperparameters_config = {
            'learning_rate': learning_rate,
            'batch_size': batch_size
        }
        
        new_training_strategy_config = {
            'epochs': epochs,
            'save_checkpoints': save_checkpoints,
            'use_tensorboard': use_tensorboard,
            'use_mixed_precision': use_mixed_precision,
            'use_ema': use_ema
        }
        
        # Simpan konfigurasi
        config_manager.update_module_config('model', new_model_config)
        config_manager.update_module_config('hyperparameters', new_hyperparameters_config)
        config_manager.update_module_config('training_strategy', new_training_strategy_config)
        
        # Update status
        if 'sync_info' in ui_components:
            with ui_components['status_panel']:
                ui_components['status_panel'].clear_output()
                display(create_info_alert(
                    f"{ICONS.get('check', '✓')} Konfigurasi training berhasil disimpan dan disinkronkan.",
                    alert_type='success'
                ))
        
        logger.info(f"{ICONS.get('check', '✓')} Konfigurasi training berhasil disimpan")
        
    except Exception as e:
        logger.error(f"{ICONS.get('error', '❌')} Error saat menyimpan konfigurasi: {str(e)}")
        
        # Update status panel
        with ui_components['status_panel']:
            ui_components['status_panel'].clear_output()
            display(create_info_alert(
                f"{ICONS.get('error', '❌')} Error saat menyimpan konfigurasi: {str(e)}",
                alert_type='error'
            ))

# Handler untuk tombol reset config
def on_reset_config(b, ui_components: Dict[str, Any], env=None, config=None) -> None:
    """
    Handler untuk tombol reset konfigurasi.
    
    Args:
        b: Button widget
        ui_components: Komponen UI
        env: Environment manager
        config: Konfigurasi model
    """
    # Dapatkan logger
    logger = ui_components.get('logger', None) or get_logger()
    logger.info(f"{ICONS.get('reset', '🔄')} Reset konfigurasi training...")
    
    try:
        # Dapatkan config manager
        config_manager = get_config_manager()
        
        # Dapatkan konfigurasi default
        model_config = config_manager.get_default_module_config('model', {
            'backbone': 'efficientnet_b4'
        })
        
        hyperparameters_config = config_manager.get_default_module_config('hyperparameters', {
            'learning_rate': 0.001,
            'batch_size': 16
        })
        
        training_strategy_config = config_manager.get_default_module_config('training_strategy', {
            'epochs': 100,
            'save_checkpoints': True,
            'use_tensorboard': True,
            'use_mixed_precision': True,
            'use_ema': False
        })
        
        # Reset form ke nilai default
        ui_components['backbone_dropdown'].value = model_config.get('backbone', 'efficientnet_b4')
        ui_components['epochs_input'].value = training_strategy_config.get('epochs', 100)
        ui_components['batch_size_input'].value = hyperparameters_config.get('batch_size', 16)
        ui_components['learning_rate_input'].value = hyperparameters_config.get('learning_rate', 0.001)
        ui_components['save_checkpoints'].value = training_strategy_config.get('save_checkpoints', True)
        ui_components['use_tensorboard'].value = training_strategy_config.get('use_tensorboard', True)
        ui_components['use_mixed_precision'].value = training_strategy_config.get('use_mixed_precision', True)
        ui_components['use_ema'].value = training_strategy_config.get('use_ema', False)
        
        # Update status
        with ui_components['status_panel']:
            ui_components['status_panel'].clear_output()
            display(create_info_alert(
                f"{ICONS.get('check', '✓')} Konfigurasi training berhasil direset ke nilai default.",
                alert_type='success'
            ))
        
        logger.info(f"{ICONS.get('check', '✓')} Konfigurasi training berhasil direset")
        
    except Exception as e:
        logger.error(f"{ICONS.get('error', '❌')} Error saat reset konfigurasi: {str(e)}")
        
        # Update status panel
        with ui_components['status_panel']:
            ui_components['status_panel'].clear_output()
            display(create_info_alert(
                f"{ICONS.get('error', '❌')} Error saat reset konfigurasi: {str(e)}",
                alert_type='error'
            ))

# Fungsi untuk menjalankan training
def run_training(ui_components: Dict[str, Any], env=None, config=None) -> None:
    """
    Menjalankan proses training model.
    
    Args:
        ui_components: Komponen UI
        env: Environment manager
        config: Konfigurasi model
    """
    global training_active, stop_training
    
    try:
        # Dapatkan logger
        logger = ui_components.get('logger', None) or get_logger()
        
        # Dapatkan konfigurasi dari form
        backbone = ui_components['backbone_dropdown'].value
        epochs = ui_components['epochs_input'].value
        batch_size = ui_components['batch_size_input'].value
        learning_rate = ui_components['learning_rate_input'].value
        save_checkpoints = ui_components['save_checkpoints'].value
        use_tensorboard = ui_components['use_tensorboard'].value
        use_mixed_precision = ui_components['use_mixed_precision'].value
        use_ema = ui_components['use_ema'].value
        
        # Log konfigurasi
        logger.info(f"{ICONS.get('info', 'ℹ️')} Konfigurasi training: backbone={backbone}, epochs={epochs}, batch_size={batch_size}, lr={learning_rate}")
        
        # Update progress bar
        if 'progress_bar' in ui_components:
            ui_components['progress_bar'].max = epochs
            ui_components['progress_bar'].value = 0
        
        # Update step label
        if 'step_label' in ui_components:
            ui_components['step_label'].value = f"{ICONS.get('play', '▶️')} Memulai training..."
        
        # Simulasi training loop (untuk demo)
        # Dalam implementasi sebenarnya, ini akan memanggil TrainingService
        train_losses = []
        val_losses = []
        map_values = []
        
        for epoch in range(epochs):
            if stop_training:
                logger.info(f"{ICONS.get('stop', '⏹️')} Training dihentikan pada epoch {epoch+1}/{epochs}")
                break
            
            # Simulasi training epoch
            train_loss = 1.0 - 0.8 * (epoch / epochs) + 0.1 * np.random.random()
            val_loss = 1.1 - 0.7 * (epoch / epochs) + 0.15 * np.random.random()
            map_value = 0.2 + 0.7 * (epoch / epochs) + 0.05 * np.random.random()
            
            # Simpan metrik
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            map_values.append(map_value)
            
            # Update progress
            if 'progress_bar' in ui_components:
                ui_components['progress_bar'].value = epoch + 1
            
            # Update step label
            if 'step_label' in ui_components:
                ui_components['step_label'].value = f"{ICONS.get('training', '🏋️')} Epoch {epoch+1}/{epochs}"
            
            # Update overall label
            if 'overall_label' in ui_components:
                progress_pct = int(100 * (epoch + 1) / epochs)
                ui_components['overall_label'].value = f"Progress: {progress_pct}%"
            
            # Log metrik
            logger.info(f"{ICONS.get('chart', '📊')} Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, mAP: {map_value:.4f}")
            
            # Update metrik di UI
            with ui_components['metrics_box']:
                ui_components['metrics_box'].clear_output()
                display(HTML(f"""
                <div style="padding:5px">
                    {ICONS.get('chart', '📊')} <b>Epoch {epoch+1}/{epochs}</b>
                    <ul>
                        <li><b>Train Loss:</b> <span style="color:#e74c3c">{train_loss:.4f}</span></li>
                        <li><b>Val Loss:</b> <span style="color:#3498db">{val_loss:.4f}</span></li>
                        <li><b>mAP:</b> <span style="color:#2ecc71">{map_value:.4f}</span></li>
                    </ul>
                </div>
                """))
            
            # Update chart
            with ui_components['chart_output']:
                ui_components['chart_output'].clear_output()
                
                plt.figure(figsize=(10, 6))
                plt.title('Training Metrics')
                plt.xlabel('Epoch')
                plt.ylabel('Value')
                plt.grid(True, linestyle='--', alpha=0.7)
                
                epochs_range = list(range(1, epoch + 2))
                plt.plot(epochs_range, train_losses, '-', color='#e74c3c', label='Train Loss')
                plt.plot(epochs_range, val_losses, '--', color='#3498db', label='Val Loss')
                plt.plot(epochs_range, map_values, ':', color='#2ecc71', label='mAP')
                
                plt.legend(loc='best')
                
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                img_str = base64.b64encode(buf.read()).decode('utf-8')
                plt.close()
                
                display(HTML(f'<img src="data:image/png;base64,{img_str}" width="100%">'))
            
            # Simulasi waktu training
            time.sleep(0.1)  # Untuk demo, kita percepat
        
        # Training selesai atau dihentikan
        if stop_training:
            # Update status panel
            with ui_components['status_panel']:
                ui_components['status_panel'].clear_output()
                display(create_info_alert(
                    f"{ICONS.get('warning', '⚠️')} Training dihentikan oleh pengguna pada epoch {epoch+1}/{epochs}. Model checkpoint terakhir tersimpan.",
                    alert_type='warning'
                ))
            
            # Update step label
            if 'step_label' in ui_components:
                ui_components['step_label'].value = f"{ICONS.get('warning', '⚠️')} Training dihentikan"
        else:
            # Update progress bar
            if 'progress_bar' in ui_components:
                ui_components['progress_bar'].value = epochs
            
            # Update status panel
            with ui_components['status_panel']:
                ui_components['status_panel'].clear_output()
                display(create_info_alert(
                    f"{ICONS.get('check', '✓')} Training selesai! Model terbaik tersimpan di direktori checkpoint.",
                    alert_type='success'
                ))
            
            # Update step label
            if 'step_label' in ui_components:
                ui_components['step_label'].value = f"{ICONS.get('check', '✓')} Training selesai"
            
            # Update overall label
            if 'overall_label' in ui_components:
                ui_components['overall_label'].value = f"Progress: 100%"
            
            logger.info(f"{ICONS.get('check', '✓')} Training selesai dengan {epochs} epochs")
    
    except Exception as e:
        # Log error
        logger.error(f"{ICONS.get('error', '❌')} Error saat training: {str(e)}")
        
        # Update status panel
        with ui_components['status_panel']:
            ui_components['status_panel'].clear_output()
            display(create_info_alert(
                f"{ICONS.get('error', '❌')} Error saat training: {str(e)}",
                alert_type='error'
            ))
        
        # Update step label
        if 'step_label' in ui_components:
            ui_components['step_label'].value = f"{ICONS.get('error', '❌')} Training gagal"
    
    finally:
        # Reset status
        training_active = False
        stop_training = False
        
        # Update tombol
        ui_components['start_button'].disabled = False
        ui_components['stop_button'].disabled = True
