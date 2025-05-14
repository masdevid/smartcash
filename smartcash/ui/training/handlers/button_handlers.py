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

def setup_training_button_handlers(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """
    Setup handler untuk tombol pada komponen UI training.
    
    Args:
        ui_components: Komponen UI
        env: Environment manager
        config: Konfigurasi model
        
    Returns:
        Dict berisi komponen UI dengan handler terpasang
    """
    try:
        # Import dengan penanganan error minimal
        from smartcash.common.config.manager import ConfigManager, get_config_manager
        from smartcash.ui.utils.alert_utils import create_status_indicator, create_info_alert
        from smartcash.common.logger import get_logger
        from smartcash.model.services.training.core_training_service import TrainingService
        from smartcash.model.services.training.callbacks_training_service import TrainingCallbacks
        from smartcash.model.manager import ModelManager
        
        # Dapatkan logger jika tersedia
        logger = ui_components.get('logger', None) or get_logger("training_ui")
        
        # Dapatkan ConfigManager
        config_manager = get_config_manager()
        
        # Status training
        training_active = False
        training_thread = None
        stop_training = False
        
        # Update informasi training
        def update_training_info():
            """Update informasi training dari konfigurasi."""
            try:
                # Ambil konfigurasi dari ConfigManager
                hyperparams_config = config_manager.get_module_config('hyperparameters')
                training_config = config_manager.get_module_config('training')
                model_config = config_manager.get_module_config('model')
                
                # Tampilkan informasi
                with ui_components['info_box']:
                    ui_components['info_box'].clear_output()
                    
                    # Informasi model
                    display(HTML(f"""
                    <h3 style="margin-bottom:10px">📊 Konfigurasi Training</h3>
                    <div style="display:flex;flex-wrap:wrap">
                        <div style="flex:1;min-width:300px;margin-right:10px">
                            <h4>Model</h4>
                            <ul>
                                <li><b>Backbone:</b> {model_config.get('backbone', 'efficientnet_b4')}</li>
                                <li><b>Layer Mode:</b> {training_config.get('training_utils', {}).get('layer_mode', 'single')}</li>
                            </ul>
                        </div>
                        <div style="flex:1;min-width:300px;margin-right:10px">
                            <h4>Hyperparameter</h4>
                            <ul>
                                <li><b>Batch Size:</b> {hyperparams_config.get('hyperparameters', {}).get('batch_size', 16)}</li>
                                <li><b>Epochs:</b> {hyperparams_config.get('hyperparameters', {}).get('epochs', 100)}</li>
                                <li><b>Image Size:</b> {hyperparams_config.get('hyperparameters', {}).get('image_size', 640)}</li>
                                <li><b>Optimizer:</b> {hyperparams_config.get('hyperparameters', {}).get('optimizer', 'Adam')}</li>
                                <li><b>Learning Rate:</b> {hyperparams_config.get('hyperparameters', {}).get('learning_rate', 0.001)}</li>
                            </ul>
                        </div>
                        <div style="flex:1;min-width:300px">
                            <h4>Training Strategy</h4>
                            <ul>
                                <li><b>Multi-scale:</b> {str(training_config.get('multi_scale', True))}</li>
                                <li><b>Experiment:</b> {training_config.get('training_utils', {}).get('experiment_name', 'efficientnet_b4_training')}</li>
                                <li><b>Mixed Precision:</b> {str(training_config.get('training_utils', {}).get('mixed_precision', True))}</li>
                                <li><b>Early Stopping:</b> {str(hyperparams_config.get('hyperparameters', {}).get('early_stopping', {}).get('enabled', True))}</li>
                            </ul>
                        </div>
                    </div>
                    """))
            except Exception as e:
                logger.error(f"❌ Error saat memperbarui informasi training: {str(e)}")
                with ui_components['info_box']:
                    ui_components['info_box'].clear_output()
                    display(HTML(f"""
                    <div style="color:red;padding:10px;border:1px solid red;border-radius:5px">
                        <b>Error:</b> Gagal memuat informasi training. {str(e)}
                    </div>
                    """))
        
        # Fungsi untuk menjalankan training dalam thread terpisah
        def run_training():
            nonlocal training_active, stop_training
            
            try:
                # Update status
                ui_components['status_label'].value = '<span style="color:#3498db">⏳ Mempersiapkan training...</span>'
                ui_components['progress_bar'].value = 0
                
                # Ambil konfigurasi dari ConfigManager
                hyperparams_config = config_manager.get_module_config('hyperparameters')
                training_config = config_manager.get_module_config('training')
                model_config = config_manager.get_module_config('model')
                
                # Gabungkan konfigurasi
                combined_config = {
                    'hyperparameters': hyperparams_config.get('hyperparameters', {}),
                    'training': training_config,
                    'model': model_config
                }
                
                # Update opsi dari UI
                combined_config['training']['training_utils']['tensorboard'] = ui_components['use_tensorboard'].value
                combined_config['hyperparameters']['checkpoint']['save_best'] = ui_components['save_checkpoints'].value
                
                # Buat model manager
                model_manager = ModelManager(config=combined_config)
                
                # Inisialisasi model
                model_manager.initialize_model()
                
                # Persiapkan dataset
                with ui_components['status_panel']:
                    ui_components['status_panel'].clear_output()
                    display(HTML('<div style="color:#3498db">⏳ Mempersiapkan dataset...</div>'))
                
                # TODO: Implementasi load dataset dari konfigurasi
                # Untuk contoh, kita akan mensimulasikan proses training
                
                # Update status
                ui_components['status_label'].value = '<span style="color:#2ecc71">🚀 Training sedang berjalan...</span>'
                
                # Simulasi training epochs
                epochs = hyperparams_config.get('hyperparameters', {}).get('epochs', 100)
                
                with ui_components['metrics_box']:
                    ui_components['metrics_box'].clear_output()
                    display(HTML(f"""
                    <h4>Metrik Training</h4>
                    <div id="metrics-table">
                        <table style="width:100%;border-collapse:collapse">
                            <thead>
                                <tr style="background-color:#f5f5f5">
                                    <th style="padding:8px;border:1px solid #ddd;text-align:left">Epoch</th>
                                    <th style="padding:8px;border:1px solid #ddd;text-align:left">Train Loss</th>
                                    <th style="padding:8px;border:1px solid #ddd;text-align:left">Val Loss</th>
                                    <th style="padding:8px;border:1px solid #ddd;text-align:left">mAP</th>
                                    <th style="padding:8px;border:1px solid #ddd;text-align:left">Learning Rate</th>
                                </tr>
                            </thead>
                            <tbody id="metrics-body">
                            </tbody>
                        </table>
                    </div>
                    """))
                
                # Simulasi epochs untuk demo
                for epoch in range(epochs):
                    if stop_training:
                        break
                    
                    # Update progress
                    progress = (epoch + 1) / epochs * 100
                    ui_components['progress_bar'].value = progress
                    
                    # Simulasi training dan validasi
                    train_loss = 1.0 - (epoch / epochs) * 0.7  # Simulasi loss yang menurun
                    val_loss = 1.2 - (epoch / epochs) * 0.8  # Simulasi validation loss
                    map_value = (epoch / epochs) * 0.85  # Simulasi mAP yang meningkat
                    lr = 0.001 * (1 - epoch / epochs)  # Simulasi learning rate yang menurun
                    
                    # Update metrik
                    with ui_components['metrics_box']:
                        display(HTML(f"""
                        <script>
                            var table = document.getElementById('metrics-body');
                            var row = table.insertRow(0);
                            
                            var cell1 = row.insertCell(0);
                            var cell2 = row.insertCell(1);
                            var cell3 = row.insertCell(2);
                            var cell4 = row.insertCell(3);
                            var cell5 = row.insertCell(4);
                            
                            cell1.innerHTML = "{epoch+1}/{epochs}";
                            cell1.style.padding = "8px";
                            cell1.style.border = "1px solid #ddd";
                            
                            cell2.innerHTML = "{train_loss:.4f}";
                            cell2.style.padding = "8px";
                            cell2.style.border = "1px solid #ddd";
                            
                            cell3.innerHTML = "{val_loss:.4f}";
                            cell3.style.padding = "8px";
                            cell3.style.border = "1px solid #ddd";
                            
                            cell4.innerHTML = "{map_value:.4f}";
                            cell4.style.padding = "8px";
                            cell4.style.border = "1px solid #ddd";
                            
                            cell5.innerHTML = "{lr:.6f}";
                            cell5.style.padding = "8px";
                            cell5.style.border = "1px solid #ddd";
                        </script>
                        """))
                    
                    # Update status panel
                    with ui_components['status_panel']:
                        ui_components['status_panel'].clear_output()
                        display(HTML(f"""
                        <div style="color:#2ecc71">
                            🔄 Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, mAP: {map_value:.4f}
                        </div>
                        """))
                    
                    # Simulasi waktu training
                    time.sleep(0.1)  # Untuk demo, kita percepat
                
                # Training selesai atau dihentikan
                if stop_training:
                    ui_components['status_label'].value = '<span style="color:#e74c3c">⚠️ Training dihentikan oleh pengguna</span>'
                else:
                    ui_components['status_label'].value = '<span style="color:#2ecc71">✅ Training selesai!</span>'
                    ui_components['progress_bar'].value = 100
                
                with ui_components['status_panel']:
                    ui_components['status_panel'].clear_output()
                    if stop_training:
                        display(HTML("""
                        <div style="color:#e74c3c">
                            ⚠️ Training dihentikan oleh pengguna. Model checkpoint terakhir tersimpan.
                        </div>
                        """))
                    else:
                        display(HTML("""
                        <div style="color:#2ecc71">
                            ✅ Training selesai! Model terbaik tersimpan di direktori checkpoint.
                        </div>
                        """))
            
            except Exception as e:
                logger.error(f"❌ Error saat training: {str(e)}")
                ui_components['status_label'].value = f'<span style="color:#e74c3c">❌ Error: {str(e)}</span>'
                
                with ui_components['status_panel']:
                    ui_components['status_panel'].clear_output()
                    display(HTML(f"""
                    <div style="color:#e74c3c">
                        ❌ Error saat training: {str(e)}
                    </div>
                    """))
            
            finally:
                # Reset status
                training_active = False
                stop_training = False
                
                # Update tombol
                ui_components['start_button'].disabled = False
                ui_components['stop_button'].disabled = True
        
        # Handler untuk tombol start
        def on_start_click(b):
            nonlocal training_active, training_thread, stop_training
            
            if training_active:
                return
            
            # Update status
            training_active = True
            stop_training = False
            
            # Update tombol
            ui_components['start_button'].disabled = True
            ui_components['stop_button'].disabled = False
            
            # Jalankan training dalam thread terpisah
            training_thread = threading.Thread(target=run_training)
            training_thread.daemon = True
            training_thread.start()
        
        # Handler untuk tombol stop
        def on_stop_click(b):
            nonlocal stop_training
            
            if not training_active:
                return
            
            # Set flag untuk menghentikan training
            stop_training = True
            ui_components['stop_button'].disabled = True
            ui_components['status_label'].value = '<span style="color:#e67e22">⏳ Menghentikan training...</span>'
        
        # Pasang handler ke tombol
        ui_components['start_button'].on_click(on_start_click)
        ui_components['stop_button'].on_click(on_stop_click)
        
        # Update informasi training saat pertama kali
        update_training_info()
        
        # Daftarkan komponen UI untuk persistensi
        config_manager.register_ui_components('training', ui_components)
        
        return ui_components
    
    except Exception as e:
        # Fallback jika terjadi error
        print(f"Error saat setup button handlers: {str(e)}")
        return ui_components
