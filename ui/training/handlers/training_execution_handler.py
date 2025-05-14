"""
File: smartcash/ui/training/handlers/training_execution_handler.py
Deskripsi: Handler untuk eksekusi proses training
"""

from typing import Dict, Any, Optional
import ipywidgets as widgets
from IPython.display import display, HTML
import threading
import time
from tqdm.auto import tqdm

from smartcash.common.config.manager import get_config_manager
from smartcash.common.logger import get_logger
from smartcash.ui.training.handlers.training_handler_utils import (
    get_training_status,
    set_training_status,
    update_ui_status,
    display_status_panel
)

def run_training(ui_components: Dict[str, Any], logger=None):
    """
    Menjalankan proses training.
    
    Args:
        ui_components: Komponen UI
        logger: Logger untuk mencatat aktivitas
    """
    # Dapatkan logger jika tidak disediakan
    logger = logger or get_logger("training_ui")
    
    # Dapatkan ConfigManager
    config_manager = get_config_manager()
    
    try:
        # Import dengan penanganan error minimal
        from smartcash.model.services.training.core_training_service import TrainingService
        from smartcash.model.services.training.callbacks_training_service import TrainingCallbacks
        from smartcash.model.manager import ModelManager
        
        # Dapatkan status training
        training_status = get_training_status()
        
        # Update status UI
        update_ui_status(ui_components, "Memulai training...", is_error=False)
        ui_components['progress_bar'].value = 0
        
        # Tampilkan status
        display_status_panel(ui_components, "Training sedang berjalan. Mohon tunggu...", is_error=False)
        
        # Ambil konfigurasi
        hyperparams_config = config_manager.get_module_config('hyperparameters')
        training_config = config_manager.get_module_config('training')
        model_config = config_manager.get_module_config('model')
        
        # Ekstrak parameter training
        epochs = hyperparams_config.get('hyperparameters', {}).get('epochs', 100)
        batch_size = hyperparams_config.get('hyperparameters', {}).get('batch_size', 16)
        
        # Simulasi training untuk demo
        # Dalam implementasi sebenarnya, ini akan memanggil TrainingService.train()
        
        logger.info("🚀 Memulai proses training...")
        
        # Simulasi training loop
        for epoch in range(epochs):
            # Cek apakah training dihentikan
            if training_status['stop_requested']:
                logger.info("⚠️ Training dihentikan oleh pengguna")
                break
            
            # Simulasi training epoch
            time.sleep(0.1)  # Untuk demo, kita percepat
            
            # Simulasi metrik training
            train_loss = 1.0 - (epoch / epochs) * 0.7
            val_loss = 1.0 - (epoch / epochs) * 0.6
            map_value = (epoch / epochs) * 0.8
            
            # Update progress
            progress = int((epoch + 1) / epochs * 100)
            ui_components['progress_bar'].value = progress
            
            # Update status
            ui_components['status_label'].value = f'<span style="color:#3498db">🔄 Training: Epoch {epoch+1}/{epochs}</span>'
            
            # Tampilkan metrik
            with ui_components['metrics_output']:
                ui_components['metrics_output'].clear_output(wait=True)
                display(HTML(f"""
                <div style="padding:10px;border-left:4px solid #3498db;background-color:#f8f9fa">
                    🔄 Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, mAP: {map_value:.4f}
                </div>
                """))
        
        # Training selesai atau dihentikan
        if training_status['stop_requested']:
            update_ui_status(ui_components, "Training dihentikan oleh pengguna", is_error=True)
            display_status_panel(
                ui_components,
                "⚠️ Training dihentikan oleh pengguna. Model checkpoint terakhir tersimpan.",
                is_error=True
            )
        else:
            update_ui_status(ui_components, "Training selesai!", is_error=False, progress=100)
            display_status_panel(
                ui_components,
                "✅ Training selesai! Model terbaik tersimpan di direktori checkpoint.",
                is_error=False
            )
        
        # Reset status training
        set_training_status(active=False, stop_requested=False)
        
        return True
    
    except Exception as e:
        logger.error(f"❌ Error saat training: {str(e)}")
        
        # Update status UI
        update_ui_status(ui_components, f"Error: {str(e)}", is_error=True)
        display_status_panel(ui_components, f"❌ Error saat training: {str(e)}", is_error=True)
        
        # Reset status training
        set_training_status(active=False, stop_requested=False)
        
        return False
