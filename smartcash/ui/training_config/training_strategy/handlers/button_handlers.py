"""
File: smartcash/ui/training_config/training_strategy/handlers/button_handlers.py
Deskripsi: Handler untuk tombol UI pada komponen strategi pelatihan
"""

from typing import Dict, Any, Optional, Callable
import ipywidgets as widgets
from IPython.display import display, HTML

def setup_training_strategy_button_handlers(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """
    Setup handler untuk tombol pada komponen UI strategi pelatihan.
    
    Args:
        ui_components: Komponen UI
        env: Environment manager
        config: Konfigurasi model
        
    Returns:
        Dict berisi komponen UI dengan handler terpasang
    """
    try:
        # Import dengan penanganan error minimal
        from smartcash.ui.training_config.config_handler import save_config, reset_config
        from smartcash.ui.utils.alert_utils import create_status_indicator
        
        # Dapatkan logger jika tersedia
        logger = ui_components.get('logger', None)
        
        # Validasi config
        if config is None: config = {}
        
        # Default config (sudah dihapus komponen yang tidak digunakan)
        default_config = {
            'training': {
                'batch_size': 16,
                'epochs': 100,
                'image_size': 640,
                'workers': 4,
                'val_frequency': 1,
                'early_stopping': True,
                'patience': 10
            }
        }
        
        # Update config dari UI
        def update_config_from_ui(current_config=None):
            if current_config is None: current_config = config
            
            # Update config dari nilai UI
            if 'training' not in current_config:
                current_config['training'] = {}
                
            current_config['training']['batch_size'] = ui_components['batch_size'].value
            current_config['training']['epochs'] = ui_components['epochs'].value
            current_config['training']['image_size'] = ui_components['image_size'].value
            current_config['training']['workers'] = ui_components['workers'].value
            current_config['training']['val_frequency'] = ui_components['val_frequency'].value
            current_config['training']['early_stopping'] = ui_components['early_stopping'].value
            current_config['training']['patience'] = ui_components['patience'].value
            
            # Update info strategi pelatihan
            update_training_strategy_info()
            
            return current_config
        
        # Update UI dari config
        def update_ui_from_config():
            if not config or 'training' not in config: return
            
            try:
                if 'batch_size' in config['training']:
                    ui_components['batch_size'].value = config['training']['batch_size']
                if 'epochs' in config['training']:
                    ui_components['epochs'].value = config['training']['epochs']
                if 'image_size' in config['training']:
                    ui_components['image_size'].value = config['training']['image_size']
                if 'workers' in config['training']:
                    ui_components['workers'].value = config['training']['workers']
                if 'val_frequency' in config['training']:
                    ui_components['val_frequency'].value = config['training']['val_frequency']
                if 'early_stopping' in config['training']:
                    ui_components['early_stopping'].value = config['training']['early_stopping']
                if 'patience' in config['training']:
                    ui_components['patience'].value = config['training']['patience']
                
                # Update info strategi pelatihan
                update_training_strategy_info()
                
                if logger: logger.info("✅ UI strategi pelatihan diperbarui dari config")
            except Exception as e:
                if logger: logger.error(f"❌ Error update UI: {e}")
        
        # Update informasi strategi pelatihan
        def update_training_strategy_info():
            try:
                # Dapatkan nilai dari UI
                batch_size = ui_components['batch_size'].value
                epochs = ui_components['epochs'].value
                image_size = ui_components['image_size'].value
                early_stopping = ui_components['early_stopping'].value
                val_frequency = ui_components['val_frequency'].value
                patience = ui_components['patience'].value
                
                # Estimasi jumlah iterasi
                # Asumsikan dataset berisi 1000 gambar (ganti dengan jumlah sebenarnya jika diketahui)
                dataset_size = 1000
                val_split = 0.2  # Default value since we removed the slider
                train_size = int(dataset_size * (1 - val_split))
                iterations_per_epoch = train_size // batch_size
                total_iterations = iterations_per_epoch * epochs
                
                # Buat informasi HTML
                info_html = f"""
                <h4>Ringkasan Strategi Pelatihan</h4>
                <ul>
                    <li><b>Batch Size:</b> {batch_size}</li>
                    <li><b>Epochs:</b> {epochs}</li>
                    <li><b>Resolusi Gambar:</b> {image_size}x{image_size}</li>
                    <li><b>Early Stopping:</b> {'Aktif' if early_stopping else 'Nonaktif'}</li>
                    <li><b>Validasi Setiap:</b> {val_frequency} epoch</li>
                    <li><b>Patience:</b> {patience} epoch</li>
                    <li><b>Estimasi Iterasi:</b> ~{total_iterations} (berdasarkan asumsi dataset)</li>
                </ul>
                <p><i>Catatan: Pastikan parameter pelatihan sesuai dengan kebutuhan dan kapasitas hardware.</i></p>
                """
                
                ui_components['training_strategy_info'].value = info_html
            except Exception as e:
                ui_components['training_strategy_info'].value = f"<p style='color:red'>❌ Error: {str(e)}</p>"
        
        # Handler buttons
        def on_save_click(b): 
            save_config(ui_components, config, "configs/training_config.yaml", update_config_from_ui, "Strategi Pelatihan")
        
        def on_reset_click(b): 
            reset_config(ui_components, config, default_config, update_ui_from_config, "Strategi Pelatihan")
        
        # Register handlers
        ui_components['save_button'].on_click(on_save_click)
        ui_components['reset_button'].on_click(on_reset_click)
        
        # Tambahkan fungsi ke ui_components
        ui_components['update_config_from_ui'] = update_config_from_ui
        ui_components['update_ui_from_config'] = update_ui_from_config
        ui_components['update_training_strategy_info'] = update_training_strategy_info
        
        # Inisialisasi UI dari config
        update_ui_from_config()
        
    except Exception as e:
        # Fallback sederhana jika terjadi error
        if 'status' in ui_components:
            with ui_components['status']: display(HTML(f"<p style='color:red'>❌ Error setup training strategy button handler: {str(e)}</p>"))
        else: print(f"❌ Error setup training strategy button handler: {str(e)}")
    
    return ui_components
