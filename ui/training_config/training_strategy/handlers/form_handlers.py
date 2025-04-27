"""
File: smartcash/ui/training_config/training_strategy/handlers/form_handlers.py
Deskripsi: Handler untuk form UI pada komponen strategi pelatihan
"""

from typing import Dict, Any, Optional, Callable
import ipywidgets as widgets
from IPython.display import display, HTML

def setup_training_strategy_form_handlers(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """
    Setup handler untuk form pada komponen UI strategi pelatihan.
    
    Args:
        ui_components: Komponen UI
        env: Environment manager
        config: Konfigurasi model
        
    Returns:
        Dict berisi komponen UI dengan handler terpasang
    """
    try:
        # Dapatkan logger jika tersedia
        logger = ui_components.get('logger', None)
        
        # Handler untuk perubahan komponen
        def on_component_change(change):
            if change['name'] == 'value':
                # Update config
                if 'update_config_from_ui' in ui_components:
                    ui_components['update_config_from_ui']()
        
        def on_early_stopping_change(change):
            if change['name'] == 'value':
                # Aktifkan/nonaktifkan patience berdasarkan early stopping
                ui_components['patience'].disabled = not change['new']
                
                # Update config
                if 'update_config_from_ui' in ui_components:
                    ui_components['update_config_from_ui']()
        
        def on_resume_change(change):
            if change['name'] == 'value':
                # Aktifkan/nonaktifkan checkpoint path berdasarkan resume
                ui_components['checkpoint_path'].disabled = not change['new']
                
                # Update config
                if 'update_config_from_ui' in ui_components:
                    ui_components['update_config_from_ui']()
        
        def on_use_multi_gpu_change(change):
            if change['name'] == 'value':
                # Aktifkan/nonaktifkan opsi multi-GPU
                ui_components['sync_bn'].disabled = not change['new']
                ui_components['distributed'].disabled = not change['new']
                
                # Update config
                if 'update_config_from_ui' in ui_components:
                    ui_components['update_config_from_ui']()
        
        # Register observers untuk semua komponen
        ui_components['batch_size'].observe(on_component_change)
        ui_components['epochs'].observe(on_component_change)
        ui_components['image_size'].observe(on_component_change)
        ui_components['workers'].observe(on_component_change)
        
        ui_components['val_split'].observe(on_component_change)
        ui_components['val_frequency'].observe(on_component_change)
        ui_components['early_stopping'].observe(on_early_stopping_change)
        ui_components['patience'].observe(on_component_change)
        
        ui_components['experiment_name'].observe(on_component_change)
        ui_components['save_period'].observe(on_component_change)
        ui_components['resume'].observe(on_resume_change)
        ui_components['checkpoint_path'].observe(on_component_change)
        
        ui_components['use_multi_gpu'].observe(on_use_multi_gpu_change)
        ui_components['sync_bn'].observe(on_component_change)
        ui_components['distributed'].observe(on_component_change)
        
        # Inisialisasi state komponen
        ui_components['patience'].disabled = not ui_components['early_stopping'].value
        ui_components['checkpoint_path'].disabled = not ui_components['resume'].value
        ui_components['sync_bn'].disabled = not ui_components['use_multi_gpu'].value
        ui_components['distributed'].disabled = not ui_components['use_multi_gpu'].value
        
        # Cleanup function
        def cleanup():
            try:
                # Hapus semua observer
                ui_components['batch_size'].unobserve(on_component_change)
                ui_components['epochs'].unobserve(on_component_change)
                ui_components['image_size'].unobserve(on_component_change)
                ui_components['workers'].unobserve(on_component_change)
                
                ui_components['val_split'].unobserve(on_component_change)
                ui_components['val_frequency'].unobserve(on_component_change)
                ui_components['early_stopping'].unobserve(on_early_stopping_change)
                ui_components['patience'].unobserve(on_component_change)
                
                ui_components['experiment_name'].unobserve(on_component_change)
                ui_components['save_period'].unobserve(on_component_change)
                ui_components['resume'].unobserve(on_resume_change)
                ui_components['checkpoint_path'].unobserve(on_component_change)
                
                ui_components['use_multi_gpu'].unobserve(on_use_multi_gpu_change)
                ui_components['sync_bn'].unobserve(on_component_change)
                ui_components['distributed'].unobserve(on_component_change)
                
                if logger: logger.info("✅ Training strategy form handlers cleaned up")
            except Exception as e:
                if logger: logger.warning(f"⚠️ Error cleanup: {e}")
        
        # Tambahkan cleanup function
        ui_components['cleanup'] = cleanup
        
    except Exception as e:
        # Fallback sederhana jika terjadi error
        if 'status' in ui_components:
            with ui_components['status']: display(HTML(f"<p style='color:red'>❌ Error setup training strategy form handler: {str(e)}</p>"))
        else: print(f"❌ Error setup training strategy form handler: {str(e)}")
    
    return ui_components
