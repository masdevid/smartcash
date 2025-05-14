"""
File: smartcash/ui/training_config/hyperparameters/handlers/form_handlers.py
Deskripsi: Handler untuk form UI pada komponen hyperparameter
"""

from typing import Dict, Any, Optional, Callable
import ipywidgets as widgets
from IPython.display import display, HTML

def setup_hyperparameters_form_handlers(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """
    Setup handler untuk form pada komponen UI hyperparameter.
    
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
        
        def on_augment_change(change):
            if change['name'] == 'value':
                # Aktifkan/nonaktifkan komponen regularisasi terkait
                ui_components['dropout'].disabled = not change['new']
                
                # Update config
                if 'update_config_from_ui' in ui_components:
                    ui_components['update_config_from_ui']()
        
        def on_scheduler_type_change(change):
            if change['name'] == 'value':
                # Aktifkan/nonaktifkan komponen scheduler berdasarkan tipe
                is_none = change['new'] == 'none'
                
                ui_components['warmup_epochs'].disabled = is_none
                ui_components['warmup_momentum'].disabled = is_none
                ui_components['warmup_bias_lr'].disabled = is_none
                
                # Update config
                if 'update_config_from_ui' in ui_components:
                    ui_components['update_config_from_ui']()
        
        def on_optimizer_type_change(change):
            if change['name'] == 'value':
                # Aktifkan/nonaktifkan momentum berdasarkan tipe optimizer
                is_sgd = change['new'] == 'SGD'
                is_adam = change['new'] in ['Adam', 'AdamW']
                
                ui_components['momentum'].disabled = not is_sgd
                ui_components['weight_decay'].disabled = not is_adam
                
                # Update config
                if 'update_config_from_ui' in ui_components:
                    ui_components['update_config_from_ui']()
        
        def on_early_stopping_change(change):
            if change['name'] == 'value':
                # Aktifkan/nonaktifkan komponen early stopping
                ui_components['early_stopping_patience'].disabled = not change['new']
                ui_components['early_stopping_min_delta'].disabled = not change['new']
                
                # Update config
                if 'update_config_from_ui' in ui_components:
                    ui_components['update_config_from_ui']()
        
        # Register observers untuk semua komponen
        # Parameter dasar
        ui_components['batch_size'].observe(on_component_change)
        ui_components['image_size'].observe(on_component_change)
        ui_components['epochs'].observe(on_component_change)
        
        # Parameter optimasi
        ui_components['optimizer_type'].observe(on_optimizer_type_change)
        ui_components['learning_rate'].observe(on_component_change)
        ui_components['weight_decay'].observe(on_component_change)
        ui_components['momentum'].observe(on_component_change)
        
        # Parameter penjadwalan
        ui_components['lr_scheduler'].observe(on_scheduler_type_change)
        ui_components['warmup_epochs'].observe(on_component_change)
        ui_components['warmup_momentum'].observe(on_component_change)
        ui_components['warmup_bias_lr'].observe(on_component_change)
        
        # Parameter regularisasi
        ui_components['augment'].observe(on_augment_change)
        ui_components['dropout'].observe(on_component_change)
        
        # Parameter loss
        ui_components['box_loss_gain'].observe(on_component_change)
        ui_components['cls_loss_gain'].observe(on_component_change)
        ui_components['obj_loss_gain'].observe(on_component_change)
        
        # Parameter early stopping & checkpoint
        ui_components['early_stopping_enabled'].observe(on_early_stopping_change)
        ui_components['early_stopping_patience'].observe(on_component_change)
        ui_components['early_stopping_min_delta'].observe(on_component_change)
        ui_components['checkpoint_save_best'].observe(on_component_change)
        ui_components['checkpoint_save_period'].observe(on_component_change)
        
        # Inisialisasi state komponen
        # Aktifkan/nonaktifkan momentum berdasarkan tipe optimizer
        is_sgd = ui_components['optimizer_type'].value == 'SGD'
        is_adam = ui_components['optimizer_type'].value in ['Adam', 'AdamW']
        ui_components['momentum'].disabled = not is_sgd
        ui_components['weight_decay'].disabled = not is_adam
        
        # Aktifkan/nonaktifkan komponen scheduler berdasarkan tipe
        is_none = ui_components['lr_scheduler'].value == 'none'
        ui_components['warmup_epochs'].disabled = is_none
        ui_components['warmup_momentum'].disabled = is_none
        ui_components['warmup_bias_lr'].disabled = is_none
        
        # Aktifkan/nonaktifkan komponen regularisasi
        is_aug_enabled = ui_components['augment'].value
        ui_components['dropout'].disabled = not is_aug_enabled
        
        # Aktifkan/nonaktifkan komponen early stopping
        is_es_enabled = ui_components['early_stopping_enabled'].value
        ui_components['early_stopping_patience'].disabled = not is_es_enabled
        ui_components['early_stopping_min_delta'].disabled = not is_es_enabled
        
        # Cleanup function
        def cleanup():
            try:
                # Hapus semua observer
                ui_components['optimizer_type'].unobserve(on_optimizer_type_change)
                ui_components['learning_rate'].unobserve(on_component_change)
                ui_components['weight_decay'].unobserve(on_component_change)
                ui_components['momentum'].unobserve(on_component_change)
                
                ui_components['scheduler_type'].unobserve(on_scheduler_type_change)
                ui_components['warmup_epochs'].unobserve(on_component_change)
                ui_components['step_size'].unobserve(on_component_change)
                ui_components['gamma'].unobserve(on_component_change)
                
                ui_components['use_augmentation'].unobserve(on_use_augmentation_change)
                ui_components['mosaic'].unobserve(on_component_change)
                ui_components['mixup'].unobserve(on_component_change)
                ui_components['flip'].unobserve(on_component_change)
                ui_components['hsv_h'].unobserve(on_component_change)
                ui_components['hsv_s'].unobserve(on_component_change)
                ui_components['hsv_v'].unobserve(on_component_change)
                
                if logger: logger.info("✅ Hyperparameter form handlers cleaned up")
            except Exception as e:
                if logger: logger.warning(f"⚠️ Error cleanup: {e}")
        
        # Tambahkan cleanup function
        ui_components['cleanup'] = cleanup
        
    except Exception as e:
        # Fallback sederhana jika terjadi error
        if 'status' in ui_components:
            with ui_components['status']: display(HTML(f"<p style='color:red'>❌ Error setup hyperparameter form handler: {str(e)}</p>"))
        else: print(f"❌ Error setup hyperparameter form handler: {str(e)}")
    
    return ui_components
