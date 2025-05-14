"""
File: smartcash/ui/training_config/hyperparameters/handlers/button_handlers.py
Deskripsi: Handler untuk tombol UI pada komponen hyperparameter
"""

from typing import Dict, Any, Optional, Callable
import ipywidgets as widgets
from IPython.display import display, HTML

def setup_hyperparameters_button_handlers(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """
    Setup handler untuk tombol pada komponen UI hyperparameter.
    
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
        
        # Dapatkan logger jika tersedia
        logger = ui_components.get('logger', None)
        
        # Validasi config
        if config is None: config = {}
        
        # Default config berdasarkan hyperparameter_config.yaml terbaru
        default_config = {
            'hyperparameters': {
                # Parameter dasar
                'batch_size': 16,
                'image_size': 640,
                'epochs': 100,
                
                # Parameter optimasi
                'optimizer': 'Adam',
                'learning_rate': 0.001,
                'weight_decay': 0.0005,
                'momentum': 0.937,
                
                # Parameter penjadwalan
                'lr_scheduler': 'cosine',
                'warmup_epochs': 3,
                'warmup_momentum': 0.8,
                'warmup_bias_lr': 0.1,
                
                # Parameter regularisasi
                'augment': True,
                'dropout': 0.0,
                
                # Parameter loss
                'box_loss_gain': 0.05,
                'cls_loss_gain': 0.5,
                'obj_loss_gain': 1.0,
                
                # Parameter anchor
                'anchor_t': 4.0,
                'fl_gamma': 0.0,
                
                # Parameter early stopping
                'early_stopping': {
                    'enabled': True,
                    'patience': 15,
                    'min_delta': 0.001
                },
                
                # Parameter checkpoint
                'checkpoint': {
                    'save_period': 10,
                    'save_best': True,
                    'metric': 'mAP'
                }
            }
        }
        
        # Update config dari UI
        def update_config_from_ui(current_config=None):
            if current_config is None: current_config = config
            
            # Pastikan struktur config ada
            if 'hyperparameters' not in current_config:
                current_config['hyperparameters'] = {}
            
            # Update parameter dasar
            current_config['hyperparameters']['batch_size'] = ui_components['batch_size'].value
            current_config['hyperparameters']['image_size'] = ui_components['image_size'].value
            current_config['hyperparameters']['epochs'] = ui_components['epochs'].value
            
            # Update parameter optimasi
            current_config['hyperparameters']['optimizer'] = ui_components['optimizer_type'].value
            current_config['hyperparameters']['learning_rate'] = ui_components['learning_rate'].value
            current_config['hyperparameters']['weight_decay'] = ui_components['weight_decay'].value
            current_config['hyperparameters']['momentum'] = ui_components['momentum'].value
            
            # Update parameter penjadwalan
            current_config['hyperparameters']['lr_scheduler'] = ui_components['lr_scheduler'].value
            current_config['hyperparameters']['warmup_epochs'] = ui_components['warmup_epochs'].value
            current_config['hyperparameters']['warmup_momentum'] = ui_components['warmup_momentum'].value
            current_config['hyperparameters']['warmup_bias_lr'] = ui_components['warmup_bias_lr'].value
            
            # Update parameter regularisasi
            current_config['hyperparameters']['augment'] = ui_components['augment'].value
            current_config['hyperparameters']['dropout'] = ui_components['dropout'].value
            
            # Update parameter loss
            current_config['hyperparameters']['box_loss_gain'] = ui_components['box_loss_gain'].value
            current_config['hyperparameters']['cls_loss_gain'] = ui_components['cls_loss_gain'].value
            current_config['hyperparameters']['obj_loss_gain'] = ui_components['obj_loss_gain'].value
            
            # Update parameter early stopping
            if 'early_stopping' not in current_config['hyperparameters']:
                current_config['hyperparameters']['early_stopping'] = {}
            
            current_config['hyperparameters']['early_stopping']['enabled'] = ui_components['early_stopping_enabled'].value
            current_config['hyperparameters']['early_stopping']['patience'] = ui_components['early_stopping_patience'].value
            current_config['hyperparameters']['early_stopping']['min_delta'] = ui_components['early_stopping_min_delta'].value
            
            # Update parameter checkpoint
            if 'checkpoint' not in current_config['hyperparameters']:
                current_config['hyperparameters']['checkpoint'] = {}
            
            current_config['hyperparameters']['checkpoint']['save_best'] = ui_components['checkpoint_save_best'].value
            current_config['hyperparameters']['checkpoint']['save_period'] = ui_components['checkpoint_save_period'].value
            
            return current_config
        
        # Update UI dari config
        def update_ui_from_config():
            if not config or 'hyperparameters' not in config: return
            
            try:
                # Optimizer
                if 'optimizer' in config['hyperparameters']:
                    if 'type' in config['hyperparameters']['optimizer']:
                        ui_components['optimizer_type'].value = config['hyperparameters']['optimizer']['type']
                    if 'lr' in config['hyperparameters']['optimizer']:
                        ui_components['learning_rate'].value = config['hyperparameters']['optimizer']['lr']
                    if 'weight_decay' in config['hyperparameters']['optimizer']:
                        ui_components['weight_decay'].value = config['hyperparameters']['optimizer']['weight_decay']
                    if 'momentum' in config['hyperparameters']['optimizer']:
                        ui_components['momentum'].value = config['hyperparameters']['optimizer']['momentum']
                
                # Scheduler
                if 'scheduler' in config['hyperparameters']:
                    if 'type' in config['hyperparameters']['scheduler']:
                        ui_components['scheduler_type'].value = config['hyperparameters']['scheduler']['type']
                    if 'warmup_epochs' in config['hyperparameters']['scheduler']:
                        ui_components['warmup_epochs'].value = config['hyperparameters']['scheduler']['warmup_epochs']
                    if 'step_size' in config['hyperparameters']['scheduler']:
                        ui_components['step_size'].value = config['hyperparameters']['scheduler']['step_size']
                    if 'gamma' in config['hyperparameters']['scheduler']:
                        ui_components['gamma'].value = config['hyperparameters']['scheduler']['gamma']
                
                # Augmentation
                if 'augmentation' in config['hyperparameters']:
                    if 'use_augmentation' in config['hyperparameters']['augmentation']:
                        ui_components['use_augmentation'].value = config['hyperparameters']['augmentation']['use_augmentation']
                    if 'mosaic' in config['hyperparameters']['augmentation']:
                        ui_components['mosaic'].value = config['hyperparameters']['augmentation']['mosaic']
                    if 'mixup' in config['hyperparameters']['augmentation']:
                        ui_components['mixup'].value = config['hyperparameters']['augmentation']['mixup']
                    if 'flip' in config['hyperparameters']['augmentation']:
                        ui_components['flip'].value = config['hyperparameters']['augmentation']['flip']
                    if 'hsv_h' in config['hyperparameters']['augmentation']:
                        ui_components['hsv_h'].value = config['hyperparameters']['augmentation']['hsv_h']
                    if 'hsv_s' in config['hyperparameters']['augmentation']:
                        ui_components['hsv_s'].value = config['hyperparameters']['augmentation']['hsv_s']
                    if 'hsv_v' in config['hyperparameters']['augmentation']:
                        ui_components['hsv_v'].value = config['hyperparameters']['augmentation']['hsv_v']
                
                # Update info hyperparameter
                update_hyperparameters_info()
                
                if logger: logger.info("✅ UI hyperparameter diperbarui dari config")
            except Exception as e:
                if logger: logger.error(f"❌ Error update UI: {e}")
        
        # Update informasi hyperparameter
        def update_hyperparameters_info():
            try:
                # Dapatkan nilai dari UI
                # Parameter dasar
                batch_size = ui_components['batch_size'].value
                image_size = ui_components['image_size'].value
                epochs = ui_components['epochs'].value
                
                # Parameter optimasi
                optimizer_type = ui_components['optimizer_type'].value
                lr = ui_components['learning_rate'].value
                weight_decay = ui_components['weight_decay'].value
                momentum = ui_components['momentum'].value
                
                # Parameter penjadwalan
                scheduler_type = ui_components['lr_scheduler'].value
                
                # Parameter regularisasi
                use_augmentation = ui_components['augment'].value
                dropout = ui_components['dropout'].value
                
                # Parameter early stopping
                early_stopping = ui_components['early_stopping_enabled'].value
                
                # Buat informasi HTML
                info_html = f"""
                <h4>Ringkasan Hyperparameter</h4>
                <ul>
                    <li><b>Parameter Dasar:</b> Batch Size: {batch_size}, Image Size: {image_size}, Epochs: {epochs}</li>
                    <li><b>Optimizer:</b> {optimizer_type} (LR: {lr:.6f}, Weight Decay: {weight_decay:.6f}, Momentum: {momentum:.3f})</li>
                    <li><b>Scheduler:</b> {scheduler_type.capitalize()}</li>
                    <li><b>Regularisasi:</b> Augmentasi: {'Aktif' if use_augmentation else 'Nonaktif'}, Dropout: {dropout:.2f}</li>
                    <li><b>Early Stopping:</b> {'Aktif' if early_stopping else 'Nonaktif'}</li>
                </ul>
                <p><i>Catatan: Hyperparameter yang tepat sangat penting untuk performa model yang optimal.</i></p>
                """
                
                ui_components['hyperparameters_info'].value = info_html
            except Exception as e:
                ui_components['hyperparameters_info'].value = f"<p style='color:red'>❌ Error: {str(e)}</p>"
        
        # Handler buttons
        def on_save_click(b):
            try:
                # Dapatkan config manager
                config_manager = get_config_manager()
                
                # Update config dari UI
                updated_config = update_config_from_ui()
                
                # Simpan ke config manager
                config_manager.save_module_config('hyperparameters', updated_config)
                
                # Register UI components untuk persistensi
                config_manager.register_ui_components('hyperparameters', ui_components)
                
                # Tampilkan pesan sukses
                with ui_components['status']:
                    display(create_info_alert("Konfigurasi hyperparameter berhasil disimpan", alert_type='success'))
                
                if logger: logger.info("✅ Konfigurasi hyperparameter berhasil disimpan")
            except Exception as e:
                with ui_components['status']:
                    display(create_info_alert(f"Gagal menyimpan konfigurasi: {str(e)}", alert_type='error'))
                if logger: logger.error(f"❌ Error menyimpan konfigurasi hyperparameter: {e}")
        
        def on_reset_click(b):
            try:
                # Dapatkan config manager
                config_manager = get_config_manager()
                
                # Reset ke default config
                config_manager.reset_module_config('hyperparameters', default_config)
                
                # Update UI dari default config
                update_ui_from_config()
                
                # Tampilkan pesan sukses
                with ui_components['status']:
                    display(create_info_alert("Konfigurasi hyperparameter berhasil direset ke default", alert_type='success'))
                
                if logger: logger.info("✅ Konfigurasi hyperparameter berhasil direset ke default")
            except Exception as e:
                with ui_components['status']:
                    display(create_info_alert(f"Gagal mereset konfigurasi: {str(e)}", alert_type='error'))
                if logger: logger.error(f"❌ Error mereset konfigurasi hyperparameter: {e}")
        
        # Register handlers
        ui_components['save_button'].on_click(on_save_click)
        ui_components['reset_button'].on_click(on_reset_click)
        
        # Tambahkan fungsi ke ui_components
        ui_components['update_config_from_ui'] = update_config_from_ui
        ui_components['update_ui_from_config'] = update_ui_from_config
        ui_components['update_hyperparameters_info'] = update_hyperparameters_info
        
        # Inisialisasi UI dari config yang disimpan
        try:
            # Dapatkan config manager
            config_manager = ConfigManager.get_instance()
            
            # Coba dapatkan konfigurasi yang disimpan
            saved_config = config_manager.get_module_config('hyperparameters')
            
            if saved_config:
                # Update UI dari konfigurasi yang disimpan
                update_ui_from_config(saved_config)
                if logger: logger.info("✅ UI hyperparameter diinisialisasi dari konfigurasi yang disimpan")
            else:
                # Jika tidak ada konfigurasi yang disimpan, gunakan default
                update_ui_from_config(default_config)
                if logger: logger.info("ℹ️ UI hyperparameter diinisialisasi dari konfigurasi default")
                
            # Register UI components untuk persistensi
            config_manager.register_ui_components('hyperparameters', ui_components)
        except Exception as e:
            # Fallback ke default jika terjadi error
            update_ui_from_config(default_config)
            if logger: logger.warning(f"⚠️ Error inisialisasi UI hyperparameter: {e}, menggunakan default")
        
    except Exception as e:
        # Fallback sederhana jika terjadi error
        if 'status' in ui_components:
            with ui_components['status']: display(HTML(f"<p style='color:red'>❌ Error setup hyperparameter button handler: {str(e)}</p>"))
        else: print(f"❌ Error setup hyperparameter button handler: {str(e)}")
    
    return ui_components
