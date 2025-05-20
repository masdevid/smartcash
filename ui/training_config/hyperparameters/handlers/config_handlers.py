"""
File: smartcash/ui/training_config/hyperparameters/handlers/config_handlers.py
Deskripsi: Handler konfigurasi untuk hyperparameters model
"""

from typing import Dict, Any, Optional
import os
import yaml
from pathlib import Path
from smartcash.common.config import get_config_manager
from smartcash.common.logger import get_logger, LogLevel
from smartcash.ui.utils.constants import ICONS
from smartcash.ui.training_config.hyperparameters.default_config import get_default_hyperparameters_config

# Setup logger dengan level CRITICAL untuk mengurangi log
logger = get_logger(__name__)
logger.set_level(LogLevel.CRITICAL)

def get_default_base_dir():
    if "COLAB_GPU" in os.environ or "COLAB_TPU_ADDR" in os.environ:
        return "/content"
    return str(Path.home() / "SmartCash")

def get_hyperparameters_config(ui_components: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Dapatkan konfigurasi hyperparameters dari config manager.
    
    Args:
        ui_components: Dictionary komponen UI (opsional)
        
    Returns:
        Dictionary konfigurasi hyperparameters
    """
    try:
        config_manager = get_config_manager(base_dir=get_default_base_dir())
        config = config_manager.get_module_config('hyperparameters')
        if config and 'hyperparameters' in config:
            return config
        logger.warning("⚠️ Konfigurasi hyperparameters tidak ditemukan, menggunakan default")
        return get_default_hyperparameters_config()
    except Exception as e:
        logger.error(f"❌ Error saat mengambil konfigurasi hyperparameters: {str(e)}")
        return get_default_hyperparameters_config()

def update_config_from_ui(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update konfigurasi hyperparameters dari UI components.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary konfigurasi yang telah diupdate
    """
    try:
        config_manager = get_config_manager(base_dir=get_default_base_dir())
        config = config_manager.get_module_config('hyperparameters') or get_default_hyperparameters_config()
        
        # Pastikan config memiliki struktur yang benar
        if 'hyperparameters' not in config:
            config = {'hyperparameters': config}
        
        # Update optimizer settings
        if 'optimizer_type_dropdown' in ui_components:
            config['hyperparameters']['optimizer']['type'] = ui_components['optimizer_type_dropdown'].value
            
        if 'learning_rate_slider' in ui_components:
            config['hyperparameters']['optimizer']['learning_rate'] = ui_components['learning_rate_slider'].value
            
        if 'weight_decay_slider' in ui_components:
            config['hyperparameters']['optimizer']['weight_decay'] = ui_components['weight_decay_slider'].value
            
        if 'momentum_slider' in ui_components:
            config['hyperparameters']['optimizer']['momentum'] = ui_components['momentum_slider'].value
            
        if 'beta1_slider' in ui_components:
            config['hyperparameters']['optimizer']['beta1'] = ui_components['beta1_slider'].value
            
        if 'beta2_slider' in ui_components:
            config['hyperparameters']['optimizer']['beta2'] = ui_components['beta2_slider'].value
            
        if 'eps_slider' in ui_components:
            config['hyperparameters']['optimizer']['eps'] = ui_components['eps_slider'].value
        
        # Update scheduler settings
        if 'scheduler_enabled_checkbox' in ui_components:
            config['hyperparameters']['scheduler']['enabled'] = ui_components['scheduler_enabled_checkbox'].value
            
        if 'scheduler_type_dropdown' in ui_components:
            config['hyperparameters']['scheduler']['type'] = ui_components['scheduler_type_dropdown'].value
            
        if 'warmup_epochs_slider' in ui_components:
            config['hyperparameters']['scheduler']['warmup_epochs'] = ui_components['warmup_epochs_slider'].value
            
        if 'min_lr_slider' in ui_components:
            config['hyperparameters']['scheduler']['min_lr'] = ui_components['min_lr_slider'].value
            
        if 'patience_slider' in ui_components:
            config['hyperparameters']['scheduler']['patience'] = ui_components['patience_slider'].value
            
        if 'factor_slider' in ui_components:
            config['hyperparameters']['scheduler']['factor'] = ui_components['factor_slider'].value
            
        if 'threshold_slider' in ui_components:
            config['hyperparameters']['scheduler']['threshold'] = ui_components['threshold_slider'].value
        
        # Update loss settings
        if 'loss_type_dropdown' in ui_components:
            config['hyperparameters']['loss']['type'] = ui_components['loss_type_dropdown'].value
            
        if 'alpha_slider' in ui_components:
            config['hyperparameters']['loss']['alpha'] = ui_components['alpha_slider'].value
            
        if 'gamma_slider' in ui_components:
            config['hyperparameters']['loss']['gamma'] = ui_components['gamma_slider'].value
            
        if 'label_smoothing_slider' in ui_components:
            config['hyperparameters']['loss']['label_smoothing'] = ui_components['label_smoothing_slider'].value
            
        if 'box_loss_gain_slider' in ui_components:
            config['hyperparameters']['loss']['box_loss_gain'] = ui_components['box_loss_gain_slider'].value
            
        if 'cls_loss_gain_slider' in ui_components:
            config['hyperparameters']['loss']['cls_loss_gain'] = ui_components['cls_loss_gain_slider'].value
            
        if 'obj_loss_gain_slider' in ui_components:
            config['hyperparameters']['loss']['obj_loss_gain'] = ui_components['obj_loss_gain_slider'].value
        
        # Update augmentation settings
        if 'augmentation_enabled_checkbox' in ui_components:
            config['hyperparameters']['augmentation']['enabled'] = ui_components['augmentation_enabled_checkbox'].value
            
        if 'mosaic_checkbox' in ui_components:
            config['hyperparameters']['augmentation']['mosaic'] = ui_components['mosaic_checkbox'].value
            
        if 'mixup_checkbox' in ui_components:
            config['hyperparameters']['augmentation']['mixup'] = ui_components['mixup_checkbox'].value
            
        if 'hsv_h_slider' in ui_components:
            config['hyperparameters']['augmentation']['hsv_h'] = ui_components['hsv_h_slider'].value
            
        if 'hsv_s_slider' in ui_components:
            config['hyperparameters']['augmentation']['hsv_s'] = ui_components['hsv_s_slider'].value
            
        if 'hsv_v_slider' in ui_components:
            config['hyperparameters']['augmentation']['hsv_v'] = ui_components['hsv_v_slider'].value
            
        if 'degrees_slider' in ui_components:
            config['hyperparameters']['augmentation']['degrees'] = ui_components['degrees_slider'].value
            
        if 'translate_slider' in ui_components:
            config['hyperparameters']['augmentation']['translate'] = ui_components['translate_slider'].value
            
        if 'scale_slider' in ui_components:
            config['hyperparameters']['augmentation']['scale'] = ui_components['scale_slider'].value
            
        if 'shear_slider' in ui_components:
            config['hyperparameters']['augmentation']['shear'] = ui_components['shear_slider'].value
            
        if 'perspective_slider' in ui_components:
            config['hyperparameters']['augmentation']['perspective'] = ui_components['perspective_slider'].value
            
        if 'flipud_slider' in ui_components:
            config['hyperparameters']['augmentation']['flipud'] = ui_components['flipud_slider'].value
            
        if 'fliplr_slider' in ui_components:
            config['hyperparameters']['augmentation']['fliplr'] = ui_components['fliplr_slider'].value
            
        if 'mosaic_prob_slider' in ui_components:
            config['hyperparameters']['augmentation']['mosaic_prob'] = ui_components['mosaic_prob_slider'].value
            
        if 'mixup_prob_slider' in ui_components:
            config['hyperparameters']['augmentation']['mixup_prob'] = ui_components['mixup_prob_slider'].value
        
        logger.info("✅ Konfigurasi hyperparameters berhasil diupdate")
        
        return config
        
    except Exception as e:
        logger.error(f"❌ Error saat update konfigurasi hyperparameters: {str(e)}")
        return get_default_hyperparameters_config()

def update_ui_from_config(ui_components: Dict[str, Any], config: Dict[str, Any] = None) -> None:
    """
    Update UI dari konfigurasi hyperparameters.
    
    Args:
        ui_components: Dictionary komponen UI
        config: Konfigurasi yang akan digunakan (opsional)
    """
    try:
        # Get config if not provided
        if config is None:
            config = get_hyperparameters_config(ui_components)
            
        # Ensure config has hyperparameters key
        if 'hyperparameters' not in config:
            config = {'hyperparameters': config}
            
        # Get hyperparameters config with safe defaults
        hp_config = config['hyperparameters']
        default_config = get_default_hyperparameters_config()['hyperparameters']
        
        # Update optimizer UI components
        optimizer = hp_config.get('optimizer', default_config['optimizer'])
        
        if 'optimizer_type_dropdown' in ui_components:
            ui_components['optimizer_type_dropdown'].value = optimizer.get('type', default_config['optimizer']['type'])
            
        if 'learning_rate_slider' in ui_components:
            ui_components['learning_rate_slider'].value = optimizer.get('learning_rate', default_config['optimizer']['learning_rate'])
            
        if 'weight_decay_slider' in ui_components:
            ui_components['weight_decay_slider'].value = optimizer.get('weight_decay', default_config['optimizer']['weight_decay'])
            
        if 'momentum_slider' in ui_components:
            ui_components['momentum_slider'].value = optimizer.get('momentum', default_config['optimizer']['momentum'])
            
        if 'beta1_slider' in ui_components:
            ui_components['beta1_slider'].value = optimizer.get('beta1', default_config['optimizer']['beta1'])
            
        if 'beta2_slider' in ui_components:
            ui_components['beta2_slider'].value = optimizer.get('beta2', default_config['optimizer']['beta2'])
            
        if 'eps_slider' in ui_components:
            ui_components['eps_slider'].value = optimizer.get('eps', default_config['optimizer']['eps'])
        
        # Update scheduler UI components
        scheduler = hp_config.get('scheduler', default_config['scheduler'])
        
        if 'scheduler_enabled_checkbox' in ui_components:
            ui_components['scheduler_enabled_checkbox'].value = scheduler.get('enabled', default_config['scheduler']['enabled'])
            
        if 'scheduler_type_dropdown' in ui_components:
            ui_components['scheduler_type_dropdown'].value = scheduler.get('type', default_config['scheduler']['type'])
            
        if 'warmup_epochs_slider' in ui_components:
            ui_components['warmup_epochs_slider'].value = scheduler.get('warmup_epochs', default_config['scheduler']['warmup_epochs'])
            
        if 'min_lr_slider' in ui_components:
            ui_components['min_lr_slider'].value = scheduler.get('min_lr', default_config['scheduler']['min_lr'])
            
        if 'patience_slider' in ui_components:
            ui_components['patience_slider'].value = scheduler.get('patience', default_config['scheduler']['patience'])
            
        if 'factor_slider' in ui_components:
            ui_components['factor_slider'].value = scheduler.get('factor', default_config['scheduler']['factor'])
            
        if 'threshold_slider' in ui_components:
            ui_components['threshold_slider'].value = scheduler.get('threshold', default_config['scheduler']['threshold'])
        
        # Update loss UI components
        loss = hp_config.get('loss', default_config['loss'])
        
        if 'loss_type_dropdown' in ui_components:
            ui_components['loss_type_dropdown'].value = loss.get('type', default_config['loss']['type'])
            
        if 'alpha_slider' in ui_components:
            ui_components['alpha_slider'].value = loss.get('alpha', default_config['loss']['alpha'])
            
        if 'gamma_slider' in ui_components:
            ui_components['gamma_slider'].value = loss.get('gamma', default_config['loss']['gamma'])
            
        if 'label_smoothing_slider' in ui_components:
            ui_components['label_smoothing_slider'].value = loss.get('label_smoothing', default_config['loss']['label_smoothing'])
            
        if 'box_loss_gain_slider' in ui_components:
            ui_components['box_loss_gain_slider'].value = loss.get('box_loss_gain', default_config['loss']['box_loss_gain'])
            
        if 'cls_loss_gain_slider' in ui_components:
            ui_components['cls_loss_gain_slider'].value = loss.get('cls_loss_gain', default_config['loss']['cls_loss_gain'])
            
        if 'obj_loss_gain_slider' in ui_components:
            ui_components['obj_loss_gain_slider'].value = loss.get('obj_loss_gain', default_config['loss']['obj_loss_gain'])
        
        # Update augmentation UI components
        augmentation = hp_config.get('augmentation', default_config['augmentation'])
        
        if 'augmentation_enabled_checkbox' in ui_components:
            ui_components['augmentation_enabled_checkbox'].value = augmentation.get('enabled', default_config['augmentation']['enabled'])
            
        if 'mosaic_checkbox' in ui_components:
            ui_components['mosaic_checkbox'].value = augmentation.get('mosaic', default_config['augmentation']['mosaic'])
            
        if 'mixup_checkbox' in ui_components:
            ui_components['mixup_checkbox'].value = augmentation.get('mixup', default_config['augmentation']['mixup'])
            
        if 'hsv_h_slider' in ui_components:
            ui_components['hsv_h_slider'].value = augmentation.get('hsv_h', default_config['augmentation']['hsv_h'])
            
        if 'hsv_s_slider' in ui_components:
            ui_components['hsv_s_slider'].value = augmentation.get('hsv_s', default_config['augmentation']['hsv_s'])
            
        if 'hsv_v_slider' in ui_components:
            ui_components['hsv_v_slider'].value = augmentation.get('hsv_v', default_config['augmentation']['hsv_v'])
            
        if 'degrees_slider' in ui_components:
            ui_components['degrees_slider'].value = augmentation.get('degrees', default_config['augmentation']['degrees'])
            
        if 'translate_slider' in ui_components:
            ui_components['translate_slider'].value = augmentation.get('translate', default_config['augmentation']['translate'])
            
        if 'scale_slider' in ui_components:
            ui_components['scale_slider'].value = augmentation.get('scale', default_config['augmentation']['scale'])
            
        if 'shear_slider' in ui_components:
            ui_components['shear_slider'].value = augmentation.get('shear', default_config['augmentation']['shear'])
            
        if 'perspective_slider' in ui_components:
            ui_components['perspective_slider'].value = augmentation.get('perspective', default_config['augmentation']['perspective'])
            
        if 'flipud_slider' in ui_components:
            ui_components['flipud_slider'].value = augmentation.get('flipud', default_config['augmentation']['flipud'])
            
        if 'fliplr_slider' in ui_components:
            ui_components['fliplr_slider'].value = augmentation.get('fliplr', default_config['augmentation']['fliplr'])
            
        if 'mosaic_prob_slider' in ui_components:
            ui_components['mosaic_prob_slider'].value = augmentation.get('mosaic_prob', default_config['augmentation']['mosaic_prob'])
            
        if 'mixup_prob_slider' in ui_components:
            ui_components['mixup_prob_slider'].value = augmentation.get('mixup_prob', default_config['augmentation']['mixup_prob'])
            
        logger.info("✅ UI berhasil diupdate dari konfigurasi hyperparameters")
        
    except Exception as e:
        logger.error(f"❌ Error saat mengupdate UI dari konfigurasi: {str(e)}")
        if 'info_panel' in ui_components:
            ui_components['info_panel'].value = f"Error: {str(e)}"

def save_config(config: Dict[str, Any], ui_components: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Simpan konfigurasi hyperparameters ke config manager dan sinkronkan dengan Google Drive.
    
    Args:
        config: Dictionary konfigurasi yang akan disimpan
        ui_components: Dictionary komponen UI (opsional)
        
    Returns:
        Konfigurasi yang telah disimpan dan disinkronkan
    """
    try:
        # Update status panel
        if ui_components:
            from smartcash.ui.training_config.hyperparameters.handlers.sync_logger import update_sync_status_only
            update_sync_status_only(ui_components, "Menyimpan konfigurasi...", 'info')
        
        # Simpan konfigurasi asli untuk verifikasi
        original_config = config.copy() if config else get_default_hyperparameters_config()
        
        # Pastikan konfigurasi memiliki struktur yang benar
        if 'hyperparameters' not in original_config:
            original_config = {'hyperparameters': original_config}
        
        # Simpan konfigurasi
        base_dir = get_default_base_dir()
        config_manager = get_config_manager(base_dir=base_dir)
        save_success = config_manager.save_module_config('hyperparameters', original_config)
        
        if not save_success:
            if ui_components:
                from smartcash.ui.training_config.hyperparameters.handlers.sync_logger import update_sync_status_only
                update_sync_status_only(ui_components, "Gagal menyimpan konfigurasi hyperparameters", 'error')
            return original_config
        
        logger.info(f"{ICONS.get('success', '✅')} Konfigurasi hyperparameters berhasil disimpan")
        
        # Log ke UI jika ui_components tersedia
        if ui_components:
            from smartcash.ui.training_config.hyperparameters.handlers.sync_logger import update_sync_status_only
            update_sync_status_only(ui_components, "Konfigurasi hyperparameters berhasil disimpan", 'success')
        
        # Verifikasi konfigurasi tersimpan dengan benar
        saved_config = config_manager.get_module_config('hyperparameters', {})
        
        # Verifikasi konsistensi
        if 'hyperparameters' in saved_config and 'hyperparameters' in original_config:
            is_consistent = True
            for key, value in original_config['hyperparameters'].items():
                if key not in saved_config['hyperparameters'] or saved_config['hyperparameters'][key] != value:
                    is_consistent = False
                    logger.warning(f"⚠️ Inkonsistensi data pada key '{key}': {value} vs {saved_config['hyperparameters'].get(key, 'tidak ada')}")
                    if ui_components:
                        from smartcash.ui.training_config.hyperparameters.handlers.sync_logger import update_sync_status_only
                        update_sync_status_only(ui_components, f"Inkonsistensi data pada key '{key}'", 'warning')
                    break
            
            if not is_consistent:
                # Coba simpan ulang jika tidak konsisten
                config_manager.save_module_config('hyperparameters', original_config)
                # Log warning
                logger.warning(f"⚠️ Data tidak konsisten setelah penyimpanan, mencoba kembali")
                if ui_components:
                    from smartcash.ui.training_config.hyperparameters.handlers.sync_logger import update_sync_status_only
                    update_sync_status_only(ui_components, "Data tidak konsisten, mencoba kembali", 'warning')
                
                # Verifikasi ulang setelah simpan ulang
                saved_config = config_manager.get_module_config('hyperparameters', {})
        
        # Sinkronisasi dengan Google Drive jika di Colab
        from smartcash.ui.training_config.hyperparameters.handlers.drive_handlers import is_colab_environment, sync_with_drive
        if is_colab_environment():
            if ui_components:
                from smartcash.ui.training_config.hyperparameters.handlers.sync_logger import update_sync_status_only
                update_sync_status_only(ui_components, "Menyinkronkan dengan Google Drive...", 'info')
            
            # Pastikan nilai yang disinkronkan menggunakan nilai original_config
            # untuk menghindari inkonsistensi
            synced_config = sync_with_drive(original_config, ui_components)
            
            # Verifikasi konfigurasi yang disinkronkan dengan membandingkan dengan nilai asli
            is_synced_consistent = True
            if 'hyperparameters' in synced_config and 'hyperparameters' in original_config:
                for key, value in original_config['hyperparameters'].items():
                    if key not in synced_config['hyperparameters'] or synced_config['hyperparameters'][key] != value:
                        is_synced_consistent = False
                        logger.warning(f"⚠️ Inkonsistensi data setelah sinkronisasi pada key '{key}': {value} vs {synced_config['hyperparameters'].get(key, 'tidak ada')}")
                        if ui_components:
                            from smartcash.ui.training_config.hyperparameters.handlers.sync_logger import update_sync_status_only
                            update_sync_status_only(ui_components, f"Inkonsistensi data setelah sinkronisasi", 'warning')
                        break
            
            if is_synced_consistent:
                if ui_components:
                    from smartcash.ui.training_config.hyperparameters.handlers.sync_logger import update_sync_status_only
                    update_sync_status_only(ui_components, "Konfigurasi berhasil disimpan dan disinkronkan", 'success')
            
            return synced_config
        
        return saved_config
    except Exception as e:
        logger.error(f"{ICONS.get('error', '❌')} Error saat menyimpan konfigurasi hyperparameters: {str(e)}")
        
        # Log ke UI jika ui_components tersedia
        if ui_components:
            from smartcash.ui.training_config.hyperparameters.handlers.sync_logger import update_sync_status_only
            update_sync_status_only(ui_components, f"Error saat menyimpan konfigurasi: {str(e)}", 'error')
        
        return config

def load_config() -> Dict[str, Any]:
    """
    Muat konfigurasi hyperparameters dari config manager.
    
    Returns:
        Dictionary konfigurasi hyperparameters
    """
    try:
        config_manager = get_config_manager(base_dir=get_default_base_dir())
        config = config_manager.get_module_config('hyperparameters')
        if config and 'hyperparameters' in config:
            return config
        logger.warning("⚠️ Konfigurasi hyperparameters tidak ditemukan, menggunakan default")
        return get_default_hyperparameters_config()
    except Exception as e:
        logger.error(f"{ICONS.get('error', '❌')} Error saat memuat konfigurasi hyperparameters: {str(e)}")
        return get_default_hyperparameters_config()

def update_hyperparameters_info(ui_components: Dict[str, Any]) -> None:
    """
    Update info panel dengan informasi hyperparameters yang dipilih.
    
    Args:
        ui_components: Dictionary komponen UI
    """
    try:
        info_panel = ui_components.get('info_panel')
        if not info_panel:
            logger.warning(f"{ICONS.get('warning', '⚠️')} Info panel tidak ditemukan")
            return
            
        # Get current config
        config = get_hyperparameters_config(ui_components)
        
        # Ensure config has hyperparameters key
        if 'hyperparameters' not in config:
            config['hyperparameters'] = get_default_hyperparameters_config()['hyperparameters']
        
        # Get hyperparameters config with safe defaults
        hp_config = config['hyperparameters']
        default_config = get_default_hyperparameters_config()['hyperparameters']
        
        # Safely get values with defaults
        optimizer_type = hp_config.get('optimizer', {}).get('type', default_config['optimizer']['type'])
        learning_rate = hp_config.get('optimizer', {}).get('learning_rate', default_config['optimizer']['learning_rate'])
        scheduler_type = hp_config.get('scheduler', {}).get('type', default_config['scheduler']['type'])
        loss_type = hp_config.get('loss', {}).get('type', default_config['loss']['type'])
        augmentation_enabled = hp_config.get('augmentation', {}).get('enabled', default_config['augmentation']['enabled'])
        
        # Update info panel dengan informasi hyperparameters
        info_text = f"""
        <div style='font-family: monospace;'>
        <h4>Hyperparameters Configuration:</h4>
        <ul>
            <li>Optimizer: {optimizer_type}</li>
            <li>Learning Rate: {learning_rate}</li>
            <li>Scheduler: {scheduler_type}</li>
            <li>Loss Function: {loss_type}</li>
            <li>Augmentation: {'Enabled' if augmentation_enabled else 'Disabled'}</li>
        </ul>
        </div>
        """
        
        info_panel.value = info_text
        
    except Exception as e:
        logger.error(f"{ICONS.get('error', '❌')} Error saat update info panel: {str(e)}")
        if info_panel:
            info_panel.value = f"{ICONS.get('error', '❌')} Error saat update info panel: {str(e)}" 