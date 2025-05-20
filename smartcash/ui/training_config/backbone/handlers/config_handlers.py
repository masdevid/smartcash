"""
File: smartcash/ui/training_config/backbone/handlers/config_handlers.py
Deskripsi: Handler konfigurasi untuk backbone model
"""

from typing import Dict, Any, Optional
import os
import yaml
from pathlib import Path
from smartcash.common.config import get_config_manager
from smartcash.common.logger import get_logger, LogLevel
from smartcash.ui.utils.constants import ICONS

# Setup logger dengan level CRITICAL untuk mengurangi log
logger = get_logger(__name__)
logger.set_level(LogLevel.CRITICAL)

def get_default_base_dir():
    if "COLAB_GPU" in os.environ or "COLAB_TPU_ADDR" in os.environ:
        return "/content"
    return str(Path.home() / "SmartCash")

def get_default_backbone_config() -> Dict[str, Any]:
    """
    Dapatkan konfigurasi default untuk backbone.
    
    Returns:
        Dictionary konfigurasi default
    """
    return {
        'model': {
            'backbone': 'efficientnet_b4',
            'model_type': 'efficient_basic',
            'use_attention': True,
            'use_residual': True,
            'use_ciou': False
        }
    }

def get_backbone_config(ui_components: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Dapatkan konfigurasi backbone dari config manager.
    
    Args:
        ui_components: Dictionary komponen UI (opsional)
        
    Returns:
        Dictionary konfigurasi backbone
    """
    try:
        config_manager = get_config_manager(base_dir=get_default_base_dir())
        config = config_manager.get_module_config('model')
        if config and 'model' in config:
            return config
        logger.warning("⚠️ Konfigurasi backbone tidak ditemukan, menggunakan default")
        return get_default_backbone_config()
    except Exception as e:
        logger.error(f"❌ Error saat mengambil konfigurasi backbone: {str(e)}")
        return get_default_backbone_config()

def update_config_from_ui(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update konfigurasi backbone dari UI components.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary konfigurasi yang telah diupdate
    """
    try:
        config_manager = get_config_manager(base_dir=get_default_base_dir())
        config = config_manager.get_module_config('model') or get_default_backbone_config()
        
        # Update config from UI
        if 'enabled_checkbox' in ui_components:
            config['model']['enabled'] = ui_components['enabled_checkbox'].value
            
        if 'backbone_dropdown' in ui_components:
            config['model']['backbone'] = ui_components['backbone_dropdown'].value
            
        if 'pretrained_checkbox' in ui_components:
            config['model']['pretrained'] = ui_components['pretrained_checkbox'].value
            
        if 'freeze_backbone_checkbox' in ui_components:
            config['model']['freeze_backbone'] = ui_components['freeze_backbone_checkbox'].value
            
        if 'freeze_bn_checkbox' in ui_components:
            config['model']['freeze_bn'] = ui_components['freeze_bn_checkbox'].value
            
        if 'dropout_slider' in ui_components:
            config['model']['dropout'] = ui_components['dropout_slider'].value
            
        if 'activation_dropdown' in ui_components:
            config['model']['activation'] = ui_components['activation_dropdown'].value
            
        if 'normalization_dropdown' in ui_components:
            config['model']['normalization']['type'] = ui_components['normalization_dropdown'].value
            
        if 'bn_momentum_slider' in ui_components:
            config['model']['normalization']['momentum'] = ui_components['bn_momentum_slider'].value
            
        if 'weights_path' in ui_components:
            config['model']['weights']['path'] = ui_components['weights_path'].value
            
        if 'strict_weights_checkbox' in ui_components:
            config['model']['weights']['strict'] = ui_components['strict_weights_checkbox'].value
            
        # Save config
        config_manager.save_module_config('model', config)
        
        logger.info("✅ Konfigurasi backbone berhasil diupdate")
        
        return config
        
    except Exception as e:
        logger.error(f"❌ Error saat update konfigurasi backbone: {str(e)}")
        return get_default_backbone_config()

def update_ui_from_config(ui_components: Dict[str, Any], config: Dict[str, Any] = None) -> None:
    """
    Update UI dari konfigurasi backbone.
    
    Args:
        ui_components: Dictionary komponen UI
        config: Konfigurasi yang akan digunakan (opsional)
    """
    try:
        # Get config if not provided
        if config is None:
            config = get_backbone_config(ui_components)
            
        # Ensure config has backbone key
        if 'model' not in config:
            config['model'] = get_default_backbone_config()['model']
            
        # Update UI components
        if 'enabled_checkbox' in ui_components:
            ui_components['enabled_checkbox'].value = config['model']['enabled']
            
        if 'backbone_dropdown' in ui_components:
            ui_components['backbone_dropdown'].value = config['model']['backbone']
            
        if 'pretrained_checkbox' in ui_components:
            ui_components['pretrained_checkbox'].value = config['model']['pretrained']
            
        if 'freeze_backbone_checkbox' in ui_components:
            ui_components['freeze_backbone_checkbox'].value = config['model']['freeze_backbone']
            
        if 'freeze_bn_checkbox' in ui_components:
            ui_components['freeze_bn_checkbox'].value = config['model']['freeze_bn']
            
        if 'dropout_slider' in ui_components:
            ui_components['dropout_slider'].value = config['model']['dropout']
            
        if 'activation_dropdown' in ui_components:
            ui_components['activation_dropdown'].value = config['model']['activation']
            
        if 'normalization_dropdown' in ui_components:
            ui_components['normalization_dropdown'].value = config['model']['normalization']['type']
            
        if 'bn_momentum_slider' in ui_components:
            ui_components['bn_momentum_slider'].value = config['model']['normalization']['momentum']
            
        if 'weights_path' in ui_components:
            ui_components['weights_path'].value = config['model']['weights']['path']
            
        if 'strict_weights_checkbox' in ui_components:
            ui_components['strict_weights_checkbox'].value = config['model']['weights']['strict']
            
        logger.info("✅ UI berhasil diupdate dari konfigurasi backbone")
        
    except Exception as e:
        logger.error(f"❌ Error saat mengupdate UI dari konfigurasi: {str(e)}")

def update_backbone_info(ui_components: Dict[str, Any]) -> None:
    """
    Update info panel dengan informasi backbone yang dipilih.
    
    Args:
        ui_components: Dictionary komponen UI
    """
    try:
        info_panel = ui_components.get('info_panel')
        if not info_panel:
            logger.warning(f"{ICONS.get('warning', '⚠️')} Info panel tidak ditemukan")
            return
            
        # Get current config
        config = get_backbone_config(ui_components)
        
        # Ensure config has backbone key
        if 'model' not in config:
            config['model'] = get_default_backbone_config()['model']
        
        # Update info panel dengan informasi backbone
        info_text = f"""
        <div style='font-family: monospace;'>
        <h4>Backbone Configuration:</h4>
        <ul>
            <li>Type: {config['model']['backbone']}</li>
            <li>Pretrained: {config['model']['pretrained']}</li>
            <li>Freeze Backbone: {config['model']['freeze_backbone']}</li>
            <li>Freeze BatchNorm: {config['model']['freeze_bn']}</li>
            <li>Dropout: {config['model']['dropout']}</li>
            <li>Activation: {config['model']['activation']}</li>
            <li>Normalization: {config['model']['normalization']['type']}</li>
            <li>BN Momentum: {config['model']['normalization']['momentum']}</li>
        </ul>
        </div>
        """
        
        info_panel.value = info_text
        
    except Exception as e:
        logger.error(f"{ICONS.get('error', '❌')} Error saat update info panel: {str(e)}")
        if info_panel:
            info_panel.value = f"{ICONS.get('error', '❌')} Error saat update info panel: {str(e)}"

def save_config(config: Dict[str, Any], ui_components: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Simpan konfigurasi backbone ke config manager dan sinkronkan dengan Google Drive.
    
    Args:
        config: Dictionary konfigurasi yang akan disimpan
        ui_components: Dictionary komponen UI (opsional)
        
    Returns:
        Konfigurasi yang telah disimpan dan disinkronkan
    """
    try:
        # Update status panel
        if ui_components:
            from smartcash.ui.training_config.backbone.handlers.sync_logger import update_sync_status_only
            update_sync_status_only(ui_components, "Menyimpan konfigurasi...", 'info')
        
        # Simpan konfigurasi asli untuk verifikasi
        original_config = config.copy() if config else get_default_backbone_config()
        
        # Pastikan konfigurasi memiliki struktur yang benar
        if 'model' not in original_config:
            original_config = {'model': original_config}
        
        # Simpan konfigurasi
        base_dir = get_default_base_dir()
        config_manager = get_config_manager(base_dir=base_dir)
        save_success = config_manager.save_module_config('model', original_config)
        
        if not save_success:
            if ui_components:
                from smartcash.ui.training_config.backbone.handlers.sync_logger import update_sync_status_only
                update_sync_status_only(ui_components, "Gagal menyimpan konfigurasi backbone", 'error')
            return original_config
        
        logger.info(f"{ICONS.get('success', '✅')} Konfigurasi backbone berhasil disimpan")
        
        # Log ke UI jika ui_components tersedia
        if ui_components:
            from smartcash.ui.training_config.backbone.handlers.sync_logger import update_sync_status_only
            update_sync_status_only(ui_components, "Konfigurasi backbone berhasil disimpan", 'success')
        
        # Verifikasi konfigurasi tersimpan dengan benar
        saved_config = config_manager.get_module_config('model', {})
        
        # Verifikasi konsistensi
        if 'model' in saved_config and 'model' in original_config:
            is_consistent = True
            for key, value in original_config['model'].items():
                if key not in saved_config['model'] or saved_config['model'][key] != value:
                    is_consistent = False
                    logger.warning(f"⚠️ Inkonsistensi data pada key '{key}': {value} vs {saved_config['model'].get(key, 'tidak ada')}")
                    if ui_components:
                        from smartcash.ui.training_config.backbone.handlers.sync_logger import update_sync_status_only
                        update_sync_status_only(ui_components, f"Inkonsistensi data pada key '{key}'", 'warning')
                    break
            
            if not is_consistent:
                # Coba simpan ulang jika tidak konsisten
                config_manager.save_module_config('model', original_config)
                # Log warning
                logger.warning(f"⚠️ Data tidak konsisten setelah penyimpanan, mencoba kembali")
                if ui_components:
                    from smartcash.ui.training_config.backbone.handlers.sync_logger import update_sync_status_only
                    update_sync_status_only(ui_components, "Data tidak konsisten, mencoba kembali", 'warning')
                
                # Verifikasi ulang setelah simpan ulang
                saved_config = config_manager.get_module_config('model', {})
        
        # Sinkronisasi dengan Google Drive jika di Colab
        from smartcash.ui.training_config.backbone.handlers.drive_handlers import is_colab_environment, sync_with_drive
        if is_colab_environment():
            if ui_components:
                from smartcash.ui.training_config.backbone.handlers.sync_logger import update_sync_status_only
                update_sync_status_only(ui_components, "Menyinkronkan dengan Google Drive...", 'info')
            
            # Pastikan nilai yang disinkronkan menggunakan nilai original_config
            # untuk menghindari inkonsistensi
            synced_config = sync_with_drive(original_config, ui_components)
            
            # Verifikasi konfigurasi yang disinkronkan dengan membandingkan dengan nilai asli
            is_synced_consistent = True
            if 'model' in synced_config and 'model' in original_config:
                for key, value in original_config['model'].items():
                    if key not in synced_config['model'] or synced_config['model'][key] != value:
                        is_synced_consistent = False
                        logger.warning(f"⚠️ Inkonsistensi data setelah sinkronisasi pada key '{key}': {value} vs {synced_config['model'].get(key, 'tidak ada')}")
                        if ui_components:
                            from smartcash.ui.training_config.backbone.handlers.sync_logger import update_sync_status_only
                            update_sync_status_only(ui_components, f"Inkonsistensi data setelah sinkronisasi", 'warning')
                        break
            
            if is_synced_consistent:
                if ui_components:
                    from smartcash.ui.training_config.backbone.handlers.sync_logger import update_sync_status_only
                    update_sync_status_only(ui_components, "Konfigurasi berhasil disimpan dan disinkronkan", 'success')
            
            return synced_config
        
        return saved_config
    except Exception as e:
        logger.error(f"{ICONS.get('error', '❌')} Error saat menyimpan konfigurasi backbone: {str(e)}")
        
        # Log ke UI jika ui_components tersedia
        if ui_components:
            from smartcash.ui.training_config.backbone.handlers.sync_logger import update_sync_status_only
            update_sync_status_only(ui_components, f"Error saat menyimpan konfigurasi: {str(e)}", 'error')
        
        return config

def load_config() -> Dict[str, Any]:
    """
    Muat konfigurasi backbone dari config manager.
    
    Returns:
        Dictionary konfigurasi backbone
    """
    try:
        config_manager = get_config_manager(base_dir=get_default_base_dir())
        config = config_manager.get_module_config('model')
        if config and 'model' in config:
            return config
        logger.warning("⚠️ Konfigurasi backbone tidak ditemukan, menggunakan default")
        return get_default_backbone_config()
    except Exception as e:
        logger.error(f"{ICONS.get('error', '❌')} Error saat memuat konfigurasi backbone: {str(e)}")
        return get_default_backbone_config()
