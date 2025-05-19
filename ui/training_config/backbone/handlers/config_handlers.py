"""
File: smartcash/ui/training_config/backbone/handlers/config_handlers.py
Deskripsi: Handler konfigurasi untuk backbone model
"""

from typing import Dict, Any, Optional
import os
import yaml
from pathlib import Path
from smartcash.common.config import get_config_manager
from smartcash.common.logger import get_logger
from smartcash.ui.utils.constants import ICONS

logger = get_logger(__name__)

def get_default_base_dir():
    if "COLAB_GPU" in os.environ or "COLAB_TPU_ADDR" in os.environ:
        return "/content"
    return str(Path.home() / "SmartCash")

def get_default_backbone_config() -> Dict[str, Any]:
    """
    Dapatkan konfigurasi default untuk backbone model.
    
    Returns:
        Dictionary konfigurasi default
    """
    return {
        "backbone": {
            "enabled": True,
            "type": "efficientnet_b4",
            "pretrained": True,
            "freeze_backbone": False,
            "freeze_bn": True,
            "dropout": 0.2,
            "activation": "relu",
            "normalization": {
                "type": "batch",
                "momentum": 0.1
            },
            "weights": {
                "path": None,
                "strict": True
            }
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
        config = config_manager.get_module_config('backbone')
        if config:
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
        config = config_manager.get_module_config('backbone') or get_default_backbone_config()
        
        # Update config from UI
        if 'enabled_checkbox' in ui_components:
            config['backbone']['enabled'] = ui_components['enabled_checkbox'].value
            
        if 'backbone_dropdown' in ui_components:
            config['backbone']['type'] = ui_components['backbone_dropdown'].value
            
        if 'pretrained_checkbox' in ui_components:
            config['backbone']['pretrained'] = ui_components['pretrained_checkbox'].value
            
        if 'freeze_backbone_checkbox' in ui_components:
            config['backbone']['freeze_backbone'] = ui_components['freeze_backbone_checkbox'].value
            
        if 'freeze_bn_checkbox' in ui_components:
            config['backbone']['freeze_bn'] = ui_components['freeze_bn_checkbox'].value
            
        if 'dropout_slider' in ui_components:
            config['backbone']['dropout'] = ui_components['dropout_slider'].value
            
        if 'activation_dropdown' in ui_components:
            config['backbone']['activation'] = ui_components['activation_dropdown'].value
            
        if 'normalization_dropdown' in ui_components:
            config['backbone']['normalization']['type'] = ui_components['normalization_dropdown'].value
            
        if 'bn_momentum_slider' in ui_components:
            config['backbone']['normalization']['momentum'] = ui_components['bn_momentum_slider'].value
            
        if 'weights_path' in ui_components:
            config['backbone']['weights']['path'] = ui_components['weights_path'].value
            
        if 'strict_weights_checkbox' in ui_components:
            config['backbone']['weights']['strict'] = ui_components['strict_weights_checkbox'].value
            
        # Save config
        config_manager.save_module_config('backbone', config)
        
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
            
        # Update UI components
        if 'enabled_checkbox' in ui_components:
            ui_components['enabled_checkbox'].value = config['backbone']['enabled']
            
        if 'backbone_dropdown' in ui_components:
            ui_components['backbone_dropdown'].value = config['backbone']['type']
            
        if 'pretrained_checkbox' in ui_components:
            ui_components['pretrained_checkbox'].value = config['backbone']['pretrained']
            
        if 'freeze_backbone_checkbox' in ui_components:
            ui_components['freeze_backbone_checkbox'].value = config['backbone']['freeze_backbone']
            
        if 'freeze_bn_checkbox' in ui_components:
            ui_components['freeze_bn_checkbox'].value = config['backbone']['freeze_bn']
            
        if 'dropout_slider' in ui_components:
            ui_components['dropout_slider'].value = config['backbone']['dropout']
            
        if 'activation_dropdown' in ui_components:
            ui_components['activation_dropdown'].value = config['backbone']['activation']
            
        if 'normalization_dropdown' in ui_components:
            ui_components['normalization_dropdown'].value = config['backbone']['normalization']['type']
            
        if 'bn_momentum_slider' in ui_components:
            ui_components['bn_momentum_slider'].value = config['backbone']['normalization']['momentum']
            
        if 'weights_path' in ui_components:
            ui_components['weights_path'].value = config['backbone']['weights']['path']
            
        if 'strict_weights_checkbox' in ui_components:
            ui_components['strict_weights_checkbox'].value = config['backbone']['weights']['strict']
            
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
        
        # Update info panel dengan informasi backbone
        info_text = f"""
        <div style='font-family: monospace;'>
        <h4>Backbone Configuration:</h4>
        <ul>
            <li>Type: {config['backbone']['type']}</li>
            <li>Pretrained: {config['backbone']['pretrained']}</li>
            <li>Freeze Backbone: {config['backbone']['freeze_backbone']}</li>
            <li>Freeze BatchNorm: {config['backbone']['freeze_bn']}</li>
            <li>Dropout: {config['backbone']['dropout']}</li>
            <li>Activation: {config['backbone']['activation']}</li>
            <li>Normalization: {config['backbone']['normalization']['type']}</li>
            <li>BN Momentum: {config['backbone']['normalization']['momentum']}</li>
        </ul>
        </div>
        """
        
        info_panel.value = info_text
        
    except Exception as e:
        logger.error(f"{ICONS.get('error', '❌')} Error saat update info panel: {str(e)}")
        if info_panel:
            info_panel.value = f"{ICONS.get('error', '❌')} Error saat update info panel: {str(e)}"
