# File: smartcash/ui/pretrained/handlers/config_extractor.py
"""
File: smartcash/ui/pretrained/handlers/config_extractor.py
Deskripsi: Config extractor untuk pretrained models dengan fail-fast approach
"""

from typing import Dict, Any
from smartcash.common.logger import get_logger

logger = get_logger(__name__)

def extract_pretrained_config(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """🔧 Extract konfigurasi dari UI components pretrained models
    
    Args:
        ui_components: Dictionary berisi komponen UI
        
    Returns:
        Dictionary berisi ekstraksi konfigurasi pretrained models
        
    Raises:
        ValueError: Jika ekstraksi config gagal atau data tidak valid
    """
    if not ui_components or not isinstance(ui_components, dict):
        raise ValueError("❌ UI components tidak valid untuk ekstraksi config")
    
    try:
        config = {'pretrained_models': {}}
        pretrained_config = config['pretrained_models']
        
        # Extract models directory
        if 'models_dir_input' in ui_components:
            models_dir = ui_components['models_dir_input'].value.strip()
            pretrained_config['models_dir'] = models_dir if models_dir else '/content/models'
        else:
            pretrained_config['models_dir'] = '/content/models'
        
        # Extract drive models directory
        if 'drive_models_dir_input' in ui_components:
            drive_dir = ui_components['drive_models_dir_input'].value.strip()
            pretrained_config['drive_models_dir'] = drive_dir if drive_dir else '/data/pretrained'
        else:
            pretrained_config['drive_models_dir'] = '/data/pretrained'
        
        # Extract pretrained type
        if 'pretrained_type_dropdown' in ui_components:
            pretrained_type = ui_components['pretrained_type_dropdown'].value
            pretrained_config['pretrained_type'] = pretrained_type if pretrained_type else 'yolov5s'
        else:
            pretrained_config['pretrained_type'] = 'yolov5s'
        
        # Extract auto download setting
        if 'auto_download_checkbox' in ui_components:
            pretrained_config['auto_download'] = ui_components['auto_download_checkbox'].value
        else:
            pretrained_config['auto_download'] = False
        
        # Extract sync drive setting
        if 'sync_drive_checkbox' in ui_components:
            pretrained_config['sync_drive'] = ui_components['sync_drive_checkbox'].value
        else:
            pretrained_config['sync_drive'] = True
        
        logger.info("✅ Config berhasil diekstrak dari UI components")
        return config
        
    except Exception as e:
        error_msg = f"❌ Gagal mengekstrak config dari UI: {str(e)}"
        logger.error(error_msg)
        raise ValueError(error_msg) from e


def validate_pretrained_config(config: Dict[str, Any]) -> bool:
    """🔍 Validasi konfigurasi pretrained models
    
    Args:
        config: Dictionary konfigurasi untuk divalidasi
        
    Returns:
        True jika valid, False jika tidak
    """
    try:
        if not isinstance(config, dict):
            return False
        
        pretrained_models = config.get('pretrained_models', {})
        if not isinstance(pretrained_models, dict):
            return False
        
        # Validasi required fields
        required_fields = ['models_dir', 'pretrained_type']
        for field in required_fields:
            if field not in pretrained_models or not pretrained_models[field]:
                logger.warning(f"⚠️ Required field missing: {field}")
                return False
        
        # Validasi pretrained_type values - Simplified to YOLOv5s only
        valid_types = ['yolov5s']
        if pretrained_models['pretrained_type'] not in valid_types:
            logger.warning(f"⚠️ Invalid pretrained_type: {pretrained_models['pretrained_type']}, using yolov5s")
            return False
        
        logger.info("✅ Config validation passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error validating config: {str(e)}")
        return False