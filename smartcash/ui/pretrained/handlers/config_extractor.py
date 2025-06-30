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
        from smartcash.ui.pretrained.handlers.defaults import DEFAULT_MODEL_URLS
        
        config = {'pretrained_models': {}}
        pretrained_config = config['pretrained_models']
        
        # Extract models directory
        if 'models_dir_input' in ui_components:
            models_dir = ui_components['models_dir_input'].value.strip()
            pretrained_config['models_dir'] = models_dir if models_dir else '/data/pretrained'
        else:
            pretrained_config['models_dir'] = '/data/pretrained'
        
        # Hardcode model type to yolov5s
        pretrained_config['pretrained_type'] = 'yolov5s'
        
        # Extract model download URLs
        model_urls = {}
        
        # YOLOv5 URL
        if 'yolo_url_input' in ui_components:
            yolo_url = ui_components['yolo_url_input'].value.strip()
            if yolo_url and yolo_url != DEFAULT_MODEL_URLS['yolov5s']:
                model_urls['yolov5s'] = yolo_url
        
        # EfficientNet URL
        if 'efficientnet_url_input' in ui_components:
            effnet_url = ui_components['efficientnet_url_input'].value.strip()
            if effnet_url and effnet_url != DEFAULT_MODEL_URLS['efficientnet']:
                model_urls['efficientnet'] = effnet_url
        
        if model_urls:
            pretrained_config['model_urls'] = model_urls
        
        # Disable sync drive by default
        pretrained_config['sync_drive'] = False
        
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