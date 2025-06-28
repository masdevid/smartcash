# File: smartcash/ui/pretrained/handlers/__init__.py
"""
File: smartcash/ui/pretrained/handlers/__init__.py
Deskripsi: Handlers package initialization dengan simplified structure
"""

# Config management
from .config_handler import PretrainedConfigHandler
from .config_extractor import extract_pretrained_config, validate_pretrained_config
from .config_updater import update_pretrained_ui, reset_pretrained_ui
from .defaults import (
    get_default_pretrained_config, 
    get_model_variants,
    get_model_descriptions,
    get_model_info
)

# Event handlers - Optional import
try:
    from .event_handlers import setup_all_handlers
    _HAS_EVENT_HANDLERS = True
except ImportError:
    _HAS_EVENT_HANDLERS = False
    
    def setup_all_handlers(ui_components, config, **kwargs):
        """Fallback handler setup"""
        from smartcash.common.logger import get_logger
        logger = get_logger(__name__)
        logger.warning("⚠️ Event handlers not available, using basic setup")
        return ui_components

__all__ = [
    # Config management
    'PretrainedConfigHandler',
    'extract_pretrained_config',
    'validate_pretrained_config',
    'update_pretrained_ui',
    'reset_pretrained_ui',
    'get_default_pretrained_config',
    'get_model_variants',
    'get_model_descriptions',
    'get_model_info',
    
    # Event handlers
    'setup_all_handlers'
]