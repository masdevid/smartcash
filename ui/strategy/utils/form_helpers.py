"""
File: smartcash/ui/strategy/utils/form_helpers.py
Deskripsi: Minimal wrapper functions sesuai config_cell_initializer_guide.md
"""

from typing import Dict, Any
from smartcash.common.logger import get_logger

logger = get_logger(__name__)


def setup_strategy_event_handlers(ui_components: Dict[str, Any]) -> None:
    """Delegate to handlers - sesuai panduan"""
    try:
        from ..handlers.strategy_handlers import setup_strategy_event_handlers as setup_handlers
        setup_handlers(ui_components)
    except Exception as e:
        logger.error(f"❌ Error setting up strategy event handlers: {str(e)}")


def setup_dynamic_summary_updates(ui_components: Dict[str, Any]) -> None:
    """Delegate to handlers - sesuai panduan"""
    try:
        from ..handlers.strategy_handlers import setup_dynamic_summary_updates as setup_updates
        setup_updates(ui_components)
    except Exception as e:
        logger.error(f"❌ Error setting up dynamic updates: {str(e)}")


def validate_strategy_config(config: Dict[str, Any]) -> tuple[bool, list[str]]:
    """Delegate to handlers - avoid duplication dengan config_handler"""
    try:
        from ..handlers.config_handler import StrategyConfigHandler
        handler = StrategyConfigHandler()
        
        # Use config_handler's validation if available
        if hasattr(handler, 'validate_config'):
            return handler.validate_config(config)
        
        # Basic validation fallback
        errors = []
        if not isinstance(config, dict):
            errors.append("❌ Config harus berupa dictionary")
        
        required_sections = ['validation', 'training_utils', 'multi_scale']
        for section in required_sections:
            if section not in config:
                errors.append(f"❌ Missing section: {section}")
        
        return len(errors) == 0, errors
        
    except Exception as e:
        logger.error(f"❌ Error validating strategy config: {str(e)}")
        return False, [f"❌ Validation error: {str(e)}"]


def reset_strategy_form_to_defaults(ui_components: Dict[str, Any]) -> None:
    """Delegate to config_handler - avoid duplication"""
    try:
        from ..handlers.config_handler import StrategyConfigHandler
        from ..handlers.defaults import get_default_strategy_config
        
        handler = StrategyConfigHandler()
        default_config = get_default_strategy_config()
        handler.update_ui(ui_components, default_config)
        
        logger.info("✅ Strategy form berhasil direset ke default")
        
    except Exception as e:
        logger.error(f"❌ Error resetting strategy form: {str(e)}")


def get_strategy_form_sections() -> Dict[str, list[str]]:
    """Get widget mapping - simple organizational helper"""
    return {
        'validation': [
            'val_frequency_slider', 'iou_thres_slider', 
            'conf_thres_slider', 'max_detections_slider'
        ],
        'training_utils': [
            'experiment_name_text', 'checkpoint_dir_text',
            'tensorboard_checkbox', 'log_metrics_slider',
            'visualize_batch_slider', 'layer_mode_dropdown'
        ],
        'multi_scale': [
            'multi_scale_checkbox', 'img_size_min_slider', 'img_size_max_slider'
        ],
        'controls': ['save_button', 'reset_button']
    }


def extract_widget_values(ui_components: Dict[str, Any], widget_keys: list[str]) -> Dict[str, Any]:
    """Extract widget values - delegate to config_extractor"""
    try:
        from ..handlers.config_extractor import extract_strategy_config
        # Use existing extractor logic instead of duplicating
        full_config = extract_strategy_config(ui_components)
        
        # Filter to requested widgets if needed
        if widget_keys:
            filtered = {}
            for key in widget_keys:
                widget = ui_components.get(key)
                if widget and hasattr(widget, 'value'):
                    filtered[key] = widget.value
            return filtered
        
        return full_config
        
    except Exception as e:
        logger.error(f"❌ Error extracting widget values: {str(e)}")
        return {}


def update_widget_values(ui_components: Dict[str, Any], values: Dict[str, Any]) -> None:
    """Update widget values - delegate to config_updater"""
    try:
        from ..handlers.config_updater import update_strategy_ui
        # Use existing updater logic instead of duplicating
        update_strategy_ui(ui_components, values)
        
    except Exception as e:
        logger.error(f"❌ Error updating widget values: {str(e)}")