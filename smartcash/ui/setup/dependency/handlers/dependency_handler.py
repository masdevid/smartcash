"""
File: smartcash/ui/setup/dependency/handlers/dependency_handler.py
Deskripsi: Main orchestrator untuk 3 handler utama: installation, analysis, dan status check
"""

from typing import Dict, Any, Callable
from concurrent.futures import ThreadPoolExecutor
from smartcash.ui.utils.ui_logger import get_current_ui_logger

def setup_dependency_handlers(ui_components: Dict[str, Any]) -> Dict[str, Callable]:
    """Setup semua handlers untuk dependency installer"""
    logger = ui_components.get('logger', get_current_ui_logger())
    
    try:
        # Import handler setups
        from smartcash.ui.setup.dependency.handlers.installation_handler import setup_installation_handler
        from smartcash.ui.setup.dependency.handlers.analysis_handler import setup_analysis_handler  
        from smartcash.ui.setup.dependency.handlers.status_check_handler import setup_status_check_handler
        
        # Setup individual handlers
        handlers = {}
        
        # Installation handler
        install_handler = setup_installation_handler(ui_components)
        handlers.update(install_handler)
        
        # Analysis handler
        analysis_handler = setup_analysis_handler(ui_components)
        handlers.update(analysis_handler)
        
        # Status check handler
        status_handler = setup_status_check_handler(ui_components)
        handlers.update(status_handler)
        
        # Add main orchestrator functions
        handlers['extract_current_config'] = lambda: extract_current_config(ui_components)
        handlers['apply_config_to_ui'] = lambda config: apply_config_to_ui(ui_components, config)
        handlers['reset_ui_to_defaults'] = lambda: reset_ui_to_defaults(ui_components)
        
        logger.info("🎯 Dependency handlers berhasil disetup")
        return handlers
        
    except Exception as e:
        logger.error(f"❌ Setup handlers failed: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {}

def extract_current_config(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Extract config dari UI menggunakan config handler"""
    try:
        from .config_extractor import extract_dependency_config
        return extract_dependency_config(ui_components)
    except Exception as e:
        logger = ui_components.get('logger', get_default_logger())
        logger.warning(f"⚠️ Extract config failed: {str(e)}")
        return {}

def apply_config_to_ui(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Apply config ke UI menggunakan config handler"""
    try:
        from .config_updater import update_dependency_ui
        update_dependency_ui(ui_components, config)
    except Exception as e:
        logger = ui_components.get('logger', get_default_logger())
        logger.error(f"❌ Apply config failed: {str(e)}")

def reset_ui_to_defaults(ui_components: Dict[str, Any]) -> None:
    """Reset UI ke default values"""
    try:
        from .config_updater import reset_dependency_ui
        reset_dependency_ui(ui_components)
        
        logger = ui_components.get('logger', get_default_logger())
        logger.info("✅ UI reset ke defaults")
    except Exception as e:
        logger = ui_components.get('logger', get_default_logger())
        logger.error(f"❌ Reset UI failed: {str(e)}")

def validate_ui_components(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Validate keberadaan komponen UI yang diperlukan"""
    required_components = [
        'install_button', 'analyze_button', 'check_button',
        'package_selector', 'custom_packages', 'status_panel'
    ]
    
    validation = {
        'valid': True,
        'missing_components': [],
        'available_components': list(ui_components.keys())
    }
    
    for component in required_components:
        if component not in ui_components:
            validation['missing_components'].append(component)
            validation['valid'] = False
    
    return validation

def get_handlers_status() -> Dict[str, Any]:
    """Get status dari semua handlers"""
    return {
        'installation_handler': 'available',
        'analysis_handler': 'available', 
        'status_check_handler': 'available',
        'config_handlers': 'available',
        'orchestrator': 'active'
    }