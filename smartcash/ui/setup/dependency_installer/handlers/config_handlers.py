"""
File: smartcash/ui/setup/dependency_installer/handlers/config_handlers.py
Deskripsi: SRP handler untuk save/reset config dengan consolidated utils
"""

from typing import Dict, Any
import ipywidgets as widgets
from smartcash.common.config.manager import get_config_manager
from smartcash.ui.setup.dependency_installer.utils.ui_state_utils import (
    clear_ui_outputs, update_status_panel, log_message_safe
)
from smartcash.ui.setup.dependency_installer.components.package_selector import (
    get_selected_packages, reset_package_selections
)

def setup_config_handlers(ui_components: Dict[str, Any], config: Dict[str, Any]):
    """Setup config handlers dengan consolidated utils"""
    
    config_manager = get_config_manager()
    
    def save_config_handler(button=None):
        """Save current package selections menggunakan consolidated utils"""
        
        try:
            clear_ui_outputs(ui_components, ['log_output', 'status'])
            
            # Extract current configuration menggunakan utils
            current_config = _extract_current_config_with_utils(ui_components, config)
            
            # Save config
            save_success = config_manager.save_config(
                {'dependency_installer': current_config}, 
                'dependency_installer'
            )
            
            if save_success:
                update_status_panel(ui_components, "✅ Konfigurasi dependency installer tersimpan", "success")
                log_message_safe(ui_components, "💾 Konfigurasi dependency installer berhasil disimpan", "success")
                
                # Update config reference
                config.update(current_config)
            else:
                update_status_panel(ui_components, "❌ Gagal menyimpan konfigurasi", "error")
                log_message_safe(ui_components, "💥 Gagal menyimpan konfigurasi dependency installer", "error")
        
        except Exception as e:
            error_msg = f"Error saving config: {str(e)}"
            update_status_panel(ui_components, f"❌ {error_msg}", "error")
            log_message_safe(ui_components, f"💥 {error_msg}", "error")
    
    def reset_config_handler(button=None):
        """Reset package selections menggunakan consolidated utils"""
        
        try:
            clear_ui_outputs(ui_components, ['log_output', 'status'])
            
            # Reset UI components menggunakan utils
            _reset_ui_components_with_utils(ui_components)
            
            # Get default config
            default_config = _get_default_dependency_config()
            
            # Save default config
            config_manager.save_config(
                {'dependency_installer': default_config}, 
                'dependency_installer'
            )
            
            update_status_panel(ui_components, "🔄 Konfigurasi direset ke default", "info")
            log_message_safe(ui_components, "🔄 Konfigurasi dependency installer direset ke default", "info")
            
            # Update config reference
            config.update(default_config)
            
        except Exception as e:
            error_msg = f"Error resetting config: {str(e)}"
            update_status_panel(ui_components, f"❌ {error_msg}", "error")
            log_message_safe(ui_components, f"💥 {error_msg}", "error")
    
    # Register handlers
    ui_components['save_button'].on_click(save_config_handler)
    ui_components['reset_button'].on_click(reset_config_handler)

def _extract_current_config_with_utils(ui_components: Dict[str, Any], base_config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract current configuration menggunakan utils - one-liner approach"""
    
    # Get current selections menggunakan utils
    selected_packages = get_selected_packages(ui_components)
    
    # Get UI state dengan safe fallback
    custom_packages_text = getattr(ui_components.get('custom_packages', widgets.Textarea()), 'value', '').strip()
    auto_analyze_enabled = getattr(ui_components.get('auto_analyze_checkbox', widgets.Checkbox(value=True)), 'value', True)
    
    return {
        'selected_packages': selected_packages,
        'custom_packages': custom_packages_text,
        'auto_analyze': auto_analyze_enabled,
        'installation': base_config.get('installation', _get_default_installation_config()),
        'analysis': base_config.get('analysis', _get_default_analysis_config())
    }

def _reset_ui_components_with_utils(ui_components: Dict[str, Any]):
    """Reset UI components menggunakan consolidated approach - one-liner"""
    
    # Reset package selections menggunakan utils
    reset_package_selections(ui_components)
    
    # Reset other UI components dengan safe attribute setting
    ui_components.get('custom_packages') and setattr(ui_components['custom_packages'], 'value', '')
    ui_components.get('auto_analyze_checkbox') and setattr(ui_components['auto_analyze_checkbox'], 'value', True)

def _get_default_dependency_config() -> Dict[str, Any]:
    """Get default dependency installer config - one-liner structure"""
    return {
        'selected_packages': [],
        'custom_packages': '',
        'auto_analyze': True,
        'installation': _get_default_installation_config(),
        'analysis': _get_default_analysis_config()
    }

def _get_default_installation_config() -> Dict[str, Any]:
    """Get default installation config - one-liner"""
    return {
        'parallel_workers': 3,
        'force_reinstall': False,
        'use_cache': True,
        'timeout': 300
    }

def _get_default_analysis_config() -> Dict[str, Any]:
    """Get default analysis config - one-liner"""
    return {
        'check_compatibility': True,
        'include_dev_deps': False
    }

def apply_saved_config_to_ui(ui_components: Dict[str, Any], saved_config: Dict[str, Any]):
    """Apply saved config ke UI components - one-liner approach"""
    
    try:
        # Apply custom packages
        custom_packages = saved_config.get('custom_packages', '')
        ui_components.get('custom_packages') and setattr(ui_components['custom_packages'], 'value', custom_packages)
        
        # Apply auto-analyze setting
        auto_analyze = saved_config.get('auto_analyze', True)
        ui_components.get('auto_analyze_checkbox') and setattr(ui_components['auto_analyze_checkbox'], 'value', auto_analyze)
        
        # Apply package selections (would need package selector utils untuk detailed implementation)
        # For now, packages default ke their default values dari package_selector
        
        log_message_safe(ui_components, "📋 Saved configuration applied to UI", "info")
        
    except Exception as e:
        log_message_safe(ui_components, f"⚠️ Error applying saved config: {str(e)}", "warning")

def get_config_summary(config: Dict[str, Any]) -> str:
    """Get config summary untuk display - one-liner"""
    
    selected_count = len(config.get('selected_packages', []))
    custom_count = len([pkg for pkg in config.get('custom_packages', '').split('\n') if pkg.strip()])
    total_packages = selected_count + custom_count
    auto_analyze = config.get('auto_analyze', True)
    
    return f"📊 Config: {total_packages} packages selected, auto-analyze: {'✅' if auto_analyze else '❌'}"