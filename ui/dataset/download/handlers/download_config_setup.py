"""
File: smartcash/ui/dataset/download/handlers/download_config_setup.py
Deskripsi: Setup config handlers untuk download dengan environment-aware configuration
"""

from typing import Dict, Any
from smartcash.common.environment import get_environment_manager
from smartcash.common.constants.paths import get_paths_for_environment
from smartcash.common.config.manager import get_config_manager

def setup_download_config_handlers(ui_components: Dict[str, Any], config: Dict[str, Any], env=None) -> Dict[str, Any]:
    """Setup config handlers dengan environment-aware configuration."""
    logger = ui_components.get('logger')
    
    try:
        # Environment manager dengan status refresh
        env_manager = get_environment_manager()
        env_manager.refresh_drive_status()
        ui_components['env_manager'] = env_manager
        
        # Get environment-appropriate paths
        paths = get_paths_for_environment(
            is_colab=env_manager.is_colab,
            is_drive_mounted=env_manager.is_drive_mounted
        )
        ui_components['paths'] = paths
        
        # Update status panel dengan environment info
        _update_status_panel_with_env_info(ui_components, env_manager)
        
        # Load dan merge configurations
        config_manager = get_config_manager()
        saved_config = _load_saved_config_safe(config_manager)
        merged_config = _merge_configs(config, saved_config, paths)
        
        # Auto-detect API key
        api_key = _detect_api_key()
        if api_key:
            merged_config['api_key'] = api_key
        
        # Update UI components dengan merged config
        _update_ui_components_from_config(ui_components, merged_config, paths)
        
        # Store defaults untuk reset functionality
        ui_components['_config_defaults'] = _create_environment_defaults(paths, api_key, env_manager)
        
        logger and logger.debug("✅ Config handlers setup berhasil")
        
    except Exception as e:
        logger and logger.warning(f"⚠️ Error setup config: {str(e)}")
        _set_minimal_fallback_config(ui_components)
    
    return ui_components

def _update_status_panel_with_env_info(ui_components: Dict[str, Any], env_manager) -> None:
    """Update status panel dengan comprehensive environment info."""
    if 'status_panel' not in ui_components:
        return
    
    if env_manager.is_colab and env_manager.is_drive_mounted:
        status_html = f"""
        <div style="background: #e8f5e8; border: 1px solid #4caf50; border-radius: 6px; padding: 12px; margin: 8px 0;">
            <div style="color: #2e7d32; font-weight: bold; font-size: 14px;">
                ✅ Google Colab + Drive Connected
            </div>
            <div style="color: #388e3c; font-size: 12px; margin-top: 4px;">
                📁 Storage: {env_manager.drive_path}<br>
                💾 Dataset akan tersimpan permanen di Google Drive
            </div>
        </div>
        """
    elif env_manager.is_colab:
        status_html = """
        <div style="background: #fff3cd; border: 1px solid #ffc107; border-radius: 6px; padding: 12px; margin: 8px 0;">
            <div style="color: #856404; font-weight: bold; font-size: 14px;">
                ⚠️ Google Colab - Drive Not Connected
            </div>
            <div style="color: #856404; font-size: 12px; margin-top: 4px;">
                📁 Storage: Local (temporary)<br>
                ⏰ Dataset akan hilang saat runtime restart
            </div>
        </div>
        """
    else:
        status_html = """
        <div style="background: #e3f2fd; border: 1px solid #2196f3; border-radius: 6px; padding: 12px; margin: 8px 0;">
            <div style="color: #1565c0; font-weight: bold; font-size: 14px;">
                ℹ️ Local Environment
            </div>
            <div style="color: #1976d2; font-size: 12px; margin-top: 4px;">
                📁 Storage: Local filesystem<br>
                💻 Running on local machine
            </div>
        </div>
        """
    
    ui_components['status_panel'].value = status_html

def _merge_configs(base_config: Dict[str, Any], saved_config: Dict[str, Any], paths: Dict[str, str]) -> Dict[str, Any]:
    """Merge configurations dengan environment-aware paths."""
    merged = base_config.copy()
    
    # Merge saved config jika ada
    if saved_config:
        merged.update(saved_config)
    
    # Ensure current environment paths
    merged['output_dir'] = paths['downloads']
    merged['backup_dir'] = paths['backup']
    
    return merged

def _update_ui_components_from_config(ui_components: Dict[str, Any], config: Dict[str, Any], paths: Dict[str, str]) -> None:
    """Update UI components dari merged config."""
    field_mapping = {
        'workspace': ('workspace', 'smartcash-wo2us'),
        'project': ('project', 'rupiah-emisi-2022'),
        'version': ('version', '3'),
        'output_dir': ('output_dir', paths['downloads']),
        'backup_dir': ('backup_dir', paths['backup']),
        'backup_before_download': ('backup_checkbox', False),
        'organize_dataset': ('organize_dataset', True)
    }
    
    for config_key, (ui_key, default_value) in field_mapping.items():
        if ui_key in ui_components and hasattr(ui_components[ui_key], 'value'):
            value = config.get(config_key, default_value)
            ui_components[ui_key].value = value

def _create_environment_defaults(paths: Dict[str, str], api_key: str, env_manager) -> Dict[str, Any]:
    """Create environment-aware defaults."""
    return {
        'workspace': 'smartcash-wo2us',
        'project': 'rupiah-emisi-2022',
        'version': '3',
        'api_key': api_key,
        'output_dir': paths['downloads'],
        'backup_dir': paths['backup'],
        'backup_before_download': False,
        'organize_dataset': True,
        'environment_type': 'Google Drive' if env_manager.is_drive_mounted else 'Local Storage',
        'storage_persistent': env_manager.is_drive_mounted
    }

def _detect_api_key() -> str:
    """Detect API key dari environment sources."""
    import os
    
    # Environment variables
    for env_key in ['ROBOFLOW_API_KEY', 'ROBOFLOW_KEY', 'RF_API_KEY']:
        api_key = os.environ.get(env_key, '').strip()
        if api_key and len(api_key) > 10:
            return api_key
    
    # Google Colab userdata
    try:
        from google.colab import userdata
        for key_name in ['ROBOFLOW_API_KEY', 'roboflow_api_key', 'ROBOFLOW_KEY', 'API_KEY']:
            try:
                api_key = userdata.get(key_name, '').strip()
                if api_key and len(api_key) > 10:
                    return api_key
            except:
                continue
    except:
        pass
    
    return ''

def _load_saved_config_safe(config_manager) -> Dict[str, Any]:
    """Load saved config dengan comprehensive error handling."""
    try:
        # Try different config loading methods
        if hasattr(config_manager, 'get_config'):
            return config_manager.get_config('download') or {}
        elif hasattr(config_manager, 'load_config'):
            return config_manager.load_config('download') or {}
        elif hasattr(config_manager, 'config'):
            return getattr(config_manager, 'config', {}).get('download', {})
        else:
            return {}
    except Exception:
        return {}

def _set_minimal_fallback_config(ui_components: Dict[str, Any]) -> None:
    """Set minimal fallback config jika terjadi error."""
    minimal_values = {
        'workspace': 'smartcash-wo2us',
        'project': 'rupiah-emisi-2022',
        'version': '3',
        'output_dir': 'data/downloads',
        'backup_dir': 'data/backup',
        'backup_checkbox': False,
        'organize_dataset': True
    }
    
    for key, value in minimal_values.items():
        if key in ui_components and hasattr(ui_components[key], 'value'):
            try:
                ui_components[key].value = value
            except:
                pass

def get_config_setup_status(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Get status config setup untuk debugging."""
    env_manager = ui_components.get('env_manager')
    paths = ui_components.get('paths', {})
    defaults = ui_components.get('_config_defaults', {})
    
    # Check UI component values
    ui_values = {}
    for key in ['workspace', 'project', 'version', 'output_dir', 'backup_dir', 'backup_checkbox', 'organize_dataset']:
        if key in ui_components and hasattr(ui_components[key], 'value'):
            ui_values[key] = ui_components[key].value
    
    return {
        'environment': {
            'is_colab': env_manager.is_colab if env_manager else False,
            'drive_mounted': env_manager.is_drive_mounted if env_manager else False,
            'storage_type': defaults.get('environment_type', 'Unknown')
        },
        'paths': paths,
        'defaults_available': bool(defaults),
        'ui_values': ui_values,
        'api_key_detected': bool(defaults.get('api_key')),
        'config_setup_complete': bool(env_manager and paths and defaults)
    }