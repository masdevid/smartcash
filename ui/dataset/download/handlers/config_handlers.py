"""
File: smartcash/ui/dataset/download/handlers/config_handlers.py
Deskripsi: Fixed config handlers dengan Drive integration, API key detection, dan status panel yang akurat
"""

import os
from typing import Dict, Any
from smartcash.common.config.manager import get_config_manager
from smartcash.common.environment import get_environment_manager

def setup_config_handlers(ui_components: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Setup config handlers dengan perbaikan lengkap."""
    
    try:
        # Environment manager untuk Drive detection
        env_manager = get_environment_manager()
        ui_components['env_manager'] = env_manager
        
        # Update status panel dengan Drive status yang akurat
        _update_status_panel_accurate(ui_components, env_manager)
        
        # Get config manager
        config_manager = get_config_manager()
        
        # Load saved config dengan error handling
        saved_config = _load_saved_config_safe(config_manager)
        
        # Merge config dengan Drive path adjustments
        merged_config = {**config, **saved_config}
        
        # Adjust paths untuk Drive jika perlu
        if env_manager.is_colab and env_manager.is_drive_mounted:
            merged_config = _adjust_paths_for_drive(merged_config, env_manager)
        
        # Setup API key detection dari semua sumber
        api_key = _detect_api_key_comprehensive()
        if api_key:
            merged_config['api_key'] = api_key
        
        # Update UI dari config dengan defaults
        _update_ui_from_config_with_defaults(ui_components, merged_config)
        
        # Update storage info widget
        _update_storage_info_widget(ui_components, env_manager)
        
        # Set default values untuk components
        _set_component_defaults(ui_components, env_manager)
        
    except Exception as e:
        logger = ui_components.get('logger')
        if logger:
            logger.warning(f"⚠️ Gagal load config: {str(e)}")
        # Set minimal defaults jika error
        _set_minimal_defaults(ui_components)
    
    return ui_components

def _update_status_panel_accurate(ui_components: Dict[str, Any], env_manager) -> None:
    """Update status panel dengan informasi Drive yang akurat."""
    if 'status_panel' in ui_components:
        if env_manager.is_colab and env_manager.is_drive_mounted:
            status_html = """
            <div style="background: #e8f5e8; border: 1px solid #4caf50; border-radius: 4px; padding: 10px; margin: 5px 0;">
                <span style="color: #2e7d32; font-weight: bold;">✅ Drive terhubung - Siap download ke Google Drive</span>
            </div>
            """
        elif env_manager.is_colab:
            status_html = """
            <div style="background: #fff3cd; border: 1px solid #ffc107; border-radius: 4px; padding: 10px; margin: 5px 0;">
                <span style="color: #856404; font-weight: bold;">⚠️ Colab terdeteksi tapi Drive tidak terhubung - Dataset akan hilang saat restart</span>
            </div>
            """
        else:
            status_html = """
            <div style="background: #e3f2fd; border: 1px solid #2196f3; border-radius: 4px; padding: 10px; margin: 5px 0;">
                <span style="color: #1565c0; font-weight: bold;">ℹ️ Environment lokal - Dataset akan disimpan lokal</span>
            </div>
            """
        ui_components['status_panel'].value = status_html

def _load_saved_config_safe(config_manager) -> Dict[str, Any]:
    """Load config dengan error handling yang aman."""
    try:
        if hasattr(config_manager, 'get_config'):
            return config_manager.get_config('dataset') or {}
        elif hasattr(config_manager, 'config'):
            return getattr(config_manager, 'config', {}).get('dataset', {})
        else:
            return {}
    except Exception:
        return {}

def _detect_api_key_comprehensive() -> str:
    """Deteksi API key dari semua sumber yang tersedia."""
    # 1. Environment variable
    api_key = os.environ.get('ROBOFLOW_API_KEY', '')
    if api_key:
        return api_key
    
    # 2. Google Colab userdata (Google Secrets)
    try:
        from google.colab import userdata
        api_key = userdata.get('ROBOFLOW_API_KEY', '')
        if api_key:
            return api_key
    except Exception:
        pass
    
    # 3. Check variants nama secret
    secret_variants = ['ROBOFLOW_API_KEY', 'roboflow_api_key', 'ROBOFLOW_KEY', 'roboflow_key']
    for variant in secret_variants:
        try:
            from google.colab import userdata
            api_key = userdata.get(variant, '')
            if api_key:
                return api_key
        except Exception:
            continue
    
    return ''

def _adjust_paths_for_drive(config: Dict[str, Any], env_manager) -> Dict[str, Any]:
    """Adjust config paths untuk Drive storage."""
    adjusted_config = config.copy()
    
    # Adjust output directory ke Drive
    if 'output_dir' not in adjusted_config or not adjusted_config['output_dir'] or adjusted_config['output_dir'] == 'data':
        adjusted_config['output_dir'] = str(env_manager.drive_path / 'downloads')
    
    # Adjust backup directory ke Drive
    if 'backup_dir' not in adjusted_config or not adjusted_config['backup_dir'] or adjusted_config['backup_dir'] == 'data/backup':
        adjusted_config['backup_dir'] = str(env_manager.drive_path / 'backups')
    
    return adjusted_config

def _update_ui_from_config_with_defaults(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Update UI components dari config dengan defaults yang benar."""
    # Default values yang akan dipakai untuk reset
    defaults = _get_default_values(ui_components.get('env_manager'))
    
    config_mapping = {
        'workspace': ('workspace', defaults['workspace']),
        'project': ('project', defaults['project']),
        'version': ('version', defaults['version']),
        'api_key': ('api_key', defaults['api_key']),
        'output_dir': ('output_dir', defaults['output_dir']),
        'backup_dir': ('backup_dir', defaults['backup_dir']),
        'validate_dataset': ('validate_dataset', defaults['validate_dataset']),
        'backup_before_download': ('backup_checkbox', defaults['backup_checkbox'])
    }
    
    for config_key, (ui_key, default_value) in config_mapping.items():
        if ui_key in ui_components and hasattr(ui_components[ui_key], 'value'):
            # Gunakan nilai dari config, atau default jika tidak ada
            value = config.get(config_key, default_value)
            ui_components[ui_key].value = value
    
    # Store defaults untuk fungsi reset
    ui_components['_defaults'] = defaults

def _get_default_values(env_manager=None) -> Dict[str, Any]:
    """Get default values berdasarkan environment."""
    # Deteksi API key
    api_key = _detect_api_key_comprehensive()
    
    # Default paths berdasarkan environment
    if env_manager and env_manager.is_colab and env_manager.is_drive_mounted:
        output_dir = str(env_manager.drive_path / 'downloads')
        backup_dir = str(env_manager.drive_path / 'backups')
    else:
        output_dir = 'data'
        backup_dir = 'data/backup'
    
    return {
        'workspace': 'smartcash-wo2us',
        'project': 'rupiah-emisi-2022',
        'version': '3',
        'api_key': api_key,
        'output_dir': output_dir,
        'backup_dir': backup_dir,
        'validate_dataset': True,
        'backup_checkbox': False
    }

def _set_component_defaults(ui_components: Dict[str, Any], env_manager) -> None:
    """Set default values ke components jika belum ada."""
    defaults = _get_default_values(env_manager)
    
    for key, default_value in defaults.items():
        ui_key = 'backup_checkbox' if key == 'backup_checkbox' else key
        if ui_key in ui_components and hasattr(ui_components[ui_key], 'value'):
            if not ui_components[ui_key].value:  # Jika kosong, set default
                ui_components[ui_key].value = default_value

def _set_minimal_defaults(ui_components: Dict[str, Any]) -> None:
    """Set minimal defaults jika terjadi error saat load config."""
    minimal_defaults = {
        'workspace': 'smartcash-wo2us',
        'project': 'rupiah-emisi-2022', 
        'version': '3',
        'output_dir': 'data',
        'backup_dir': 'data/backup',
        'validate_dataset': True,
        'backup_checkbox': False
    }
    
    for key, value in minimal_defaults.items():
        ui_key = 'backup_checkbox' if key == 'backup_checkbox' else key
        if ui_key in ui_components and hasattr(ui_components[ui_key], 'value'):
            ui_components[ui_key].value = value
    
    ui_components['_defaults'] = minimal_defaults

def _update_storage_info_widget(ui_components: Dict[str, Any], env_manager) -> None:
    """Update storage info widget berdasarkan Drive status yang akurat."""
    if 'drive_info' in ui_components:
        if env_manager.is_colab and env_manager.is_drive_mounted:
            info_html = f"""
            <div style="background: #e8f5e8; border: 1px solid #4caf50; border-radius: 4px; padding: 8px; margin: 5px 0;">
                <span style="color: #2e7d32;">✅ Dataset akan disimpan di Google Drive: {env_manager.drive_path}</span>
            </div>
            """
        elif env_manager.is_colab:
            info_html = """
            <div style="background: #fff3cd; border: 1px solid #ffc107; border-radius: 4px; padding: 8px; margin: 5px 0;">
                <span style="color: #856404;">⚠️ Drive tidak terhubung - dataset akan disimpan lokal (hilang saat restart)</span>
            </div>
            """
        else:
            info_html = """
            <div style="background: #e3f2fd; border: 1px solid #2196f3; border-radius: 4px; padding: 8px; margin: 5px 0;">
                <span style="color: #1565c0;">ℹ️ Environment lokal - dataset akan disimpan lokal</span>
            </div>
            """
        ui_components['drive_info'].value = info_html