
"""
File: smartcash/ui/dataset/download/handlers/config_handlers.py
Deskripsi: Updated config handlers dengan Drive path integration
"""

from smartcash.common.config.manager import get_config_manager
from smartcash.common.environment import get_environment_manager

def setup_config_handlers(ui_components: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Setup config handlers dengan Drive path integration."""
    
    try:
        # Environment manager untuk Drive detection
        env_manager = get_environment_manager()
        ui_components['env_manager'] = env_manager
        
        # Get config manager
        config_manager = get_config_manager()
        
        # Load saved config
        if hasattr(config_manager, 'get_config'):
            saved_config = config_manager.get_config('dataset')
        else:
            saved_config = getattr(config_manager, 'config', {}).get('dataset', {})
        
        # Merge config dengan Drive path adjustments
        merged_config = {**config, **saved_config}
        
        # Adjust paths untuk Drive jika perlu
        if env_manager.is_colab and env_manager.is_drive_mounted:
            merged_config = _adjust_paths_for_drive(merged_config, env_manager)
        
        # Update UI dari config
        _update_ui_from_config(ui_components, merged_config)
        
        # Setup API key detection
        _setup_api_key_detection(ui_components)
        
        # Update storage info widget
        _update_storage_info_widget(ui_components, env_manager)
        
    except Exception as e:
        logger = ui_components.get('logger')
        if logger:
            logger.warning(f"⚠️ Gagal load config: {str(e)}")
    
    return ui_components

def _adjust_paths_for_drive(config: Dict[str, Any], env_manager) -> Dict[str, Any]:
    """Adjust config paths untuk Drive storage."""
    adjusted_config = config.copy()
    
    # Adjust output directory ke Drive
    if 'output_dir' not in adjusted_config or adjusted_config['output_dir'] == 'data':
        adjusted_config['output_dir'] = str(env_manager.drive_path / 'downloads')
    
    # Adjust backup directory ke Drive
    if 'backup_dir' not in adjusted_config or adjusted_config['backup_dir'] == 'data/backup':
        adjusted_config['backup_dir'] = str(env_manager.drive_path / 'backups')
    
    return adjusted_config

def _update_ui_from_config(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Update UI components dari config."""
    config_mapping = {
        'workspace': 'workspace',
        'project': 'project',
        'version': 'version',
        'output_dir': 'output_dir',
        'backup_dir': 'backup_dir',
        'validate_dataset': 'validate_dataset',
        'backup_before_download': 'backup_checkbox'
    }
    
    for config_key, ui_key in config_mapping.items():
        if config_key in config and ui_key in ui_components:
            if hasattr(ui_components[ui_key], 'value'):
                ui_components[ui_key].value = config[config_key]

def _update_storage_info_widget(ui_components: Dict[str, Any], env_manager) -> None:
    """Update storage info widget berdasarkan Drive status."""
    if 'drive_info' in ui_components:
        if env_manager.is_drive_mounted:
            info_html = f"""
            <div style="background: #e8f5e8; border: 1px solid #4caf50; border-radius: 4px; padding: 8px; margin: 5px 0;">
                <span style="color: #2e7d32;">✅ Dataset akan disimpan di Google Drive: {env_manager.drive_path}</span>
            </div>
            """
        else:
            info_html = f"""
            <div style="background: #fff3cd; border: 1px solid #ffc107; border-radius: 4px; padding: 8px; margin: 5px 0;">
                <span style="color: #856404;">⚠️ Drive tidak terhubung - dataset akan disimpan lokal</span>
            </div>
            """
        ui_components['drive_info'].value = info_html

def _setup_api_key_detection(ui_components: Dict[str, Any]) -> None:
    """Setup API key detection dengan prioritas sources."""
    if 'api_key' in ui_components and not ui_components['api_key'].value:
        # Coba dari environment
        api_key = os.environ.get('ROBOFLOW_API_KEY', '')
        
        # Coba dari Colab secrets
        if not api_key:
            try:
                from google.colab import userdata
                api_key = userdata.get('ROBOFLOW_API_KEY', '')
            except:
                pass
        
        if api_key:
            ui_components['api_key'].value = api_key
            logger = ui_components.get('logger')
            if logger:
                logger.success("🔑 API key Roboflow ditemukan")