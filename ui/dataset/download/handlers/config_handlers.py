"""
File: smartcash/ui/dataset/download/handlers/config_handlers.py  
Deskripsi: Updated config handlers dengan path defaults yang benar dan konsisten
"""

import os
from typing import Dict, Any
from smartcash.common.config.manager import get_config_manager
from smartcash.common.environment import get_environment_manager
from smartcash.common.constants.paths import get_paths_for_environment

def setup_config_handlers(ui_components: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Setup config handlers dengan path management yang benar."""
    
    print("🔧 Setting up config handlers...")
    
    try:
        # Get environment manager dan refresh status
        env_manager = get_environment_manager()
        env_manager.refresh_drive_status()
        ui_components['env_manager'] = env_manager
        
        # Get paths berdasarkan environment
        paths = get_paths_for_environment(
            is_colab=env_manager.is_colab,
            is_drive_mounted=env_manager.is_drive_mounted
        )
        ui_components['paths'] = paths
        
        # Update status panel dengan info yang akurat
        _update_status_panel_with_env_info(ui_components, env_manager)
        
        # Load saved config
        config_manager = get_config_manager()
        saved_config = _load_saved_config_safe(config_manager)
        
        # Merge dengan priority: saved_config > passed_config > defaults
        merged_config = _merge_configs(config, saved_config, paths)
        
        # CRITICAL: Load API key dari semua sumber dan update UI
        api_key = _detect_and_load_api_key(ui_components)
        if api_key:
            merged_config['api_key'] = api_key
        
        # Update semua UI components dari merged config
        _update_all_ui_components(ui_components, merged_config, paths)
        
        # Update storage info widget
        _update_storage_info_widget(ui_components, env_manager)
        
        # Store defaults untuk reset function
        ui_components['_defaults'] = _create_smart_defaults(paths, api_key)
        
        print("✅ Config handlers setup completed successfully")
        
    except Exception as e:
        logger = ui_components.get('logger')
        if logger:
            logger.warning(f"⚠️ Error setup config: {str(e)}")
        print(f"❌ Config setup error: {str(e)}")
        # Set minimal fallback
        _set_minimal_fallback(ui_components)
    
    return ui_components

def _detect_and_load_api_key(ui_components: Dict[str, Any]) -> str:
    """Deteksi dan load API key dengan logging detail."""
    print("🔑 Detecting API key from all sources...")
    
    # 1. Environment variable
    api_key = os.environ.get('ROBOFLOW_API_KEY', '')
    if api_key:
        print(f"✅ API key found in environment variable: {'*' * (len(api_key)-4)}{api_key[-4:]}")
        # Update UI field immediately
        if 'api_key' in ui_components and hasattr(ui_components['api_key'], 'value'):
            ui_components['api_key'].value = api_key
        return api_key
    
    # 2. Google Colab userdata
    try:
        from google.colab import userdata
        print("🔍 Checking Google Colab secrets...")
        
        # Primary key
        try:
            api_key = userdata.get('ROBOFLOW_API_KEY')
            if api_key:
                print(f"✅ API key found in Colab secrets (ROBOFLOW_API_KEY): {'*' * (len(api_key)-4)}{api_key[-4:]}")
                # Update UI field immediately
                if 'api_key' in ui_components and hasattr(ui_components['api_key'], 'value'):
                    ui_components['api_key'].value = api_key
                return api_key
        except Exception as e:
            print(f"⚠️ Error accessing ROBOFLOW_API_KEY: {str(e)}")
        
        # Alternative keys
        alternative_keys = ['roboflow_api_key', 'ROBOFLOW_KEY', 'roboflow_key', 'API_KEY']
        for key_name in alternative_keys:
            try:
                api_key = userdata.get(key_name)
                if api_key:
                    print(f"✅ API key found in Colab secrets ({key_name}): {'*' * (len(api_key)-4)}{api_key[-4:]}")
                    # Update UI field immediately
                    if 'api_key' in ui_components and hasattr(ui_components['api_key'], 'value'):
                        ui_components['api_key'].value = api_key
                    return api_key
            except Exception:
                continue
                
        print("❌ No API key found in Google Colab secrets")
        
    except ImportError:
        print("ℹ️ Not in Google Colab environment - skipping secrets check")
    except Exception as e:
        print(f"⚠️ Error accessing Colab secrets: {str(e)}")
    
    print("❌ API key not found from any source")
    return ''

def _update_status_panel_with_env_info(ui_components: Dict[str, Any], env_manager) -> None:
    """Update status panel dengan environment info yang detail."""
    if 'status_panel' not in ui_components:
        return
    
    # Get system info untuk detail lebih lengkap  
    sys_info = env_manager.get_system_info()
    
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
        status_html = f"""
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
        status_html = f"""
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
    """Merge configurations dengan path yang benar."""
    # Start dengan base config
    merged = base_config.copy()
    
    # Override dengan saved config jika ada
    if saved_config:
        merged.update(saved_config)
    
    # Set paths dari environment-aware paths
    merged['output_dir'] = paths['downloads']  # Download ke downloads folder
    merged['backup_dir'] = paths['backup']     # Backup ke backup folder
    
    return merged

def _update_all_ui_components(ui_components: Dict[str, Any], config: Dict[str, Any], paths: Dict[str, str]) -> None:
    """Update semua UI components dari config dengan path yang benar."""
    print("🔄 Updating UI components from config...")
    
    # Field mapping dengan path yang benar
    field_mapping = {
        'workspace': ('workspace', 'smartcash-wo2us'),
        'project': ('project', 'rupiah-emisi-2022'),
        'version': ('version', '3'),
        'output_dir': ('output_dir', paths['downloads']),      # Use downloads path
        'backup_dir': ('backup_dir', paths['backup']),        # Use backup path
        'validate_dataset': ('validate_dataset', True),
        'backup_before_download': ('backup_checkbox', False)
    }
    
    # Update each field
    for config_key, (ui_key, default_value) in field_mapping.items():
        if ui_key in ui_components and hasattr(ui_components[ui_key], 'value'):
            value = config.get(config_key, default_value)
            ui_components[ui_key].value = value
            print(f"   • {ui_key}: {value}")
    
    # API key sudah di-handle di _detect_and_load_api_key
    print("✅ UI components updated")

def _create_smart_defaults(paths: Dict[str, str], api_key: str) -> Dict[str, Any]:
    """Create smart defaults berdasarkan environment paths."""
    return {
        'workspace': 'smartcash-wo2us',
        'project': 'rupiah-emisi-2022',
        'version': '3',
        'api_key': api_key,
        'output_dir': paths['downloads'],    # Downloads folder
        'backup_dir': paths['backup'],       # Backup folder
        'validate_dataset': True,
        'backup_checkbox': False
    }

def _load_saved_config_safe(config_manager) -> Dict[str, Any]:
    """Load saved config dengan error handling."""
    try:
        if hasattr(config_manager, 'get_config'):
            return config_manager.get_config('dataset') or {}
        elif hasattr(config_manager, 'config'):
            return getattr(config_manager, 'config', {}).get('dataset', {})
        else:
            return {}
    except Exception as e:
        print(f"⚠️ Error loading saved config: {str(e)}")
        return {}

def _set_minimal_fallback(ui_components: Dict[str, Any]) -> None:
    """Set minimal fallback jika terjadi error."""
    minimal_values = {
        'workspace': 'smartcash-wo2us',
        'project': 'rupiah-emisi-2022',
        'version': '3',
        'output_dir': 'data/downloads',  # Fallback ke downloads
        'backup_dir': 'data/backup'      # Fallback ke backup
    }
    
    for key, value in minimal_values.items():
        if key in ui_components and hasattr(ui_components[key], 'value'):
            ui_components[key].value = value

def _update_storage_info_widget(ui_components: Dict[str, Any], env_manager) -> None:
    """Update storage info widget dengan path info."""
    if 'drive_info' not in ui_components:
        return
        
    if env_manager.is_colab and env_manager.is_drive_mounted:
        info_html = f"""
        <div style="background: #e8f5e8; border: 1px solid #4caf50; border-radius: 4px; padding: 8px; margin: 5px 0;">
            <span style="color: #2e7d32;">✅ Dataset akan disimpan di Google Drive</span><br>
            <small style="color: #388e3c;">Path: {env_manager.drive_path}/data/</small>
        </div>
        """
    elif env_manager.is_colab:
        info_html = """
        <div style="background: #fff3cd; border: 1px solid #ffc107; border-radius: 4px; padding: 8px; margin: 5px 0;">
            <span style="color: #856404;">⚠️ Drive tidak terhubung - dataset akan disimpan lokal</span><br>
            <small style="color: #856404;">Path: /content/data/ (hilang saat restart)</small>
        </div>
        """
    else:
        info_html = """
        <div style="background: #e3f2fd; border: 1px solid #2196f3; border-radius: 4px; padding: 8px; margin: 5px 0;">
            <span style="color: #1565c0;">ℹ️ Environment lokal - dataset akan disimpan lokal</span><br>
            <small style="color: #1976d2;">Path: ./data/</small>
        </div>
        """
    
    ui_components['drive_info'].value = info_html