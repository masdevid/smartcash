"""
File: smartcash/ui/dataset/download/handlers/reset_action.py
Deskripsi: Fixed reset action yang mengembalikan ke nilai default, bukan kosong
"""

from typing import Dict, Any

def execute_reset_action(ui_components: Dict[str, Any], button: Any = None) -> None:
    """Eksekusi reset form ke nilai default, bukan kosong."""
    logger = ui_components.get('logger')
    if logger:
        logger.info("🔄 Mereset form ke nilai default")
    
    try:
        # Reset form fields ke defaults
        _reset_form_to_defaults(ui_components)
        
        # Reset progress
        _reset_progress(ui_components)
        
        # Clear confirmation area
        if 'confirmation_area' in ui_components and hasattr(ui_components['confirmation_area'], 'clear_output'):
            ui_components['confirmation_area'].clear_output()
        
        if logger:
            logger.success("✅ Form berhasil direset ke nilai default")
            
    except Exception as e:
        if logger:
            logger.error(f"❌ Error saat reset: {str(e)}")

def _reset_form_to_defaults(ui_components: Dict[str, Any]) -> None:
    """Reset semua field form ke nilai default yang tersimpan."""
    # Ambil defaults yang sudah disiapkan di config_handlers
    defaults = ui_components.get('_defaults', {})
    
    # Jika tidak ada defaults, buat yang baru
    if not defaults:
        defaults = _get_fallback_defaults(ui_components)
    
    # Mapping field ke component
    field_mapping = {
        'workspace': 'workspace',
        'project': 'project', 
        'version': 'version',
        'api_key': 'api_key',
        'output_dir': 'output_dir',
        'backup_dir': 'backup_dir',
        'validate_dataset': 'validate_dataset',
        'backup_checkbox': 'backup_checkbox'
    }
    
    # Reset setiap field ke default value
    for default_key, ui_key in field_mapping.items():
        if ui_key in ui_components and hasattr(ui_components[ui_key], 'value'):
            default_value = defaults.get(default_key, '')
            ui_components[ui_key].value = default_value

def _get_fallback_defaults(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Get fallback defaults jika tidak ada defaults tersimpan."""
    env_manager = ui_components.get('env_manager')
    
    # Deteksi API key
    api_key = _detect_api_key()
    
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

def _detect_api_key() -> str:
    """Deteksi API key dari environment atau Colab secrets."""
    import os
    
    # Environment variable
    api_key = os.environ.get('ROBOFLOW_API_KEY', '')
    if api_key:
        return api_key
    
    # Google Colab secrets
    try:
        from google.colab import userdata
        return userdata.get('ROBOFLOW_API_KEY', '')
    except Exception:
        return ''

def _reset_progress(ui_components: Dict[str, Any]) -> None:
    """Reset semua progress indicator."""
    for widget_key in ['progress_bar', 'current_progress']:
        if widget_key in ui_components:
            ui_components[widget_key].value = 0
            ui_components[widget_key].description = "Progress: 0%"
            if hasattr(ui_components[widget_key], 'layout'):
                ui_components[widget_key].layout.visibility = 'hidden'
    
    for label_key in ['overall_label', 'step_label']:
        if label_key in ui_components:
            ui_components[label_key].value = ""
            if hasattr(ui_components[label_key], 'layout'):
                ui_components[label_key].layout.visibility = 'hidden'
    
    if 'progress_container' in ui_components:
        ui_components['progress_container'].layout.display = 'none'