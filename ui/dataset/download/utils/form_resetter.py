
"""
File: smartcash/ui/dataset/download/utils/form_resetter.py
Deskripsi: Reset form fields ke nilai default
"""

def reset_form_fields(ui_components: Dict[str, Any]) -> None:
    """Reset semua field form ke nilai default."""
    
    # Default values
    defaults = {
        'workspace': '',
        'project': '',
        'version': '',
        'output_dir': 'data',
        'backup_dir': 'data/backup',
        'validate_dataset': True,
        'backup_checkbox': False
    }
    
    # Reset text fields
    for field, default_value in defaults.items():
        if field in ui_components and hasattr(ui_components[field], 'value'):
            ui_components[field].value = default_value
    
    # API key hanya reset jika ada checkbox reset_api yang dicentang
    if ('reset_api_checkbox' in ui_components and 
        ui_components['reset_api_checkbox'].value and 
        'api_key' in ui_components):
        ui_components['api_key'].value = ''