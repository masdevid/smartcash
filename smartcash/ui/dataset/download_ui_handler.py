"""
File: smartcash/ui/dataset/download_ui_handler.py
Deskripsi: Handler untuk UI komponen download dataset dan event UI related
"""

from typing import Dict, Any, Optional
import ipywidgets as widgets
from IPython.display import display, HTML

from smartcash.ui.utils.constants import COLORS, ICONS

def setup_ui_handlers(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """
    Setup handler untuk komponen UI download dataset.
    
    Args:
        ui_components: Dictionary komponen UI
        env: Environment manager
        config: Konfigurasi aplikasi
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    logger = ui_components.get('logger')
    
    try:
        # Handler untuk API Key input (menampilkan/menyembunyikan)
        api_key_input = ui_components.get('api_key_input')
        api_key_toggle = ui_components.get('api_key_toggle')
        
        if api_key_input and api_key_toggle:
            def toggle_api_key_visibility(change):
                if change['name'] != 'value': 
                    return
                    
                if change['new']:  # Jika toggle diaktifkan
                    api_key_input.layout.display = ''
                else:
                    api_key_input.layout.display = 'none'
            
            # Clear handlers jika ada
            if hasattr(api_key_toggle, '_observe_handlers'):
                for handler in api_key_toggle._observe_handlers:
                    if handler.change_type == 'value':
                        api_key_toggle.unobserve(handler.handler, names='value')
            
            # Setup handler
            api_key_toggle.observe(toggle_api_key_visibility, names='value')
            
            # Set initial state
            if api_key_toggle.value:
                api_key_input.layout.display = ''
            else:
                api_key_input.layout.display = 'none'
                
            if logger:
                logger.debug("🔄 Handler visibilitas API key terdaftar")
                
        # Handler untuk pembuatan API Key jika ada tombol create_key
        create_key_button = ui_components.get('create_key_button')
        if create_key_button:
            def show_api_key_help(b):
                # Tampilkan popup/bantuan untuk membuat API key
                with ui_components.get('status'):
                    display(HTML(f"""
                        <div style="padding:15px; background-color:{COLORS['alert_info_bg']}; 
                                  color:{COLORS['alert_info_text']}; 
                                  border-radius:4px; margin:10px 0;">
                            <h4 style="margin-top:0">{ICONS['info']} Cara Mendapatkan API Key Roboflow</h4>
                            <ol>
                                <li>Buka <a href="https://app.roboflow.com/login" target="_blank">Roboflow</a> dan login ke akun Anda</li>
                                <li>Klik nama Anda di pojok kanan atas dan pilih 'Settings'</li>
                                <li>Scroll ke bawah dan temukan bagian 'API Key'</li>
                                <li>Klik 'Create New API Key' jika belum memilikinya</li>
                                <li>Salin API Key dan paste di field API Key di atas</li>
                            </ol>
                        </div>
                    """))
            
            # Clear handlers jika ada
            if hasattr(create_key_button, '_click_handlers'):
                create_key_button._click_handlers.callbacks = []
                
            # Setup handler
            create_key_button.on_click(show_api_key_help)
            
            if logger:
                logger.debug("🔄 Handler bantuan API key terdaftar")
        
        # Handler untuk form validation (jika ada)
        form_fields = {
            'api_key_input': ui_components.get('api_key_input'),
            'workspace_input': ui_components.get('workspace_input'),
            'project_input': ui_components.get('project_input'),
            'version_input': ui_components.get('version_input')
        }
        
        # Fungsi validasi sederhana untuk mengubah warna tombol download
        def validate_form():
            download_button = ui_components.get('download_button')
            if not download_button:
                return
                
            # Cek apakah semua field terisi
            all_filled = all([
                field and hasattr(field, 'value') and field.value.strip()
                for field in form_fields.values()
                if field is not None
            ])
            
            # Update style tombol
            if all_filled:
                download_button.button_style = 'primary'
                download_button.disabled = False
            else:
                download_button.button_style = ''
                download_button.disabled = True
        
        # Register handler untuk setiap field form
        for field_name, field in form_fields.items():
            if field and hasattr(field, 'observe'):
                # Fungsi observe untuk field ini
                def create_observer(field_name):
                    def field_observer(change):
                        if change['name'] != 'value':
                            return
                        validate_form()
                    return field_observer
                
                # Clear handlers yang ada untuk menghindari duplikasi
                if hasattr(field, '_observe_handlers'):
                    for handler in field._observe_handlers:
                        if handler.change_type == 'value':
                            field.unobserve(handler.handler, names='value')
                
                # Register handler
                field.observe(create_observer(field_name), names='value')
                
        # Inisialisasi validasi form
        validate_form()
        
        if logger:
            logger.debug("🔄 Handler validasi form terdaftar")
        
        # Handler untuk format dropdown jika ada
        format_select = ui_components.get('format_select')
        if format_select:
            # Set nilai default ke yolov5pytorch
            if hasattr(format_select, 'options') and 'yolov5pytorch' in format_select.options:
                format_select.value = 'yolov5pytorch'
            
            if logger:
                logger.debug("🔄 Format download disetel ke yolov5pytorch")
                
    except Exception as e:
        if logger:
            logger.error(f"❌ Error setup UI handlers: {str(e)}")
    
    return ui_components