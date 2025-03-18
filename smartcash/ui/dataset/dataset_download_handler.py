"""
File: smartcash/ui/dataset/dataset_download_handler.py
Deskripsi: Handler utama untuk setup dan koordinasi download dataset dengan integrasi dataset manager
"""

from IPython.display import display, HTML, clear_output
from typing import Dict, Any, Optional

from smartcash.ui.utils.constants import COLORS, ICONS
from smartcash.ui.utils.ui_helpers import create_info_alert

def setup_dataset_download_handlers(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """
    Setup handler untuk komponen UI dataset download.
    
    Args:
        ui_components: Dictionary berisi widget UI
        env: Environment manager
        config: Konfigurasi aplikasi
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    try:
        # Import handler
        from smartcash.ui.dataset.roboflow_download_handler import download_from_roboflow
        from smartcash.ui.dataset.local_upload_handler import process_local_upload
        
        # Dapatkan logger jika tersedia
        logger = ui_components.get('logger')
        
        # Dapatkan komponen UI
        download_button = ui_components.get('download_button')
        upload_button = ui_components.get('upload_button')
        
        # Buat status panel jika belum ada
        if 'status_panel' not in ui_components:
            from smartcash.ui.components.alerts import create_info_box
            ui_components['status_panel'] = create_info_box(
                "Status Download",
                "Siap untuk download dataset...",
                "info"
            )
        
        # Handler untuk Roboflow download
        def on_download_click(b):
            # Dapatkan semua parameter
            api_key = ui_components.get('api_key_input').value
            workspace = ui_components.get('workspace_input').value
            project = ui_components.get('project_input').value
            version = ui_components.get('version_input').value
            format_select = ui_components.get('format_select')
            
            # Validasi input
            status_widget = ui_components.get('status')
            if not all([api_key, workspace, project, version]):
                with status_widget:
                    clear_output(wait=True)
                    display(create_info_alert(
                        "Mohon lengkapi semua field Roboflow API Key, Workspace, Project, dan Version!",
                        "warning"
                    ))
                return
            
            # Gunakan dataset_manager melalui download_from_roboflow function
            try:
                download_format = format_select.value if format_select else "yolov5"
                download_from_roboflow(
                    ui_components,
                    api_key=api_key,
                    workspace=workspace,
                    project=project,
                    version=int(version),
                    format=download_format
                )
                
                # Log ke logger jika tersedia
                if logger:
                    logger.success(f"✅ Dataset berhasil didownload dari Roboflow: {project} (v{version})")
                    
            except Exception as e:
                # Error sudah ditangani di download_from_roboflow, tambahkan logging saja
                if logger:
                    logger.error(f"❌ Gagal download dataset: {str(e)}")
        
        # Handler untuk local upload
        def on_upload_click(b):
            # Dapatkan file upload widget
            file_upload = ui_components.get('file_upload')
            
            if not file_upload or not file_upload.value:
                with ui_components.get('status'):
                    clear_output(wait=True)
                    display(create_info_alert(
                        "Mohon pilih file dataset terlebih dahulu!",
                        "warning"
                    ))
                return
                
            try:
                # Proses upload file
                process_local_upload(ui_components, env)
                
                # Log ke logger jika tersedia
                if logger:
                    logger.success(f"✅ File dataset berhasil diupload")
                    
            except Exception as e:
                # Tampilkan error
                with ui_components.get('status'):
                    clear_output(wait=True)
                    display(HTML(f"""
                        <div style="padding:10px; background-color:{COLORS['alert_danger_bg']}; 
                                  color:{COLORS['alert_danger_text']}; 
                                  border-radius:4px; margin:5px 0;">
                            <p style="margin:5px 0">{ICONS['error']} Error saat upload file: {str(e)}</p>
                        </div>
                    """))
                
                # Log ke logger jika tersedia
                if logger:
                    logger.error(f"❌ Gagal upload file dataset: {str(e)}")
        
        # Check source type change
        def on_source_change(change):
            if not change or change['name'] != 'value':
                return
                
            source_type = change['new']
            
            # Toggle visibility roboflow vs local upload
            if source_type == 'roboflow':
                ui_components.get('roboflow_section').layout.display = 'block'
                ui_components.get('local_section').layout.display = 'none'
            else:
                ui_components.get('roboflow_section').layout.display = 'none'
                ui_components.get('local_section').layout.display = 'block'
        
        # Register handlers
        if download_button:
            download_button.on_click(on_download_click)
            
        if upload_button:
            upload_button.on_click(on_upload_click)
            
        # Register source change handler
        source_select = ui_components.get('source_select')
        if source_select:
            source_select.observe(on_source_change, names='value')
            
            # Initialize visibility
            on_source_change({'name': 'value', 'new': source_select.value})
        
        # Tambahkan config ke UI components
        ui_components['config'] = config
        
        # Tambahkan cleanup function
        def cleanup():
            # Unregister handlers
            if download_button:
                download_button._click_handlers.callbacks = []
                
            if upload_button:
                upload_button._click_handlers.callbacks = []
                
            if source_select:
                source_select.unobserve(on_source_change, names='value')
                
            if logger:
                logger.info("🧹 Dataset download handlers dibersihkan")
                
        ui_components['cleanup'] = cleanup
        
    except Exception as e:
        # Fallback sederhana jika terjadi error
        if 'status' in ui_components:
            with ui_components['status']:
                display(HTML(f"""
                    <div style="padding:10px; background-color:{COLORS['alert_danger_bg']}; 
                              color:{COLORS['alert_danger_text']}; 
                              border-radius:4px; margin:5px 0;">
                        <p style="margin:5px 0">{ICONS['error']} Error setup dataset download handler: {str(e)}</p>
                    </div>
                """))
    
    return ui_components