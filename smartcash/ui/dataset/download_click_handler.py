"""
File: smartcash/ui/dataset/download_click_handler.py
Deskripsi: Handler untuk event click tombol download dan upload dataset
"""

from typing import Dict, Any, Optional
from IPython.display import display, HTML, clear_output

from smartcash.ui.utils.constants import COLORS, ICONS

def setup_click_handlers(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """
    Setup handlers untuk tombol download dataset.
    
    Args:
        ui_components: Dictionary komponen UI
        env: Environment manager
        config: Konfigurasi aplikasi
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    logger = ui_components.get('logger')
    
    # Dapatkan tombol-tombol dari UI components
    download_button = ui_components.get('download_button')
    upload_button = ui_components.get('upload_button')
    
    # Handler untuk tombol download Roboflow
    def on_download_click(b):
        # Dapatkan semua parameter dari UI
        api_key = ui_components.get('api_key_input').value
        workspace = ui_components.get('workspace_input').value
        project = ui_components.get('project_input').value
        version = ui_components.get('version_input').value
        format_select = ui_components.get('format_select')
        
        # Validasi input
        status_widget = ui_components.get('status')
        if not all([api_key, workspace, project, version]):
            if status_widget:
                with status_widget:
                    clear_output(wait=True)
                    display(HTML(f"""
                        <div style="padding:10px; background-color:{COLORS['alert_warning_bg']}; 
                                  color:{COLORS['alert_warning_text']}; 
                                  border-radius:4px; margin:5px 0;">
                            <p style="margin:5px 0">{ICONS['warning']} Mohon lengkapi semua field Roboflow API Key, Workspace, Project, dan Version!</p>
                        </div>
                    """))
            return
        
        # Import fungsi download dari handler yang sudah dibuat
        try:
            from smartcash.ui.dataset.roboflow_download_handler import download_from_roboflow
            
            # Dapatkan format yang dipilih atau default ke yolov5pytorch
            download_format = format_select.value if format_select else "yolov5pytorch"
            
            # Jalankan fungsi download
            result = download_from_roboflow(
                ui_components,
                api_key=api_key,
                workspace=workspace,
                project=project,
                version=int(version),
                format=download_format
            )
            
            # Log hasil jika berhasil
            if logger:
                logger.success(f"✅ Dataset berhasil didownload dari Roboflow: {project} (v{version})")
                
            # Jika ada callback onDownloadSuccess, panggil
            if 'onDownloadSuccess' in ui_components and callable(ui_components['onDownloadSuccess']):
                ui_components['onDownloadSuccess'](result)
                
        except Exception as e:
            # Error sudah ditangani di download_from_roboflow, tambahkan logging saja
            if logger:
                logger.error(f"❌ Gagal download dataset: {str(e)}")
            
            # Jika ada callback onDownloadError, panggil
            if 'onDownloadError' in ui_components and callable(ui_components['onDownloadError']):
                ui_components['onDownloadError'](str(e))
    
    # Handler untuk tombol upload lokal
    def on_upload_click(b):
        # Dapatkan file upload widget
        file_upload = ui_components.get('file_upload')
        
        if not file_upload or not file_upload.value:
            status_widget = ui_components.get('status')
            if status_widget:
                with status_widget:
                    clear_output(wait=True)
                    display(HTML(f"""
                        <div style="padding:10px; background-color:{COLORS['alert_warning_bg']}; 
                                  color:{COLORS['alert_warning_text']}; 
                                  border-radius:4px; margin:5px 0;">
                            <p style="margin:5px 0">{ICONS['warning']} Mohon pilih file dataset terlebih dahulu!</p>
                        </div>
                    """))
            return
            
        try:
            # Import fungsi process_local_upload dari handler yang sudah dibuat
            from smartcash.ui.dataset.local_upload_handler import process_local_upload
            
            # Proses upload file
            result = process_local_upload(ui_components, env)
            
            # Log ke logger jika tersedia
            if logger:
                logger.success(f"✅ File dataset berhasil diupload")
                
            # Jika ada callback onUploadSuccess, panggil
            if 'onUploadSuccess' in ui_components and callable(ui_components['onUploadSuccess']):
                ui_components['onUploadSuccess'](result)
                
        except Exception as e:
            # Tampilkan error jika belum ditangani
            status_widget = ui_components.get('status')
            if status_widget:
                with status_widget:
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
                
            # Jika ada callback onUploadError, panggil
            if 'onUploadError' in ui_components and callable(ui_components['onUploadError']):
                ui_components['onUploadError'](str(e))
    
    # Register handlers ke tombol-tombol
    if download_button:
        # Clear handlers yang ada untuk menghindari duplikasi
        if hasattr(download_button, '_click_handlers'):
            download_button._click_handlers.callbacks = []
        download_button.on_click(on_download_click)
        if logger:
            logger.debug("🔄 Handler tombol download terdaftar")
            
    if upload_button:
        # Clear handlers yang ada untuk menghindari duplikasi
        if hasattr(upload_button, '_click_handlers'):
            upload_button._click_handlers.callbacks = []
        upload_button.on_click(on_upload_click)
        if logger:
            logger.debug("🔄 Handler tombol upload terdaftar")
    
    # Setup handler untuk perubahan tipe sumber dataset
    source_select = ui_components.get('source_select')
    if source_select:
        def on_source_change(change):
            if not change or change['name'] != 'value':
                return
                
            source_type = change['new']
            roboflow_section = ui_components.get('roboflow_section')
            local_section = ui_components.get('local_section')
            
            # Toggle visibility roboflow vs local upload
            if source_type == 'roboflow':
                if roboflow_section:
                    roboflow_section.layout.display = 'block'
                if local_section:
                    local_section.layout.display = 'none'
            else:
                if roboflow_section:
                    roboflow_section.layout.display = 'none'
                if local_section:
                    local_section.layout.display = 'block'
        
        # Clear handlers yang ada untuk menghindari duplikasi
        if hasattr(source_select, '_observe_handlers'):
            for handler in source_select._observe_handlers:
                if handler.change_type == 'value':
                    source_select.unobserve(handler.handler, names='value')
        
        # Register handler
        source_select.observe(on_source_change, names='value')
        
        # Initialize visibility
        on_source_change({'name': 'value', 'new': source_select.value})
        
        if logger:
            logger.debug("🔄 Handler selector sumber dataset terdaftar")
    
    # Tambahkan function cleanup untuk penghapusan handler saat dibutuhkan
    def cleanup():
        # Unregister handlers
        if download_button:
            if hasattr(download_button, '_click_handlers'):
                download_button._click_handlers.callbacks = []
                
        if upload_button:
            if hasattr(upload_button, '_click_handlers'):
                upload_button._click_handlers.callbacks = []
                
        if source_select:
            if hasattr(source_select, '_observe_handlers'):
                for handler in source_select._observe_handlers:
                    if handler.change_type == 'value':
                        source_select.unobserve(handler.handler, names='value')
                
        if logger:
            logger.info("🧹 Download click handlers dibersihkan")
    
    ui_components['cleanup_handlers'] = cleanup
    
    return ui_components