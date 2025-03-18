"""
File: smartcash/ui/dataset/roboflow_download_handler.py
Deskripsi: Handler untuk download dataset dari Roboflow menggunakan DatasetManager
"""

from typing import Dict, Any, Optional
from IPython.display import display, HTML, clear_output

from smartcash.ui.utils.constants import COLORS, ICONS
from smartcash.common.exceptions import DatasetError

def download_from_roboflow(
    ui_components: Dict[str, Any],
    api_key: str,
    workspace: str,
    project: str,
    version: int,
    format: str = "yolov5pytorch",
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Download dataset dari Roboflow menggunakan DatasetManager.
    
    Args:
        ui_components: Dictionary berisi widget UI
        api_key: API key Roboflow
        workspace: Nama workspace Roboflow
        project: Nama project Roboflow
        version: Versi dataset
        format: Format download ("yolov5", "coco", dll)
        output_dir: Direktori output opsional
        
    Returns:
        Dictionary berisi informasi hasil download
        
    Raises:
        DatasetError: Jika terjadi error saat download
    """
    status_widget = ui_components.get('status')
    
    try:
        # Tampilkan status loading
        if status_widget:
            with status_widget:
                clear_output(wait=True)
                display(HTML(f"""
                    <div style="padding:10px; background-color:{COLORS['alert_info_bg']}; 
                              color:{COLORS['alert_info_text']}; 
                              border-radius:4px; margin:5px 0;">
                        <p style="margin:5px 0">{ICONS['download']} Memulai download dataset dari Roboflow...</p>
                    </div>
                """))
        
        # Dapatkan dataset_manager
        from smartcash.dataset.manager import DatasetManager
        
        # Coba dapatkan config dari UI components atau gunakan default
        config = ui_components.get('config', {})
        
        # Buat instance DatasetManager
        dataset_manager = DatasetManager(config=config)
        
        # Download menggunakan dataset_manager
        result = dataset_manager.download_from_roboflow(
            api_key=api_key,
            workspace=workspace,
            project=project,
            version=version,
            format=format,
            output_dir=output_dir
        )
        
        # Tampilkan hasil sukses
        if status_widget:
            with status_widget:
                clear_output(wait=True)
                display(HTML(f"""
                    <div style="padding:10px; background-color:{COLORS['alert_success_bg']}; 
                              color:{COLORS['alert_success_text']}; 
                              border-radius:4px; margin:5px 0;">
                        <p style="margin:5px 0">{ICONS['success']} Dataset berhasil didownload!</p>
                        <p>Project: {project} (v{version}) dari workspace {workspace}</p>
                        <p>Format: {format}</p>
                    </div>
                """))
        
        return result
        
    except DatasetError as e:
        # Dataset manager sudah menangani banyak exceptions dengan DatasetError
        if status_widget:
            with status_widget:
                clear_output(wait=True)
                display(HTML(f"""
                    <div style="padding:10px; background-color:{COLORS['alert_danger_bg']}; 
                              color:{COLORS['alert_danger_text']}; 
                              border-radius:4px; margin:5px 0;">
                        <p style="margin:5px 0">{ICONS['error']} {str(e)}</p>
                    </div>
                """))
        raise
        
    except Exception as e:
        # Tangani exception lain
        error_message = f"Error saat download dataset: {str(e)}"
        if status_widget:
            with status_widget:
                clear_output(wait=True)
                display(HTML(f"""
                    <div style="padding:10px; background-color:{COLORS['alert_danger_bg']}; 
                              color:{COLORS['alert_danger_text']}; 
                              border-radius:4px; margin:5px 0;">
                        <p style="margin:5px 0">{ICONS['error']} {error_message}</p>
                    </div>
                """))
        raise DatasetError(error_message)