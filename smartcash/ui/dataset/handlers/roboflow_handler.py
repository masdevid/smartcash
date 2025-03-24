"""
File: smartcash/ui/dataset/handlers/roboflow_handler.py
Deskripsi: Handler untuk delegasi download dataset dari Roboflow ke service
"""

import time
from pathlib import Path
from typing import Dict, Any, Tuple

# CATATAN: File ini menjadi adapter/delegator ke service yang sudah ada
# Implementasi langsung sudah digantikan oleh download_handler.py

def download_from_roboflow(ui_components: Dict[str, Any], config: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Delegasi download dataset dari Roboflow ke DownloadService.
    
    Args:
        ui_components: Dictionary komponen UI
        config: Konfigurasi download dataset
        
    Returns:
        Tuple (success, message)
    """
    logger = ui_components.get('logger')
    
    try:
        # Import service download
        from smartcash.dataset.services.downloader.download_service import DownloadService
        
        # Get konfigurasi Roboflow
        api_key = config.get('api_key')
        workspace = config.get('workspace')
        project = config.get('project')
        version = config.get('version')
        output_dir = config.get('output_dir', 'data')
        backup_existing = config.get('backup_existing', False)
        
        # Update progress
        _update_progress(ui_components, 30, f"Menggunakan DownloadService untuk {workspace}/{project}:{version}...")
        
        # Setup service
        service_config = {'data': {'roboflow': {
            'api_key': api_key,
            'workspace': workspace,
            'project': project,
            'version': version
        }}}
        
        download_service = DownloadService(output_dir=output_dir, config=service_config, logger=logger)
        
        # Jalankan download
        result = download_service.download_from_roboflow(
            api_key=api_key,
            workspace=workspace,
            project=project,
            version=version,
            output_dir=output_dir,
            show_progress=True,
            backup_existing=backup_existing
        )
        
        # Format pesan sukses
        success_msg = f"Dataset berhasil didownload dari Roboflow: {result.get('stats', {}).get('total_images', 0)} gambar"
        return True, success_msg
    
    except Exception as e:
        # Log dan return error
        if logger: logger.error(f"❌ Error saat download dari Roboflow: {str(e)}")
        return False, f"Error saat download dari Roboflow: {str(e)}"

def _update_progress(ui_components: Dict[str, Any], value: int, message: str) -> None:
    """
    Update progress bar dan progress tracker.
    
    Args:
        ui_components: Dictionary komponen UI
        value: Nilai progress (0-100)
        message: Pesan progress
    """
    # Update progress bar
    progress_bar = ui_components.get('progress_bar')
    progress_message = ui_components.get('progress_message')
    
    if progress_bar:
        progress_bar.value = value
    
    if progress_message:
        progress_message.value = message
    
    # Update progress tracker
    tracker_key = 'dataset_downloader_tracker'
    if tracker_key in ui_components:
        tracker = ui_components[tracker_key]
        tracker.update(value, message)
        
    # Log ke logger
    logger = ui_components.get('logger')
    if logger and value % 20 == 0:  # Log setiap 20%
        logger.info(f"🔄 {message} ({value}%)")