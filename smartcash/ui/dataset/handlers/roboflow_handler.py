"""
File: smartcash/ui/dataset/handlers/roboflow_handler.py
Deskripsi: Handler untuk download dataset dari Roboflow
"""

import time
from pathlib import Path
from typing import Dict, Any, Tuple

def download_from_roboflow(ui_components: Dict[str, Any], config: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Download dataset dari Roboflow.
    
    Args:
        ui_components: Dictionary komponen UI
        config: Konfigurasi download dataset
        
    Returns:
        Tuple (success, message)
    """
    logger = ui_components.get('logger')
    
    try:
        # Update progress
        _update_progress(ui_components, 20, "Mempersiapkan koneksi ke Roboflow...")
        
        # Import roboflow
        try:
            import roboflow
        except ImportError:
            _update_progress(ui_components, 10, "Menginstall package roboflow...")
            
            # Install roboflow
            import subprocess
            import sys
            result = subprocess.run([sys.executable, "-m", "pip", "install", "roboflow"], 
                                   capture_output=True, text=True)
            
            if result.returncode != 0:
                return False, f"Gagal menginstall package roboflow: {result.stderr}"
            
            import roboflow
        
        # Get konfigurasi Roboflow
        workspace_id = config.get('workspace')
        project_id = config.get('project')
        version = config.get('version')
        api_key = config.get('api_key')
        output_dir = config.get('output_dir', 'data')
        output_format = config.get('output_format', 'YOLO v5')
        
        # Map format output ke format Roboflow
        format_mapping = {
            'YOLO v5': 'yolov5pytorch',
            'COCO': 'coco',
            'VOC': 'voc'
        }
        
        rf_format = format_mapping.get(output_format, 'yolov5pytorch')
        
        # Update progress
        _update_progress(ui_components, 30, f"Inisialisasi Roboflow API...")
        
        # Initiate Roboflow API
        rf = roboflow.Roboflow(api_key=api_key)
        
        # Get workspace
        _update_progress(ui_components, 40, f"Mengakses workspace {workspace_id}...")
        workspace = rf.workspace(workspace_id)
        
        # Get project
        _update_progress(ui_components, 50, f"Mengakses project {project_id}...")
        project = workspace.project(project_id)
        
        # Get version
        _update_progress(ui_components, 60, f"Mengakses version {version}...")
        # Untuk mendukung format 'v1' dan '1'
        version_number = version[1:] if version.startswith('v') else version
        dataset = project.version(version_number).download(rf_format)
        
        # Pastikan direktori output ada
        output_path = Path(output_dir)
        if not output_path.exists():
            output_path.mkdir(parents=True, exist_ok=True)
        
        # Log success
        _update_progress(ui_components, 100, "Dataset berhasil didownload")
        return True, f"Dataset berhasil didownload ke {output_dir}"
    
    except Exception as e:
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