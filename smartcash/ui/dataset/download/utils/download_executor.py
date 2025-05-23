"""
File: smartcash/ui/dataset/download/utils/download_executor.py
Deskripsi: Download executor yang diperbaiki dengan progress tracking dan error handling yang tepat
"""

import time
from typing import Dict, Any
from smartcash.ui.dataset.download.services.ui_download_service_final import UIDownloadServiceFinal

def execute_roboflow_download(ui_components: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """Eksekusi download dengan progress tracking yang terintegrasi dan error handling yang kuat."""
    
    logger = ui_components.get('logger')
    start_time = time.time()
    
    try:
        # 🚀 Initialize download service dengan progress integration
        if logger:
            logger.info("🔧 Menginisialisasi download service...")
        
        download_service = UIDownloadServiceFinal(ui_components)
        
        # 📊 Setup progress tracking
        _setup_progress_tracking(ui_components)
        
        # 🎯 Execute download dengan progress callback
        if logger:
            logger.info("⬇️ Memulai download dataset...")
        
        result = download_service.download_dataset(params)
        
        # 📈 Log hasil download
        duration = time.time() - start_time
        if result.get('status') == 'success':
            _handle_download_success(ui_components, result, duration)
        else:
            _handle_download_error(ui_components, result, duration)
        
        return result
        
    except Exception as e:
        duration = time.time() - start_time
        error_result = {
            'status': 'error', 
            'message': str(e),
            'duration': duration
        }
        
        _handle_download_error(ui_components, error_result, duration)
        return error_result

def _setup_progress_tracking(ui_components: Dict[str, Any]) -> None:
    """Setup progress tracking UI components."""
    # Show progress container
    if 'progress_container' in ui_components:
        ui_components['progress_container'].layout.display = 'block'
        ui_components['progress_container'].layout.visibility = 'visible'
    
    # Reset progress bars
    progress_widgets = ['progress_bar', 'current_progress']
    for widget_key in progress_widgets:
        if widget_key in ui_components:
            ui_components[widget_key].value = 0
            ui_components[widget_key].description = "Progress: 0%"
            if hasattr(ui_components[widget_key], 'layout'):
                ui_components[widget_key].layout.visibility = 'visible'

def _handle_download_success(ui_components: Dict[str, Any], result: Dict[str, Any], duration: float) -> None:
    """Handle successful download dengan logging dan UI update."""
    logger = ui_components.get('logger')
    
    if logger:
        stats = result.get('stats', {})
        output_dir = result.get('output_dir', 'Unknown')
        total_images = stats.get('total_images', 0)
        drive_storage = result.get('drive_storage', False)
        storage_type = "Google Drive" if drive_storage else "Local Storage"
        
        logger.success(f"🎉 Download berhasil dalam {duration:.1f} detik!")
        logger.info(f"📁 Lokasi: {storage_type}")
        logger.info(f"📂 Path: {output_dir}")
        logger.info(f"🖼️ Total gambar: {total_images}")
        
        # Log breakdown per split jika ada
        for split in ['train', 'valid', 'test']:
            split_key = f'{split}_images'
            if split_key in stats:
                count = stats[split_key]
                if count > 0:
                    logger.info(f"   • {split}: {count} gambar")
    
    # Update progress ke 100%
    _complete_progress(ui_components, "Download selesai")

def _handle_download_error(ui_components: Dict[str, Any], result: Dict[str, Any], duration: float) -> None:
    """Handle download error dengan logging dan UI update."""
    logger = ui_components.get('logger')
    
    if logger:
        error_msg = result.get('message', 'Unknown error')
        logger.error(f"💥 Download gagal setelah {duration:.1f} detik")
        logger.error(f"❌ Error: {error_msg}")
        
        # Berikan saran troubleshooting
        if 'api_key' in error_msg.lower():
            logger.warning("💡 Periksa API key Roboflow Anda")
        elif 'network' in error_msg.lower() or 'connection' in error_msg.lower():
            logger.warning("💡 Periksa koneksi internet Anda")
        elif 'permission' in error_msg.lower():
            logger.warning("💡 Periksa permission direktori output")
    
    # Update progress dengan error state
    _error_progress(ui_components, f"Error: {result.get('message', 'Unknown')}")

def _complete_progress(ui_components: Dict[str, Any], message: str) -> None:
    """Complete progress tracking dengan success state."""
    # Update progress bars ke 100%
    progress_widgets = ['progress_bar', 'current_progress']
    for widget_key in progress_widgets:
        if widget_key in ui_components:
            ui_components[widget_key].value = 100
            ui_components[widget_key].description = "Progress: 100%"
    
    # Update labels
    label_widgets = ['overall_label', 'step_label']
    for label_key in label_widgets:
        if label_key in ui_components:
            ui_components[label_key].value = message

def _error_progress(ui_components: Dict[str, Any], message: str) -> None:
    """Set progress tracking ke error state."""
    # Reset progress bars
    progress_widgets = ['progress_bar', 'current_progress']
    for widget_key in progress_widgets:
        if widget_key in ui_components:
            ui_components[widget_key].value = 0
            ui_components[widget_key].description = "Error"
    
    # Update labels dengan error message
    label_widgets = ['overall_label', 'step_label']
    for label_key in label_widgets:
        if label_key in ui_components:
            ui_components[label_key].value = f"❌ {message}"

def test_download_connection(ui_components: Dict[str, Any]) -> bool:
    """Test koneksi download service untuk debugging."""
    logger = ui_components.get('logger')
    
    try:
        # Test service initialization
        test_service = UIDownloadServiceFinal(ui_components)
        
        if logger:
            logger.info("✅ Download service berhasil diinisialisasi")
        
        # Test progress callback
        if hasattr(test_service, '_progress_callback'):
            if logger:
                logger.info("✅ Progress callback tersedia")
        else:
            if logger:
                logger.warning("⚠️ Progress callback tidak tersedia")
        
        return True
        
    except Exception as e:
        if logger:
            logger.error(f"❌ Test download connection gagal: {str(e)}")
        return False