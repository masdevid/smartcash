"""
File: smartcash/ui/dataset/downloader/utils/backend_utils.py
Deskripsi: Backend integration utilities dengan proper scanner usage
"""

from typing import Dict, Any, Tuple, Optional
from smartcash.ui.utils.logger_bridge import UILoggerBridge

def check_existing_dataset(logger_bridge: Optional[UILoggerBridge] = None) -> Tuple[bool, int, Dict[str, Any]]:
    """
    Check existing dataset menggunakan backend scanner.
    
    Args:
        logger_bridge: LoggerBridge instance untuk logging terpusat
        
    Returns:
        Tuple[bool, int, Dict]: (has_content, total_images, summary_data)
    """
    try:
        if logger_bridge:
            logger_bridge.debug("Memeriksa dataset yang sudah ada...")
            
        from smartcash.dataset.downloader.dataset_scanner import create_dataset_scanner
        
        scanner = create_dataset_scanner(logger_bridge)
        
        # Use quick check untuk determine if content exists
        has_content = scanner.quick_check_existing()
        
        if not has_content:
            if logger_bridge:
                logger_bridge.debug("Tidak ada dataset yang ditemukan")
            return False, 0, {}
        
        if logger_bridge:
            logger_bridge.info("Dataset ditemukan, memindai detail...")
            
        # Get detailed summary jika ada content
        result = scanner.scan_existing_dataset_parallel()
        
        if result.get('status') == 'success':
            summary = result.get('summary', {})
            splits = result.get('splits', {})
            total_images = summary.get('total_images', 0)
            
            if logger_bridge:
                logger_bridge.info(f"Ditemukan {total_images} gambar dalam dataset")
            
            # Extract split breakdown
            split_breakdown = {}
            for split_name, split_data in splits.items():
                if split_data.get('status') == 'success':
                    split_breakdown[split_name] = split_data.get('images', 0)
            
            return True, total_images, {
                'summary': summary,
                'splits': split_breakdown,
                'downloads': result.get('downloads', {}),
                'scan_result': result
            }
        
        if logger_bridge:
            logger_bridge.warning("Pemindaian dataset gagal atau tidak lengkap")
        return False, 0, {}
            
    except Exception as e:
        error_msg = f"Gagal memeriksa dataset yang ada: {str(e)}"
        if logger_bridge:
            logger_bridge.error(error_msg, exc_info=True)
        return False, 0, {}

def get_cleanup_targets(logger_bridge: Optional[UILoggerBridge] = None) -> Dict[str, Any]:
    """
    Get cleanup targets menggunakan backend scanner.
    
    Args:
        logger_bridge: LoggerBridge instance untuk logging terpusat
        
    Returns:
        Dictionary dengan cleanup targets atau error info
    """
    try:
        if logger_bridge:
            logger_bridge.debug("Mendapatkan daftar target cleanup...")
            
        from smartcash.dataset.downloader.dataset_scanner import create_dataset_scanner
        
        scanner = create_dataset_scanner(logger_bridge)
        result = scanner.get_cleanup_targets()
        
        if logger_bridge and result.get('status') == 'success':
            logger_bridge.info("Berhasil mendapatkan daftar target cleanup")
        
        return result
        
    except Exception as e:
        error_msg = f"Gagal mendapatkan cleanup targets: {str(e)}"
        if logger_bridge:
            logger_bridge.error(error_msg, exc_info=True)
        return {
            'status': 'error',
            'message': error_msg,
            'targets': {},
            'summary': {'total_files': 0, 'total_size': 0}
        }

def validate_backend_service_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate config untuk backend service.
    
    Args:
        config: Service configuration
        
    Returns:
        Validation result
    """
    try:
        from smartcash.dataset.downloader import validate_config_quick
        
        if validate_config_quick(config):
            return {'valid': True, 'message': 'Config valid untuk backend service'}
        else:
            return {'valid': False, 'message': 'Config tidak valid untuk backend service'}
            
    except Exception as e:
        return {'valid': False, 'message': f'Error validating config: {str(e)}'}

def create_backend_downloader(ui_config: Dict[str, Any], logger_bridge: Optional[UILoggerBridge] = None):
    """
    Create downloader instance dari UI config.
    
    Args:
        ui_config: UI configuration
        logger_bridge: LoggerBridge instance untuk logging terpusat
        
    Returns:
        Downloader instance atau None
    """
    try:
        if logger_bridge:
            logger_bridge.debug("Membuat instance downloader...")
            
        from smartcash.dataset.downloader.downloader import DatasetDownloader
        
        # Extract config
        roboflow = ui_config.get('data', {}).get('roboflow', {})
        download = ui_config.get('download', {})
        
        # Validate required fields
        required = ['workspace', 'project', 'version', 'api_key']
        missing = [field for field in required if not roboflow.get(field)]
        if missing:
            error_msg = f"Field yang diperlukan tidak lengkap: {', '.join(missing)}"
            if logger_bridge:
                logger_bridge.error(error_msg)
            return None
            
        if logger_bridge:
            logger_bridge.info("Menginisialisasi DatasetDownloader...")
            
        return DatasetDownloader(
            workspace=roboflow['workspace'],
            project=roboflow['project'],
            version=roboflow['version'],
            api_key=roboflow['api_key'],
            output_format=roboflow.get('output_format', 'yolov5pytorch'),
            rename_files=download.get('rename_files', True),
            validate_download=download.get('validate_download', True),
            backup_existing=download.get('backup_existing', False),
            logger=logger_bridge  # Pass the logger_bridge directly
        )
        
    except Exception as e:
        error_msg = f"Gagal membuat downloader: {str(e)}"
        if logger_bridge:
            logger_bridge.error(error_msg, exc_info=True)
        return None

def create_backend_cleanup_service(logger_bridge: Optional[UILoggerBridge] = None):
    """
    Create cleanup service instance.
    
    Args:
        logger_bridge: LoggerBridge instance untuk logging terpusat
        
    Returns:
        Cleanup service instance atau None
    """
    try:
        if logger_bridge:
            logger_bridge.debug("Membuat instance cleanup service...")
            
        from smartcash.dataset.downloader.cleanup_service import DatasetCleanupService
        return DatasetCleanupService(logger=logger_bridge)  # Pass the logger_bridge directly
        
    except Exception as e:
        error_msg = f"Gagal membuat cleanup service: {str(e)}"
        if logger_bridge:
            logger_bridge.error(error_msg, exc_info=True)
        return None