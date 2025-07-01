"""
File: smartcash/ui/dataset/downloader/utils/backend_utils.py
Deskripsi: Backend integration utilities dengan proper scanner usage
"""

from typing import Dict, Any, Tuple, Optional
from smartcash.ui.utils.ui_logger import UILogger
from smartcash.ui.handlers.error_handler import create_error_response
from smartcash.dataset.downloader.dataset_scanner import create_dataset_scanner
from smartcash.dataset.downloader.download_service import create_download_service
from smartcash.dataset.downloader.cleanup_service import create_cleanup_service

def check_existing_dataset(logger: Optional[UILogger] = None) -> Tuple[bool, int, Dict[str, Any]]:
    """
    Check existing dataset menggunakan backend scanner.
    
    Args:
        logger: UILogger instance untuk logging terpusat
        
    Returns:
        Tuple[bool, int, Dict]: (has_content, total_images, summary_data)
    """
    try:
        if logger:
            logger.debug("Memeriksa dataset yang sudah ada...")
            
        scanner = create_dataset_scanner(logger)
        
        # Use quick check untuk determine if content exists
        has_content = scanner.quick_check_existing()
        
        if not has_content:
            if logger:
                logger.debug("Tidak ada dataset yang ditemukan")
            return False, 0, {}
        
        if logger:
            logger.info("Dataset ditemukan, memindai detail...")
            
        # Get detailed summary jika ada content
        result = scanner.scan_existing_dataset_parallel()
        
        if result.get('status') is True or result.get('status') == 'success':
            summary = result.get('summary', {})
            splits = result.get('splits', {})
            total_images = summary.get('total_images', 0)
            
            if logger:
                logger.info(f"Ditemukan {total_images} gambar dalam dataset")
            
            # Extract split breakdown
            split_breakdown = {}
            for split_name, split_data in splits.items():
                if split_data.get('status') is True or split_data.get('status') == 'success':
                    split_breakdown[split_name] = split_data.get('images', 0)
            
            return True, total_images, {
                'summary': summary,
                'splits': split_breakdown,
                'downloads': result.get('downloads', {}),
                'scan_result': result
            }
        
        if logger:
            logger.warning("Pemindaian dataset gagal atau tidak lengkap")
        return False, 0, {}
            
    except Exception as e:
        error_msg = f"Gagal memeriksa dataset yang ada: {str(e)}"
        if logger:
            logger.error(error_msg, exc_info=True)
        error_response = create_error_response(error_msg, e, "Dataset Check Error")
        return False, 0, {
            'status': False,
            'message': error_msg,
            'error_details': error_response.get('error_details', {})
        }

def get_cleanup_targets(logger: Optional[UILogger] = None) -> Dict[str, Any]:
    """
    Get cleanup targets menggunakan backend scanner.
    
    Args:
        logger: UILogger instance untuk logging terpusat
        
    Returns:
        Dictionary dengan cleanup targets atau error info
    """
    try:
        if logger:
            logger.debug("Mendapatkan daftar target cleanup...")
            
        scanner = create_dataset_scanner(logger)
        result = scanner.get_cleanup_targets()
        
        if logger and (result.get('status') is True or result.get('status') == 'success'):
            logger.info("Berhasil mendapatkan daftar target cleanup")
        elif logger:
            logger.warning(f"Gagal mendapatkan daftar target cleanup: {result.get('message', 'Unknown error')}")
        
        return result
        
    except Exception as e:
        error_msg = f"Gagal mendapatkan cleanup targets: {str(e)}"
        if logger:
            logger.error(error_msg, exc_info=True)
        error_response = create_error_response(error_msg, e, "Cleanup Error")
        return {
            'status': False,
            'message': error_msg,
            'targets': {},
            'summary': {'total_files': 0, 'total_size': 0},
            'error_details': error_response.get('error_details', {})
        }

def validate_backend_service_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate config untuk backend service.
    
    Args:
        config: Service configuration
        
    Returns:
        Validation result with 'status' key for consistent API response
    """
    try:
        # Validate required fields
        required_fields = {
            'data.roboflow.workspace': 'Workspace',
            'data.roboflow.project': 'Project',
            'data.roboflow.version': 'Version',
            'data.roboflow.api_key': 'API Key',
        }
        
        missing = []
        for path, label in required_fields.items():
            parts = path.split('.')
            value = config
            for part in parts:
                value = value.get(part, {})
                if not value and not isinstance(value, (int, float, bool)):
                    missing.append(label)
                    break
        
        if missing:
            return {
                'status': False, 
                'message': f"Missing required fields: {', '.join(missing)}"
            }
        else:
            return {'status': True, 'message': 'Config valid untuk backend service'}
            
    except Exception as e:
        error_msg = f'Error validating config: {str(e)}'
        error_response = create_error_response(error_msg, e, "Config Validation Error")
        return {
            'status': False, 
            'message': error_msg,
            'error_details': error_response.get('error_details', {})
        }

def create_backend_downloader(ui_config: Dict[str, Any], logger: Optional[UILogger] = None) -> Dict[str, Any]:
    """
    Create downloader instance dari UI config.
    
    Args:
        ui_config: UI configuration
        logger: UILogger instance untuk logging terpusat
        
    Returns:
        Dictionary with 'status' key and 'downloader' instance if successful
    """
    try:
        if logger:
            logger.debug("Membuat instance downloader...")
        
        # Extract config
        roboflow = ui_config.get('data', {}).get('roboflow', {})
        download = ui_config.get('download', {})
        
        # Validate required fields
        required = ['workspace', 'project', 'version', 'api_key']
        missing = [field for field in required if not roboflow.get(field)]
        if missing:
            error_msg = f"Field yang diperlukan tidak lengkap: {', '.join(missing)}"
            if logger:
                logger.error(error_msg)
            return {
                'status': False,
                'message': error_msg,
                'downloader': None
            }
            
        if logger:
            logger.info("Menginisialisasi DownloadService...")
            
        downloader = create_download_service(
            workspace=roboflow['workspace'],
            project=roboflow['project'],
            version=roboflow['version'],
            api_key=roboflow['api_key'],
            output_format=roboflow.get('output_format', 'yolov5pytorch'),
            rename_files=download.get('rename_files', True),
            validate_download=download.get('validate_download', True),
            backup_existing=download.get('backup_existing', False),
            logger=logger  # Pass the logger directly
        )
        
        return {
            'status': True,
            'message': 'Downloader created successfully',
            'downloader': downloader
        }
        
    except Exception as e:
        error_msg = f"Gagal membuat downloader: {str(e)}"
        if logger:
            logger.error(error_msg, exc_info=True)
        error_response = create_error_response(error_msg, e, "Downloader Creation Error")
        return {
            'status': False,
            'message': error_msg,
            'downloader': None,
            'error_details': error_response.get('error_details', {})
        }

def create_backend_cleanup_service(logger: Optional[UILogger] = None) -> Dict[str, Any]:
    """
    Create cleanup service instance.
    
    Args:
        logger: UILogger instance untuk logging terpusat
        
    Returns:
        Dictionary with 'status' key and 'service' instance if successful
    """
    try:
        if logger:
            logger.debug("Membuat instance cleanup service...")
        service = create_cleanup_service(logger)  # Pass the logger directly
        
        return {
            'status': True,
            'message': 'Cleanup service created successfully',
            'service': service
        }
        
    except Exception as e:
        error_msg = f"Gagal membuat cleanup service: {str(e)}"
        if logger:
            logger.error(error_msg, exc_info=True)
        error_response = create_error_response(error_msg, e, "Cleanup Service Creation Error")
        return {
            'status': False,
            'message': error_msg,
            'service': None,
            'error_details': error_response.get('error_details', {})
        }