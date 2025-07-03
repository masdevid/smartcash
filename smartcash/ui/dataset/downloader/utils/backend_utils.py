"""
File: smartcash/ui/dataset/downloader/utils/backend_utils.py
Deskripsi: Backend integration utilities dengan proper scanner usage
"""

from typing import Dict, Any, Tuple

def check_existing_dataset(logger=None) -> Tuple[bool, int, Dict[str, Any]]:
    """
    Check existing dataset menggunakan backend scanner.
    
    Args:
        logger: Logger instance
        
    Returns:
        Tuple[bool, int, Dict]: (has_content, total_images, summary_data)
    """
    try:
        from smartcash.dataset.downloader.dataset_scanner import create_dataset_scanner
        
        scanner = create_dataset_scanner(logger)
        
        # Quick check untuk determine if content exists
        if not scanner.quick_check_existing():
            return False, 0, {}
        
        # Get detailed summary jika ada content
        result = scanner.scan_existing_dataset_parallel()
        
        if result.get('status') == 'success':
            summary = result.get('summary', {})
            splits = result.get('splits', {})
            
            # Extract split breakdown dengan one-liner
            split_breakdown = {name: data.get('images', 0) 
                              for name, data in splits.items() 
                              if data.get('status') == 'success'}
            
            return True, summary.get('total_images', 0), {
                'summary': summary,
                'splits': split_breakdown,
                'downloads': result.get('downloads', {}),
                'scan_result': result
            }
        
        return True, 0, {}
        
    except Exception as e:
        if logger:
            logger.warning(f"⚠️ Error checking existing dataset: {str(e)}")
        return False, 0, {}

def get_cleanup_targets(logger=None) -> Dict[str, Any]:
    """
    Get cleanup targets menggunakan backend scanner.
    
    Args:
        logger: Logger instance
        
    Returns:
        Dictionary dengan cleanup targets atau error info
    """
    try:
        from smartcash.dataset.downloader.dataset_scanner import create_dataset_scanner
        
        scanner = create_dataset_scanner(logger)
        return scanner.get_cleanup_targets()
        
    except Exception as e:
        if logger:
            logger.error(f"❌ Error getting cleanup targets: {str(e)}")
        return {
            'status': 'error',
            'message': f'Error getting cleanup targets: {str(e)}',
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

def create_backend_downloader(ui_config: Dict[str, Any], logger=None):
    """
    Create downloader instance dari UI config.
    
    Args:
        ui_config: UI configuration
        logger: Logger instance
        
    Returns:
        Downloader instance atau None
    """
    try:
        from smartcash.dataset.downloader import get_downloader_instance, create_ui_compatible_config
        
        # Convert UI config ke backend format dan enhance dengan optimal settings
        service_config = create_ui_compatible_config(ui_config)
        download_config = ui_config.get('download', {})
        
        # Update dengan performance settings dari download_config
        service_config.update({
            'max_workers': download_config.get('max_workers', 4),
            'parallel_downloads': download_config.get('parallel_downloads', True),
            'chunk_size': download_config.get('chunk_size', 8192),
            'timeout': download_config.get('timeout', 30),
            'retry_count': download_config.get('retry_count', 3)
        })
        
        return get_downloader_instance(service_config, logger)
        
    except Exception as e:
        if logger:
            logger.error(f"❌ Error creating backend downloader: {str(e)}")
        return None

def create_backend_cleanup_service(logger=None):
    """
    Create cleanup service instance.
    
    Args:
        logger: Logger instance
        
    Returns:
        Cleanup service instance atau None
    """
    try:
        from smartcash.dataset.downloader.cleanup_service import create_cleanup_service
        return create_cleanup_service(logger)
        
    except Exception as e:
        if logger:
            logger.error(f"❌ Error creating cleanup service: {str(e)}")
        return None