"""
File: smartcash/dataset/downloader/__init__.py
Deskripsi: Enhanced factory dan exports dengan UUID file renaming support dan one-liner style
"""

from typing import Dict, Any, Optional, List
from smartcash.common.logger import get_logger

def get_downloader_instance(config: Dict[str, Any], logger=None) -> Optional['DownloadService']:
    """
    Enhanced factory untuk downloader instance dengan UUID renaming support.
    
    Args:
        config: Configuration dictionary dengan format yang benar
        logger: Optional logger instance
        
    Returns:
        Downloader service instance atau None jika terjadi error
    """
    try:
        # Validasi config
        if not config:
            if logger:
                logger.error("❌ Config tidak boleh kosong")
            return None
        
        # Validasi field yang diperlukan dengan nilai default
        required_fields = ['workspace', 'project', 'version', 'api_key']
        default_values = {
            'workspace': 'smartcash-wo2us',
            'project': 'rupiah-emisi-2022',
            'version': '3'
        }
        
        # Terapkan nilai default jika field tidak ada atau kosong
        for field in required_fields:
            if field not in config or not config[field]:
                if field in default_values:
                    config[field] = default_values[field]
                    if logger:
                        logger.info(f"🔄 Menggunakan nilai default untuk {field}: {config[field]}")
        
        # Enhanced default values untuk UUID renaming
        enhanced_defaults = {
            'rename_files': True,  # Enable UUID renaming by default
            'output_format': 'yolov5pytorch',
            'validate_download': True,
            'organize_dataset': True,
            'backup_existing': False
        }
        
        for field, default_value in enhanced_defaults.items():
            if field not in config:
                config[field] = default_value
                if logger and field == 'rename_files':
                    logger.info(f"🔄 UUID file renaming diaktifkan secara default")
        
        # Periksa lagi setelah menerapkan nilai default
        missing_fields = [field for field in required_fields if field not in config or not config[field]]
        
        if missing_fields:
            if logger:
                logger.error(f"❌ Konfigurasi tidak lengkap: {', '.join(missing_fields)} tidak ditemukan")
            return None
        
        # Buat service
        from smartcash.dataset.downloader.download_service import create_download_service
        return create_download_service(config, logger)
    except ImportError as e:
        if logger:
            logger.error(f"❌ Gagal import module: {str(e)}")
        return None
    except Exception as e:
        if logger:
            logger.error(f"❌ Gagal membuat download service: {str(e)}")
        return None

def create_roboflow_downloader(api_key: str, config: Dict[str, Any] = None, logger=None, enable_uuid_renaming: bool = True):
    """
    Enhanced Roboflow downloader creation dengan UUID renaming support.
    
    Args:
        api_key: Roboflow API key
        config: Optional config override
        logger: Optional logger
        enable_uuid_renaming: Enable UUID file renaming
        
    Returns:
        Configured downloader instance dengan UUID support
    """
    # One-liner config merging dengan UUID support
    merged_config = {
        'api_key': api_key, 'output_format': 'yolov5pytorch',
        'validate_download': True, 'organize_dataset': True, 'backup_existing': False,
        'rename_files': enable_uuid_renaming,
        **(config or {})
    }
    
    return get_downloader_instance(merged_config, logger)

def create_optimized_downloader(api_key: str, workspace: str, project: str, version: str, 
                               logger=None, enable_uuid_renaming: bool = True, **kwargs) -> Any:
    """
    Enhanced optimized downloader dengan pre-configured parameters dan UUID support.
    
    Args:
        api_key: Roboflow API key
        workspace: Roboflow workspace
        project: Roboflow project
        version: Dataset version
        logger: Optional logger
        enable_uuid_renaming: Enable UUID file renaming
        **kwargs: Additional config parameters
        
    Returns:
        Pre-configured downloader instance dengan UUID support
    """
    # One-liner optimized config creation dengan UUID
    optimized_config = {
        'api_key': api_key, 'workspace': workspace, 'project': project, 'version': version,
        'output_format': 'yolov5pytorch', 'validate_download': True, 'organize_dataset': True,
        'backup_existing': False, 'retry_count': 3, 'timeout': 30, 'chunk_size': 8192,
        'rename_files': enable_uuid_renaming,
        **kwargs
    }
    
    return get_downloader_instance(optimized_config, logger)

# One-liner component factories dengan enhanced features
def create_roboflow_client(api_key: str, logger=None):
    """One-liner Roboflow client factory"""
    from smartcash.dataset.downloader.roboflow_client import create_roboflow_client
    return create_roboflow_client(api_key, logger)

def create_file_processor(logger=None, max_workers: int = None, enable_uuid_renaming: bool = True):
    """Enhanced file processor factory dengan UUID support"""
    from smartcash.dataset.downloader.file_processor import create_file_processor
    processor = create_file_processor(logger, max_workers)
    
    # UUID renaming sudah built-in di FileProcessor yang baru
    if logger and enable_uuid_renaming:
        logger.info("🔄 File processor dengan UUID renaming support diaktifkan")
    
    return processor

def create_progress_tracker():
    """One-liner progress tracker factory"""
    from smartcash.dataset.downloader.progress_tracker import create_download_tracker
    return create_download_tracker()

def create_dataset_validator(logger=None, max_workers: int = None):
    """One-liner dataset validator factory"""
    from smartcash.dataset.downloader.validators import create_dataset_validator
    return create_dataset_validator(logger, max_workers)

# One-liner utility functions dengan UUID support
def validate_config_quick(config: Dict[str, Any]) -> bool:
    """Quick one-liner config validation"""
    required_fields = ['api_key', 'workspace', 'project', 'version']
    return all(config.get(field, '').strip() for field in required_fields)

def get_default_download_config(api_key: str = "", workspace: str = "", project: str = "", version: str = "", 
                               enable_uuid_renaming: bool = True) -> Dict[str, Any]:
    """Enhanced default config creation dengan UUID support"""
    return {
        'api_key': api_key, 'workspace': workspace, 'project': project, 'version': version,
        'output_format': 'yolov5pytorch', 'validate_download': True, 'organize_dataset': True,
        'backup_existing': False, 'retry_count': 3, 'timeout': 30, 'chunk_size': 8192,
        'rename_files': enable_uuid_renaming
    }

def create_download_session(api_key: str, workspace: str, project: str, version: str, 
                           logger=None, enable_uuid_renaming: bool = True) -> Dict[str, Any]:
    """
    Enhanced download session creation dengan all components dan UUID support.
    
    Returns:
        Dictionary containing all downloader components dengan UUID support
    """
    config = get_default_download_config(api_key, workspace, project, version, enable_uuid_renaming)
    
    return {
        'service': get_downloader_instance(config, logger),
        'client': create_roboflow_client(api_key, logger),
        'processor': create_file_processor(logger, enable_uuid_renaming=enable_uuid_renaming),
        'validator': create_dataset_validator(logger),
        'tracker': create_progress_tracker(),
        'config': config,
        'uuid_support': enable_uuid_renaming
    }

# Enhanced error handling factories
def create_safe_downloader(config: Dict[str, Any], logger=None, fallback_config: Dict[str, Any] = None):
    """Enhanced safe downloader creation dengan fallback dan UUID support"""
    try:
        return get_downloader_instance(config, logger)
    except Exception as e:
        logger = logger or get_logger('downloader.safe_factory')
        logger.warning(f"⚠️ Primary config failed: {str(e)}, using fallback")
        fallback = fallback_config or get_default_download_config(enable_uuid_renaming=True)
        return get_downloader_instance(fallback, logger) if fallback else None

def get_downloader_info(downloader_instance) -> Dict[str, Any]:
    """Enhanced downloader instance information dengan UUID support info"""
    base_info = (downloader_instance.get_service_info() if hasattr(downloader_instance, 'get_service_info') 
                else {'type': type(downloader_instance).__name__, 'available': True})
    
    # Add UUID support info jika tersedia
    if hasattr(downloader_instance, 'file_processor') and hasattr(downloader_instance.file_processor, 'naming_manager'):
        base_info['uuid_renaming_support'] = True
        base_info['naming_statistics'] = downloader_instance.file_processor.get_naming_statistics()
    else:
        base_info['uuid_renaming_support'] = False
    
    return base_info

# UUID-specific utility functions
def create_uuid_enabled_downloader(api_key: str, workspace: str = None, project: str = None, 
                                  version: str = None, logger=None) -> Any:
    """One-liner UUID-enabled downloader creation"""
    workspace = workspace or 'smartcash-wo2us'
    project = project or 'rupiah-emisi-2022'
    version = version or '3'
    
    return create_optimized_downloader(api_key, workspace, project, version, logger, enable_uuid_renaming=True)

def validate_uuid_support(downloader_instance) -> bool:
    """One-liner UUID support validation"""
    return (hasattr(downloader_instance, 'file_processor') and 
            hasattr(downloader_instance.file_processor, 'naming_manager'))

def get_uuid_statistics(downloader_instance) -> Dict[str, Any]:
    """One-liner UUID statistics retrieval"""
    if validate_uuid_support(downloader_instance):
        return downloader_instance.file_processor.get_naming_statistics()
    return {'uuid_support': False, 'message': 'UUID renaming tidak didukung'}

# Batch operations dengan UUID support
def create_batch_downloaders(configs: List[Dict[str, Any]], logger=None, enable_uuid_renaming: bool = True) -> List[Any]:
    """Create multiple downloaders dengan UUID support untuk batch processing"""
    downloaders = []
    for config in configs:
        if 'rename_files' not in config:
            config['rename_files'] = enable_uuid_renaming
        
        downloader = get_downloader_instance(config, logger)
        if downloader:
            downloaders.append(downloader)
        elif logger:
            logger.warning(f"⚠️ Gagal membuat downloader untuk config: {config.get('project', 'unknown')}")
    
    return downloaders

def download_multiple_datasets(api_key: str, dataset_configs: List[Dict[str, str]], 
                              logger=None, enable_uuid_renaming: bool = True) -> Dict[str, Any]:
    """One-liner multiple dataset download dengan UUID consistency"""
    results = {}
    
    for dataset_config in dataset_configs:
        workspace = dataset_config.get('workspace', 'smartcash-wo2us')
        project = dataset_config.get('project', 'rupiah-emisi-2022')
        version = dataset_config.get('version', '3')
        
        downloader = create_optimized_downloader(api_key, workspace, project, version, logger, enable_uuid_renaming)
        if downloader:
            result = downloader.download_dataset()
            results[f"{workspace}/{project}:{version}"] = result
        else:
            results[f"{workspace}/{project}:{version}"] = {'status': 'error', 'message': 'Failed to create downloader'}
    
    return results

# Compatibility aliases
create_downloader = get_downloader_instance  # Backward compatibility
get_roboflow_downloader = create_roboflow_downloader  # Alternative name

# Export everything untuk comprehensive access
__all__ = [
    # Main factories
    'get_downloader_instance', 'create_roboflow_downloader', 'create_optimized_downloader',
    
    # Component factories  
    'create_roboflow_client', 'create_file_processor', 'create_progress_tracker', 'create_dataset_validator',
    
    # Utility functions
    'validate_config_quick', 'get_default_download_config', 'create_download_session',
    
    # Enhanced factories
    'create_safe_downloader', 'get_downloader_info',
    
    # UUID-specific functions
    'create_uuid_enabled_downloader', 'validate_uuid_support', 'get_uuid_statistics',
    
    # Batch operations
    'create_batch_downloaders', 'download_multiple_datasets',
    
    # Compatibility aliases
    'create_downloader', 'get_roboflow_downloader'
]

# One-liner version info
__version__ = '2.1.0'
__description__ = 'Enhanced SmartCash dataset downloader dengan UUID file renaming dan optimized performance'