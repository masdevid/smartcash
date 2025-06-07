"""
File: smartcash/dataset/downloader/download_service.py
Deskripsi: Updated download service menggunakan shared base components
"""

import time
from typing import Dict, Any, Optional, Callable
from pathlib import Path
from smartcash.dataset.downloader.base import (
    BaseDownloaderComponent, ValidationHelper, PathHelper, FileHelper
)
from smartcash.dataset.downloader.roboflow_client import create_roboflow_client
from smartcash.dataset.downloader.file_processor import create_file_processor
from smartcash.dataset.downloader.validators import create_dataset_validator


class DownloadService(BaseDownloaderComponent):
    """Download service menggunakan shared components"""
    
    def __init__(self, config: Dict[str, Any], logger=None):
        super().__init__(logger)
        self.config = config
        
        # Lazy initialization
        self._roboflow_client = None
        self._file_processor = None
        self._validator = None
    
    @property
    def roboflow_client(self):
        if not self._roboflow_client:
            self._roboflow_client = create_roboflow_client(self.config.get('api_key', ''), self.logger)
            if self._progress_callback:
                self._roboflow_client.set_progress_callback(self._progress_callback)
        return self._roboflow_client
    
    @property
    def file_processor(self):
        if not self._file_processor:
            self._file_processor = create_file_processor(self.logger)
            if self._progress_callback:
                self._file_processor.set_progress_callback(self._progress_callback)
        return self._file_processor
    
    @property
    def validator(self):
        if not self._validator:
            self._validator = create_dataset_validator(self.logger)
        return self._validator
    
    def download_dataset(self) -> Dict[str, Any]:
        """Download dataset dengan streamlined flow"""
        start_time = time.time()
        
        try:
            # Validate config menggunakan shared helper
            validation = ValidationHelper.validate_config(
                self.config, ['workspace', 'project', 'version', 'api_key']
            )
            if not validation['valid']:
                return self._create_error_result(f"Config tidak valid: {'; '.join(validation['errors'])}")
            
            # Extract parameters
            params = self._extract_params()
            
            # Setup paths menggunakan shared helper
            paths = PathHelper.setup_download_paths(
                params['workspace'], params['project'], params['version']
            )
            
            # Execute download flow
            metadata = self._get_metadata(params)
            
            if params['backup_existing']:
                self._handle_backup(paths)
            
            self._download_and_extract(metadata['download_url'], paths)
            stats = self._organize_dataset(paths, params)
            
            if params['validate_download']:
                self._validate_results(paths['final_dir'])
            
            # Cleanup menggunakan shared helper
            FileHelper.cleanup_temp(paths['temp_dir'])
            
            return self._create_success_result(
                output_dir=str(paths['final_dir']),
                stats=stats,
                duration=time.time() - start_time,
                uuid_renamed=params['rename_files'],
                metadata={
                    'workspace': params['workspace'],
                    'project': params['project'],
                    'version': params['version']
                }
            )
            
        except Exception as e:
            return self._create_error_result(f"Download error: {str(e)}")
    
    def _extract_params(self) -> Dict[str, Any]:
        """Extract parameters dengan defaults"""
        return {
            'workspace': self.config.get('workspace', ''),
            'project': self.config.get('project', ''),
            'version': self.config.get('version', ''),
            'api_key': self.config.get('api_key', ''),
            'output_format': self.config.get('output_format', 'yolov5pytorch'),
            'validate_download': self.config.get('validate_download', True),
            'backup_existing': self.config.get('backup_existing', False),
            'rename_files': self.config.get('rename_files', True)
        }
    
    def _get_metadata(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get metadata menggunakan roboflow client"""
        self._notify_progress('metadata', 0, 100, "📊 Getting metadata...")
        
        result = self.roboflow_client.get_dataset_metadata(
            params['workspace'], params['project'], 
            params['version'], params['output_format']
        )
        
        if result['status'] != 'success':
            raise Exception(f"Metadata failed: {result['message']}")
        
        return result
    
    def _handle_backup(self, paths: Dict[str, Path]) -> None:
        """Handle backup menggunakan shared helper"""
        if self._has_existing_dataset(paths['final_dir']):
            self.logger.info("💾 Creating backup...")
            backup_result = FileHelper.backup_directory(paths['final_dir'])
            
            if backup_result['success']:
                self.logger.info(f"✅ Backup: {backup_result['backup_path']}")
            else:
                self.logger.warning(f"⚠️ Backup failed: {backup_result['message']}")
    
    def _download_and_extract(self, download_url: str, paths: Dict[str, Path]) -> None:
        """Download dan extract menggunakan components"""
        # Download
        download_result = self.roboflow_client.download_dataset(download_url, paths['temp_zip'])
        if download_result['status'] != 'success':
            raise Exception(f"Download failed: {download_result['message']}")
        
        # Extract
        extract_result = self.file_processor.extract_zip(paths['temp_zip'], paths['extract_dir'])
        if extract_result['status'] != 'success':
            raise Exception(f"Extract failed: {extract_result['message']}")
    
    def _organize_dataset(self, paths: Dict[str, Path], params: Dict[str, Any]) -> Dict[str, Any]:
        """Organize dataset menggunakan file processor"""
        if params['rename_files']:
            result = self.file_processor.organize_dataset_with_renaming(
                paths['extract_dir'], paths['final_dir']
            )
        else:
            result = self.file_processor.organize_dataset(
                paths['extract_dir'], paths['final_dir']
            )
        
        if result['status'] != 'success':
            raise Exception(f"Organization failed: {result['message']}")
        
        stats = {
            'total_images': result['total_images'],
            'total_labels': result['total_labels'],
            'splits': result['splits']
        }
        
        if params['rename_files']:
            naming_stats = self.file_processor.get_naming_statistics()
            stats['naming_stats'] = naming_stats
            self.logger.info(f"🔄 UUID renaming: {naming_stats.get('total_files', 0)} files")
        
        return stats
    
    def _validate_results(self, final_dir: Path) -> None:
        """Validate results menggunakan validator"""
        self._notify_progress('validate', 80, 100, "✅ Validating results...")
        
        validation_result = self.validator.validate_extracted_dataset(final_dir)
        
        if not validation_result['valid'] and validation_result.get('issues'):
            self.logger.warning(f"⚠️ Validation issues: {', '.join(validation_result['issues'][:3])}")
    
    def _has_existing_dataset(self, dataset_dir: Path) -> bool:
        """Check existing dataset"""
        if not dataset_dir.exists():
            return False
        
        return any(
            (dataset_dir / split).exists() and any((dataset_dir / split).iterdir())
            for split in ['train', 'valid', 'test']
        )
    
    def get_service_info(self) -> Dict[str, Any]:
        """Get service information"""
        from smartcash.common.environment import get_environment_manager
        env_manager = get_environment_manager()
        
        return {
            'has_progress_callback': self._progress_callback is not None,
            'environment': {
                'is_colab': env_manager.is_colab,
                'drive_mounted': env_manager.is_drive_mounted
            },
            'components_loaded': {
                'roboflow_client': self._roboflow_client is not None,
                'file_processor': self._file_processor is not None,
                'validator': self._validator is not None
            },
            'uuid_support': True
        }


def create_download_service(config: Dict[str, Any], logger=None) -> Optional[DownloadService]:
    """Factory untuk download service dengan validation"""
    from smartcash.common.logger import get_logger
    logger = logger or get_logger('downloader.factory')
    
    try:
        if not config:
            logger.error("❌ Config kosong")
            return None
        
        # Apply defaults
        defaults = {
            'workspace': 'smartcash-wo2us',
            'project': 'rupiah-emisi-2022',
            'version': '3',
            'rename_files': True,
            'organize_dataset': True,
            'validate_download': True,
            'backup_existing': False
        }
        
        for key, default_value in defaults.items():
            if key not in config or not config[key]:
                config[key] = default_value
        
        # Validate menggunakan shared helper
        validation = ValidationHelper.validate_config(
            config, ['workspace', 'project', 'version', 'api_key']
        )
        
        if not validation['valid']:
            logger.error(f"❌ Config validation failed: {'; '.join(validation['errors'])}")
            return None
        
        return DownloadService(config, logger)
        
    except Exception as e:
        logger.error(f"❌ Error creating download service: {str(e)}")
        return None