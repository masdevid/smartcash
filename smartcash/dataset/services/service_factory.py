"""
File: smartcash/dataset/services/service_factory.py
Deskripsi: Factory untuk membuat service dataset dengan pendekatan lazy-loading
"""

from typing import Dict, Any, Optional, Type, Union, Callable
import os
from smartcash.common.logger import get_logger

class ServiceFactory:
    """Factory untuk membuat instance service dengan pendekatan lazy-loading."""
    
    def __init__(self, config: Dict[str, Any], logger=None):
        """
        Inisialisasi ServiceFactory.
        
        Args:
            config: Konfigurasi dataset
            logger: Logger untuk mencatat aktivitas
        """
        self.config = config
        self.logger = logger
        self._service_registry = self._register_services()
    
    def _register_services(self) -> Dict[str, Dict[str, Any]]:
        """
        Register service class dan parameter untuk factory.
        
        Returns:
            Dictionary registry service
        """
        return {
            'loader': {
                'class': 'smartcash.dataset.services.loader.dataset_loader.DatasetLoaderService',
                'params': {
                    'dataset_dir': self.config['dataset_dir'],
                    'img_size': self.config['img_size'],
                    'multilayer': self.config.get('multilayer', True),
                    'logger': self.logger
                }
            },
            'validator': {
                'class': 'smartcash.dataset.services.validator.dataset_validator.DatasetValidatorService',
                'params': {
                    'dataset_dir': self.config['dataset_dir'],
                    'logger': self.logger
                }
            },
            'explorer': {
                'class': 'smartcash.dataset.services.explorer.explorer_service.ExplorerService',
                'params': {
                    'config': self.config,
                    'data_dir': self.config['dataset_dir'],
                    'logger': self.logger
                }
            },
            'balancer': {
                'class': 'smartcash.dataset.services.balancer.balance_service.BalanceService',
                'params': {
                    'config': self.config,
                    'data_dir': self.config['dataset_dir'],
                    'logger': self.logger
                }
            },
            'reporter': {
                'class': 'smartcash.dataset.services.reporter.report_service.ReportService',
                'params': {
                    'config': self.config,
                    'data_dir': self.config['dataset_dir'],
                    'logger': self.logger
                }
            },
            'downloader': {
                'class': 'smartcash.dataset.services.downloader.download_service.DownloadService',
                'params': {
                    'config': self.config,
                    'output_dir': self.config['dataset_dir'],
                    'logger': self.logger,
                    'num_workers': self.config.get('num_workers', 4)
                }
            }
        }
    
    def create_service(self, service_name: str) -> Any:
        """
        Buat instance service berdasarkan nama.
        
        Args:
            service_name: Nama service yang akan dibuat
            
        Returns:
            Instance service
            
        Raises:
            ValueError: Jika service tidak terdaftar
        """
        if service_name not in self._service_registry:
            raise ValueError(f"Service '{service_name}' tidak terdaftar")
            
        service_info = self._service_registry[service_name]
        
        try:
            # Import class dinamis
            module_path, class_name = service_info['class'].rsplit('.', 1)
            module = __import__(module_path, fromlist=[class_name])
            service_class = getattr(module, class_name)
            
            # Buat instance dengan parameter yang sesuai
            return service_class(**service_info['params'])
        except ImportError as e:
            if self.logger:
                self.logger.error(f"❌ Gagal mengimpor modul untuk service '{service_name}': {str(e)}")
            raise ImportError(f"Gagal mengimpor service '{service_name}': {str(e)}")
        except Exception as e:
            if self.logger:
                self.logger.error(f"❌ Gagal membuat instance service '{service_name}': {str(e)}")
            raise ValueError(f"Gagal membuat service '{service_name}': {str(e)}")


def get_dataset_service(service_name: str, config: Optional[Dict[str, Any]] = None) -> Any:
    """
    Mendapatkan instance service dataset berdasarkan nama.
    
    Args:
        service_name: Nama service yang akan dibuat
        config: Konfigurasi dataset (opsional)
        
    Returns:
        Instance service
        
    Raises:
        ValueError: Jika service tidak terdaftar
    """
    # Jika config tidak disediakan, gunakan default
    if config is None:
        from smartcash.common.config import get_config_manager
        config_manager = get_config_manager()
        # Dapatkan konfigurasi dataset dari ConfigManager
        dataset_config = config_manager.get_module_config('dataset', {})
        
        # Dapatkan dataset path dari konfigurasi
        dataset_path = dataset_config.get('dataset_path', None)
        
        if not dataset_path:
            raise ValueError("Dataset path tidak ditemukan dalam konfigurasi")
            
        config = {
            'dataset_dir': dataset_path,
            'img_size': dataset_config.get('img_size', 640),
            'multilayer': dataset_config.get('multilayer', True)
        }
    
    # Buat instance factory
    factory = ServiceFactory(config, logger=get_logger())
    
    # Buat dan kembalikan service
    return factory.create_service(service_name)