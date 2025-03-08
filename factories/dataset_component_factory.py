# File: smartcash/factories/dataset_component_factory.py
# Author: Alfrida Sabar
# Deskripsi: Factory untuk pembuatan komponen dataset dengan integrasi utils dan handlers

from typing import Dict, Optional, Union, Any

from smartcash.utils.logger import SmartCashLogger
from smartcash.utils.dataset import EnhancedDatasetValidator
from smartcash.utils.augmentation import AugmentationManager
from smartcash.utils.layer_config_manager import get_layer_config

# Imports dari handlers
from smartcash.handlers.dataset.multilayer_dataset import MultilayerDataset
from smartcash.handlers.dataset.core.dataset_transformer import DatasetTransformer
from smartcash.handlers.dataset.operations.dataset_split_operation import DatasetSplitOperation
from smartcash.handlers.dataset.operations.dataset_merger_operation import DatasetMergerOperation


class DatasetComponentFactory:
    """
    Factory untuk membuat komponen dataset yang terintegrasi antara
    utils/dataset dan handlers/dataset.
    
    Implementasi Factory Pattern untuk memastikan tidak ada duplikasi
    saat membuat objek dan semua dependency dikelola dengan tepat.
    """
    
    @staticmethod
    def create_validator(
        config: Dict,
        data_dir: Optional[str] = None,
        logger: Optional[SmartCashLogger] = None,
        **kwargs
    ) -> EnhancedDatasetValidator:
        """
        Membuat validator dataset.
        
        Args:
            config: Konfigurasi
            data_dir: Direktori dataset
            logger: Custom logger
            **kwargs: Parameter tambahan
            
        Returns:
            Instance EnhancedDatasetValidator
        """
        num_workers = kwargs.get('num_workers', config.get('model', {}).get('workers', 4))
        
        return EnhancedDatasetValidator(
            config=config,
            data_dir=data_dir or config.get('data_dir', 'data'),
            logger=logger,
            num_workers=num_workers
        )
    
    @staticmethod
    def create_augmentor(
        config: Dict,
        output_dir: Optional[str] = None,
        logger: Optional[SmartCashLogger] = None,
        **kwargs
    ) -> AugmentationManager:
        """
        Membuat augmentor dataset.
        
        Args:
            config: Konfigurasi
            output_dir: Direktori output
            logger: Custom logger
            **kwargs: Parameter tambahan
            
        Returns:
            Instance AugmentationManager
        """
        num_workers = kwargs.get('num_workers', config.get('model', {}).get('workers', 4))
        checkpoint_interval = kwargs.get('checkpoint_interval', 50)
        
        return AugmentationManager(
            config=config,
            output_dir=output_dir or config.get('data_dir', 'data'),
            logger=logger,
            num_workers=num_workers,
            checkpoint_interval=checkpoint_interval
        )
    
    @staticmethod
    def create_multilayer_dataset(
        config: Dict,
        data_path: str,
        mode: str = 'train',
        transform=None,
        logger: Optional[SmartCashLogger] = None,
        **kwargs
    ) -> MultilayerDataset:
        """
        Membuat multilayer dataset.
        
        Args:
            config: Konfigurasi
            data_path: Path ke data
            mode: Mode dataset ('train', 'val', 'test')
            transform: Transformasi kustom
            logger: Custom logger
            **kwargs: Parameter tambahan
            
        Returns:
            Instance MultilayerDataset
        """
        img_size = tuple(config.get('model', {}).get('img_size', [640, 640]))
        layers = kwargs.get('layers', config.get('layers', ['banknote']))
        require_all_layers = kwargs.get('require_all_layers', False)
        
        return MultilayerDataset(
            data_path=data_path,
            img_size=img_size,
            mode=mode,
            transform=transform,
            layers=layers,
            require_all_layers=require_all_layers,
            logger=logger
        )
    
    @staticmethod
    def create_transformer(
        config: Dict,
        logger: Optional[SmartCashLogger] = None
    ) -> DatasetTransformer:
        """
        Membuat dataset transformer.
        
        Args:
            config: Konfigurasi
            logger: Custom logger
            
        Returns:
            Instance DatasetTransformer
        """
        img_size = tuple(config.get('model', {}).get('img_size', [640, 640]))
        
        return DatasetTransformer(
            config=config,
            img_size=img_size,
            logger=logger
        )
    
    @staticmethod
    def create_dataset_manager(
        config: Dict,
        data_dir: Optional[str] = None,
        cache_dir: Optional[str] = None,
        logger: Optional[SmartCashLogger] = None
    ) -> 'DatasetManager':
        """
        Membuat dataset manager terintegrasi.
        
        Args:
            config: Konfigurasi
            data_dir: Direktori dataset
            cache_dir: Direktori cache
            logger: Custom logger
            
        Returns:
            Instance DatasetManager
        """
        # Import di sini untuk menghindari circular import
        from smartcash.handlers.dataset.dataset_manager import DatasetManager
        
        return DatasetManager(
            config=config,
            data_dir=data_dir,
            cache_dir=cache_dir,
            logger=logger
        )
    
    @staticmethod
    def create_dataset_splitter(
        data_dir: str,
        output_dir: Optional[str] = None,
        logger: Optional[SmartCashLogger] = None
    ) -> DatasetSplitOperation:
        """
        Membuat dataset splitter.
        
        Args:
            data_dir: Direktori dataset
            output_dir: Direktori output
            logger: Custom logger
            
        Returns:
            Instance DatasetSplitOperation
        """
        return DatasetSplitOperation(
            data_dir=data_dir,
            output_dir=output_dir,
            logger=logger
        )
    
    @staticmethod
    def create_dataset_merger(
        data_dir: str,
        output_dir: Optional[str] = None,
        logger: Optional[SmartCashLogger] = None
    ) -> DatasetMergerOperation:
        """
        Membuat dataset merger.
        
        Args:
            data_dir: Direktori dataset
            output_dir: Direktori output
            logger: Custom logger
            
        Returns:
            Instance DatasetMergerOperation
        """
        return DatasetMergerOperation(
            data_dir=data_dir,
            output_dir=output_dir,
            logger=logger
        )
    
    @staticmethod
    def create_multilayer_dataset(
        config: Dict,
        data_path: str,
        mode: str = 'train',
        transform=None,
        logger: Optional[SmartCashLogger] = None,
        **kwargs
    ) -> MultilayerDataset:
        """
        Membuat multilayer dataset.
        """
        img_size = tuple(config.get('model', {}).get('img_size', [640, 640]))
        
        # Gunakan layer_config_manager untuk mendapatkan layer jika tidak disediakan
        if 'layers' not in kwargs:
            layer_config = get_layer_config()
            layers = kwargs.get('layers', config.get('layers', layer_config.get_layer_names()))
        else:
            layers = kwargs.get('layers')
        
        require_all_layers = kwargs.get('require_all_layers', False)
        
        return MultilayerDataset(
            data_path=data_path,
            img_size=img_size,
        mode=mode,
        transform=transform,
        layers=layers,
        require_all_layers=require_all_layers,
        logger=logger
    )