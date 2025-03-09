# File: smartcash/factories/dataset_component_factory.py
# Author: Alfrida Sabar
# Deskripsi: Factory pattern untuk pembuatan komponen dataset dengan integrasi utils dan handlers

from typing import Dict, List, Optional, Union, Any, Callable, Type

from smartcash.utils.logger import SmartCashLogger, get_logger
from smartcash.utils.dataset import EnhancedDatasetValidator
from smartcash.utils.augmentation import AugmentationManager
from smartcash.utils.layer_config_manager import get_layer_config
from smartcash.utils.cache import CacheManager

# Lazy imports untuk menghindari circular import
_multilayer_dataset = None
_dataset_transformer = None
_dataset_validator = None
_dataset_augmentor = None
_dataset_balancer = None
_dataset_split_operation = None
_dataset_merge_operation = None
_dataset_explorer_facade = None
_visualization_facade = None


class DatasetComponentFactory:
    """
    Factory untuk membuat komponen dataset yang terintegrasi antara
    utils/dataset dan handlers/dataset.
    
    Implementasi Factory Pattern untuk memastikan tidak ada duplikasi
    saat membuat objek dan semua dependency dikelola dengan tepat.
    """
    
    @staticmethod
    def _get_class(module_path: str, class_name: str) -> Type:
        """
        Load kelas secara dinamis untuk lazy loading.
        
        Args:
            module_path: Path ke modul
            class_name: Nama kelas
            
        Returns:
            Kelas yang diminta
        """
        import importlib
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    
    @staticmethod
    def get_multilayer_dataset():
        """Lazy import untuk MultilayerDataset."""
        global _multilayer_dataset
        if _multilayer_dataset is None:
            from smartcash.handlers.dataset.multilayer.multilayer_dataset import MultilayerDataset
            _multilayer_dataset = MultilayerDataset
        return _multilayer_dataset
    
    @staticmethod
    def get_dataset_transformer():
        """Lazy import untuk DatasetTransformer."""
        global _dataset_transformer
        if _dataset_transformer is None:
            from smartcash.handlers.dataset.core.dataset_transformer import DatasetTransformer
            _dataset_transformer = DatasetTransformer
        return _dataset_transformer
    
    @staticmethod
    def get_dataset_validator():
        """Lazy import untuk DatasetValidator."""
        global _dataset_validator
        if _dataset_validator is None:
            from smartcash.handlers.dataset.core.dataset_validator import DatasetValidator
            _dataset_validator = DatasetValidator
        return _dataset_validator
    
    @staticmethod
    def get_dataset_augmentor():
        """Lazy import untuk DatasetAugmentor."""
        global _dataset_augmentor
        if _dataset_augmentor is None:
            from smartcash.handlers.dataset.core.dataset_augmentor import DatasetAugmentor
            _dataset_augmentor = DatasetAugmentor
        return _dataset_augmentor
    
    @staticmethod
    def get_dataset_balancer():
        """Lazy import untuk DatasetBalancer."""
        global _dataset_balancer
        if _dataset_balancer is None:
            from smartcash.handlers.dataset.core.dataset_balancer import DatasetBalancer
            _dataset_balancer = DatasetBalancer
        return _dataset_balancer
    
    @staticmethod
    def get_dataset_split_operation():
        """Lazy import untuk DatasetSplitOperation."""
        global _dataset_split_operation
        if _dataset_split_operation is None:
            from smartcash.handlers.dataset.operations.dataset_split_operation import DatasetSplitOperation
            _dataset_split_operation = DatasetSplitOperation
        return _dataset_split_operation
    
    @staticmethod
    def get_dataset_merge_operation():
        """Lazy import untuk DatasetMergeOperation."""
        global _dataset_merge_operation
        if _dataset_merge_operation is None:
            from smartcash.handlers.dataset.operations.dataset_merge_operation import DatasetMergeOperation
            _dataset_merge_operation = DatasetMergeOperation
        return _dataset_merge_operation
    
    @staticmethod
    def get_dataset_explorer_facade():
        """Lazy import untuk DatasetExplorerFacade."""
        global _dataset_explorer_facade
        if _dataset_explorer_facade is None:
            from smartcash.handlers.dataset.explorers.dataset_explorer_facade import DatasetExplorerFacade
            _dataset_explorer_facade = DatasetExplorerFacade
        return _dataset_explorer_facade
    
    @staticmethod
    def get_visualization_facade():
        """Lazy import untuk VisualizationFacade."""
        global _visualization_facade
        if _visualization_facade is None:
            from smartcash.handlers.dataset.facades.visualization_facade import VisualizationFacade
            _visualization_facade = VisualizationFacade
        return _visualization_facade
    
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
    def create_dataset_validator(
        config: Dict,
        data_dir: Optional[str] = None,
        logger: Optional[SmartCashLogger] = None,
        **kwargs
    ):
        """
        Membuat dataset validator dari handlers/dataset.
        
        Args:
            config: Konfigurasi
            data_dir: Direktori dataset
            logger: Custom logger
            **kwargs: Parameter tambahan
            
        Returns:
            Instance DatasetValidator
        """
        DatasetValidator = DatasetComponentFactory.get_dataset_validator()
        num_workers = kwargs.get('num_workers', config.get('model', {}).get('workers', 4))
        
        return DatasetValidator(
            config=config,
            data_dir=data_dir or config.get('data_dir', 'data'),
            logger=logger or get_logger("dataset_validator"),
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
    def create_dataset_augmentor(
        config: Dict,
        data_dir: Optional[str] = None,
        logger: Optional[SmartCashLogger] = None,
        **kwargs
    ):
        """
        Membuat dataset augmentor dari handlers/dataset.
        
        Args:
            config: Konfigurasi
            data_dir: Direktori dataset
            logger: Custom logger
            **kwargs: Parameter tambahan
            
        Returns:
            Instance DatasetAugmentor
        """
        DatasetAugmentor = DatasetComponentFactory.get_dataset_augmentor()
        num_workers = kwargs.get('num_workers', config.get('model', {}).get('workers', 4))
        
        return DatasetAugmentor(
            config=config,
            data_dir=data_dir or config.get('data_dir', 'data'),
            logger=logger or get_logger("dataset_augmentor"),
            num_workers=num_workers
        )
    
    @staticmethod
    def create_dataset_balancer(
        data_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        logger: Optional[SmartCashLogger] = None,
        **kwargs
    ):
        """
        Membuat dataset balancer dari handlers/dataset.
        
        Args:
            data_dir: Direktori dataset
            output_dir: Direktori output
            logger: Custom logger
            **kwargs: Parameter tambahan
            
        Returns:
            Instance DatasetBalancer
        """
        DatasetBalancer = DatasetComponentFactory.get_dataset_balancer()
        
        return DatasetBalancer(
            data_dir=data_dir,
            output_dir=output_dir,
            logger=logger or get_logger("dataset_balancer")
        )
    
    @staticmethod
    def create_multilayer_dataset(
        config: Dict,
        data_path: str,
        mode: str = 'train',
        transform=None,
        logger: Optional[SmartCashLogger] = None,
        **kwargs
    ):
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
        MultilayerDataset = DatasetComponentFactory.get_multilayer_dataset()
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
            logger=logger or get_logger("multilayer_dataset"),
            config=config
        )
    
    @staticmethod
    def create_transformer(
        config: Dict,
        logger: Optional[SmartCashLogger] = None
    ):
        """
        Membuat dataset transformer.
        
        Args:
            config: Konfigurasi
            logger: Custom logger
            
        Returns:
            Instance DatasetTransformer
        """
        DatasetTransformer = DatasetComponentFactory.get_dataset_transformer()
        img_size = tuple(config.get('model', {}).get('img_size', [640, 640]))
        
        return DatasetTransformer(
            config=config,
            img_size=img_size,
            logger=logger or get_logger("dataset_transformer")
        )
    
    @staticmethod
    def create_dataset_manager(
        config: Dict,
        data_dir: Optional[str] = None,
        cache_dir: Optional[str] = None,
        logger: Optional[SmartCashLogger] = None
    ):
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
            logger=logger or get_logger("dataset_manager")
        )
    
    @staticmethod
    def create_dataset_splitter(
        data_dir: str,
        output_dir: Optional[str] = None,
        logger: Optional[SmartCashLogger] = None
    ):
        """
        Membuat dataset splitter.
        
        Args:
            data_dir: Direktori dataset
            output_dir: Direktori output
            logger: Custom logger
            
        Returns:
            Instance DatasetSplitOperation
        """
        DatasetSplitOperation = DatasetComponentFactory.get_dataset_split_operation()
        
        return DatasetSplitOperation(
            data_dir=data_dir,
            output_dir=output_dir,
            logger=logger or get_logger("dataset_splitter")
        )
    
    @staticmethod
    def create_dataset_merger(
        data_dir: str,
        output_dir: Optional[str] = None,
        logger: Optional[SmartCashLogger] = None
    ):
        """
        Membuat dataset merger.
        
        Args:
            data_dir: Direktori dataset
            output_dir: Direktori output
            logger: Custom logger
            
        Returns:
            Instance DatasetMergerOperation
        """
        DatasetMergeOperation = DatasetComponentFactory.get_dataset_merge_operation()
        
        return DatasetMergeOperation(
            data_dir=data_dir,
            output_dir=output_dir,
            logger=logger or get_logger("dataset_merger")
        )
    
    @staticmethod
    def create_explorer_facade(
        config: Dict,
        data_dir: Optional[str] = None,
        logger: Optional[SmartCashLogger] = None
    ):
        """
        Membuat dataset explorer facade.
        
        Args:
            config: Konfigurasi
            data_dir: Direktori dataset
            logger: Custom logger
            
        Returns:
            Instance DatasetExplorerFacade
        """
        DatasetExplorerFacade = DatasetComponentFactory.get_dataset_explorer_facade()
        
        return DatasetExplorerFacade(
            config=config,
            data_dir=data_dir or config.get('data_dir', 'data'),
            logger=logger or get_logger("dataset_explorer")
        )
    
    @staticmethod
    def create_visualization_facade(
        config: Dict,
        data_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        logger: Optional[SmartCashLogger] = None
    ):
        """
        Membuat visualization facade.
        
        Args:
            config: Konfigurasi
            data_dir: Direktori dataset
            output_dir: Direktori output
            logger: Custom logger
            
        Returns:
            Instance VisualizationFacade
        """
        VisualizationFacade = DatasetComponentFactory.get_visualization_facade()
        
        return VisualizationFacade(
            config=config,
            data_dir=data_dir or config.get('data_dir', 'data'),
            output_dir=output_dir,
            logger=logger or get_logger("visualization_facade")
        )
    
    @staticmethod
    def create_cache_manager(
        config: Dict,
        cache_dir: Optional[str] = None,
        max_size_gb: float = 1.0,
        logger: Optional[SmartCashLogger] = None
    ) -> CacheManager:
        """
        Membuat cache manager.
        
        Args:
            config: Konfigurasi
            cache_dir: Direktori cache
            max_size_gb: Ukuran maksimum cache dalam GB
            logger: Custom logger
            
        Returns:
            Instance CacheManager
        """
        cache_config = config.get('data', {}).get('preprocessing', {}).get('cache', {})
        
        return CacheManager(
            cache_dir=cache_dir or config.get('data', {}).get('preprocessing', {}).get('cache_dir', '.cache/smartcash'),
            max_size_gb=cache_config.get('max_size_gb', max_size_gb),
            ttl_hours=cache_config.get('ttl_hours', 24),
            auto_cleanup=cache_config.get('auto_cleanup', True),
            cleanup_interval_mins=cache_config.get('cleanup_interval_mins', 30),
            logger=logger or get_logger("cache_manager")
        )