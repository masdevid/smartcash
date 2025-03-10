# File: smartcash/handlers/dataset/__init__.py
# Deskripsi: Package untuk pengelolaan dataset multilayer SmartCash (versi terstruktur)

# Facade utama dan manager
from smartcash.handlers.dataset.dataset_manager import DatasetManager

# Facade-facade berbasis fungsi
from smartcash.handlers.dataset.facades.data_loading_facade import DataLoadingFacade
from smartcash.handlers.dataset.facades.data_processing_facade import DataProcessingFacade
from smartcash.handlers.dataset.facades.data_operations_facade import DataOperationsFacade
from smartcash.handlers.dataset.facades.visualization_facade import VisualizationFacade
from smartcash.handlers.dataset.facades.pipeline_facade import PipelineFacade

# Komponen dataset
from smartcash.handlers.dataset.multilayer.multilayer_dataset import MultilayerDataset
from smartcash.handlers.dataset.multilayer.multilayer_label_handler import MultilayerLabelHandler

# Collate functions
from smartcash.handlers.dataset.collate_functions import multilayer_collate_fn, flat_collate_fn

# Export semua komponen publik
__all__ = [
    # Manager utama
    'DatasetManager',
    
    # Facades berbasis fungsi
    'DataLoadingFacade',
    'DataProcessingFacade',
    'DataOperationsFacade',
    'VisualizationFacade',
    'PipelineFacade',
    
    # Komponen dataset
    'MultilayerDataset',
    'MultilayerLabelHandler',
    
    # Collate functions
    'multilayer_collate_fn',
    'flat_collate_fn',
]