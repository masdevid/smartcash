# File: smartcash/handlers/dataset/facades/data_operations_facade.py
# Deskripsi: Facade untuk operasi manipulasi dataset seperti split, merge, dan pelaporan

from typing import Dict, List, Optional, Any

from smartcash.handlers.dataset.facades.dataset_base_facade import DatasetBaseFacade
from smartcash.handlers.dataset.dataset_utils_adapter import DatasetUtilsAdapter
from smartcash.handlers.dataset.operations.dataset_merge_operation import DatasetMergeOperation
from smartcash.handlers.dataset.operations.dataset_reporting_operation import DatasetReportingOperation
from smartcash.utils.dataset.dataset_utils import DEFAULT_SPLITS


class DataOperationsFacade(DatasetBaseFacade):
    """Facade untuk operasi manipulasi dataset."""
    
    @property
    def explorer(self) -> 'DatasetExplorerFacade':
        """Lazy load explorer."""
        from smartcash.handlers.dataset.facades.dataset_explorer_facade import DatasetExplorerFacade
        return self._get_component('explorer', lambda: DatasetExplorerFacade(
            config=self.config,
            data_dir=str(self.data_dir),
            logger=self.logger
        ))
    
    @property
    def utils_adapter(self) -> DatasetUtilsAdapter:
        """Lazy load utils_adapter."""
        return self._get_component('utils_adapter', lambda: DatasetUtilsAdapter(
            config=self.config,
            data_dir=str(self.data_dir),
            logger=self.logger
        ))
    
    @property
    def merge_operation(self) -> DatasetMergeOperation:
        """Lazy load merge_operation."""
        return self._get_component('merge_operation', lambda: DatasetMergeOperation(
            data_dir=str(self.data_dir),
            logger=self.logger
        ))
    
    @property
    def reporting_operation(self) -> DatasetReportingOperation:
        """Lazy load reporting_operation."""
        return self._get_component('reporting_operation', lambda: DatasetReportingOperation(
            config=self.config,
            data_dir=str(self.data_dir),
            logger=self.logger
        ))
    
    # ===== DatasetExplorerFacade Methods =====
    
    def get_split_statistics(self) -> Dict[str, Dict[str, int]]:
        """Dapatkan statistik untuk semua split dataset."""
        return self.explorer.get_split_statistics()
    
    def get_layer_statistics(self, split: str) -> Dict[str, Any]:
        """Dapatkan statistik layer untuk split dataset tertentu."""
        return self.explorer.get_layer_statistics(split)
    
    def get_class_statistics(self, split: str) -> Dict[str, Any]:
        """Dapatkan statistik kelas untuk split dataset tertentu."""
        return self.explorer.get_class_statistics(split)
    
    # ===== Dataset Operations Methods =====
    
    def split_dataset(self, **kwargs) -> Dict[str, int]:
        """Pecah dataset menjadi train/val/test."""
        return self.utils_adapter.split_dataset(**kwargs)
    
    def merge_splits(self, **kwargs) -> Dict[str, int]:
        """Gabungkan semua split menjadi direktori flat."""
        return self.merge_operation.merge_splits(**kwargs)
    
    def merge_datasets(self, **kwargs) -> Dict[str, int]:
        """Gabungkan beberapa dataset terpisah."""
        return self.merge_operation.merge_datasets(**kwargs)
    
    # ===== Reporting Methods =====
    
    def generate_dataset_report(self, splits: List[str] = DEFAULT_SPLITS, 
                          **kwargs) -> Dict[str, Any]:
        """Generate laporan lengkap tentang dataset."""
        return self.reporting_operation.generate_dataset_report(splits, **kwargs)