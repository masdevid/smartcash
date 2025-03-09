# File: smartcash/handlers/dataset/facades/data_operations_facade.py
# Author: Alfrida Sabar
# Deskripsi: Facade khusus untuk operasi manipulasi dataset seperti split, merge, dan visualisasi

from typing import Dict, List, Optional, Any, Tuple

from smartcash.handlers.dataset.facades.dataset_base_facade import DatasetBaseFacade
from smartcash.handlers.dataset.operations.dataset_split_operation import DatasetSplitOperation
from smartcash.handlers.dataset.operations.dataset_merge_operation import DatasetMergeOperation
from smartcash.handlers.dataset.operations.dataset_reporting_operation import DatasetReportingOperation
from smartcash.handlers.dataset.explorers.dataset_explorer_facade import DatasetExplorerFacade


class DataOperationsFacade(DatasetBaseFacade):
    """
    Facade yang menyediakan akses ke operasi manipulasi dataset seperti 
    split, merge, dan pembuatan laporan.
    """
    
    @property
    def explorer(self) -> DatasetExplorerFacade:
        """Akses ke komponen explorer dengan lazy initialization."""
        return self._get_component('explorer', lambda: DatasetExplorerFacade(
            config=self.config,
            data_dir=str(self.data_dir),
            logger=self.logger
        ))
    
    @property
    def split_operation(self) -> DatasetSplitOperation:
        """Akses ke komponen split_operation dengan lazy initialization."""
        return self._get_component('split_operation', lambda: DatasetSplitOperation(
            data_dir=str(self.data_dir),
            logger=self.logger
        ))
    
    @property
    def merge_operation(self) -> DatasetMergeOperation:
        """Akses ke komponen merge_operation dengan lazy initialization."""
        return self._get_component('merge_operation', lambda: DatasetMergeOperation(
            data_dir=str(self.data_dir),
            logger=self.logger
        ))
    
    @property
    def reporting_operation(self) -> DatasetReportingOperation:
        """Akses ke komponen reporting_operation dengan lazy initialization."""
        return self._get_component('reporting_operation', lambda: DatasetReportingOperation(
            config=self.config,
            data_dir=str(self.data_dir),
            logger=self.logger
        ))
    
    # ===== Metode dari DatasetExplorerFacade =====
    
    def get_split_statistics(self) -> Dict[str, Dict[str, int]]:
        """
        Dapatkan statistik untuk semua split dataset.
        
        Returns:
            Dict berisi jumlah file di setiap split
        """
        return self.explorer.get_split_statistics()
    
    def get_layer_statistics(self, split: str) -> Dict[str, Any]:
        """
        Dapatkan statistik layer untuk split dataset tertentu.
        
        Args:
            split: Split dataset ('train', 'valid', 'test')
            
        Returns:
            Dict berisi statistik layer
        """
        return self.explorer.get_layer_statistics(split)
    
    def get_class_statistics(self, split: str) -> Dict[str, Any]:
        """
        Dapatkan statistik kelas untuk split dataset tertentu.
        
        Args:
            split: Split dataset ('train', 'valid', 'test')
            
        Returns:
            Dict berisi statistik kelas
        """
        return self.explorer.get_class_statistics(split)
    
    # ===== Metode dari DatasetSplitOperation dan DatasetMergeOperation =====
    
    def split_dataset(self, **kwargs) -> Dict[str, int]:
        """
        Pecah dataset menjadi train/val/test.
        
        Args:
            **kwargs: Parameter untuk DatasetSplitOperation.split_dataset
            
        Returns:
            Dict berisi jumlah file di setiap split
        """
        return self.split_operation.split_dataset(**kwargs)
    
    def merge_splits(self, **kwargs) -> Dict[str, int]:
        """
        Gabungkan semua split menjadi satu direktori flat.
        
        Args:
            **kwargs: Parameter untuk DatasetMergeOperation.merge_splits
            
        Returns:
            Dict berisi jumlah file di direktori gabungan
        """
        return self.merge_operation.merge_splits(**kwargs)
    
    def merge_datasets(self, **kwargs) -> Dict[str, int]:
        """
        Gabungkan beberapa dataset terpisah menjadi satu dataset.
        
        Args:
            **kwargs: Parameter untuk DatasetMergeOperation.merge_datasets
            
        Returns:
            Dict berisi jumlah file di direktori gabungan
        """
        return self.merge_operation.merge_datasets(**kwargs)
    
    # ===== Metode dari DatasetReportingOperation =====
    
    def generate_dataset_report(self, splits: List[str] = ['train', 'valid', 'test'], visualize: bool = True) -> Dict[str, Any]:
        """
        Generate laporan lengkap tentang dataset.
        
        Args:
            splits: List split yang akan dianalisis
            visualize: Jika True, buat visualisasi
            
        Returns:
            Dict berisi informasi lengkap tentang dataset
        """
        return self.reporting_operation.generate_dataset_report(splits, visualize)