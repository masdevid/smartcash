# File: smartcash/handlers/dataset/facades/data_loading_facade.py
# Author: Alfrida Sabar
# Deskripsi: Facade untuk operasi loading data dan dataloader

import torch
from typing import Dict, Optional, Any, List

from smartcash.handlers.dataset.facades.dataset_base_facade import DatasetBaseFacade
from smartcash.handlers.dataset.core.dataset_loader import DatasetLoader
from smartcash.handlers.dataset.core.dataset_downloader import DatasetDownloader
from smartcash.handlers.dataset.core.dataset_transformer import DatasetTransformer


class DataLoadingFacade(DatasetBaseFacade):
    """Facade untuk loading dataset dan pembuatan dataloader."""
    
    @property
    def loader(self) -> DatasetLoader:
        """Lazy load loader."""
        return self._get_component('loader', lambda: DatasetLoader(
            config=self.config,
            data_dir=str(self.data_dir),
            cache_dir=str(self.cache_dir),
            logger=self.logger
        ))
    
    @property
    def downloader(self) -> DatasetDownloader:
        """Lazy load downloader."""
        return self._get_component('downloader', lambda: DatasetDownloader(
            config=self.config,
            data_dir=str(self.data_dir),
            logger=self.logger
        ))
    
    @property
    def transformer(self) -> DatasetTransformer:
        """Lazy load transformer."""
        return self._get_component('transformer', lambda: DatasetTransformer(
            config=self.config,
            img_size=self.img_size,
            logger=self.logger
        ))
    
    # ===== Dataset Loader Methods =====
    
    def get_dataset(self, split: str, **kwargs):
        """Dapatkan dataset untuk split tertentu."""
        return self.loader.get_dataset(split, **kwargs)
    
    def get_dataloader(self, split: str, **kwargs) -> torch.utils.data.DataLoader:
        """Dapatkan dataloader untuk split tertentu."""
        return self.loader.get_dataloader(split, **kwargs)
    
    def get_all_dataloaders(self, **kwargs) -> Dict[str, torch.utils.data.DataLoader]:
        """Dapatkan semua dataloader sekaligus."""
        return self.loader.get_all_dataloaders(**kwargs)
    
    def get_train_loader(self, **kwargs) -> torch.utils.data.DataLoader:
        """Shortcut untuk get_dataloader('train')."""
        return self.loader.get_train_loader(**kwargs)
    
    def get_val_loader(self, **kwargs) -> torch.utils.data.DataLoader:
        """Shortcut untuk get_dataloader('val')."""
        return self.loader.get_val_loader(**kwargs)
    
    def get_test_loader(self, **kwargs) -> torch.utils.data.DataLoader:
        """Shortcut untuk get_dataloader('test')."""
        return self.loader.get_test_loader(**kwargs)
    
    # ===== Dataset Downloader Methods =====
    
    def download_dataset(self, **kwargs) -> str:
        """Download dataset dari Roboflow."""
        return self.downloader.download_dataset(**kwargs)
    
    def export_to_local(self, roboflow_dir: str, **kwargs) -> tuple:
        """Export dataset Roboflow ke struktur folder lokal."""
        return self.downloader.export_to_local(roboflow_dir, **kwargs)
    
    def pull_dataset(self, **kwargs):
        """Download dan setup dataset dalam satu langkah."""
        return self.downloader.pull_dataset(**kwargs)
    
    def get_dataset_info(self) -> Dict:
        """Dapatkan informasi dataset."""
        return self.downloader.get_dataset_info()
    
    # ===== Transformer Methods =====
    
    def get_transform(self, mode: str = 'train') -> Any:
        """Dapatkan transformasi untuk mode tertentu."""
        return self.transformer.get_transform(mode)
    
    def create_custom_transform(self, **kwargs) -> Any:
        """Buat transformasi kustom."""
        return self.transformer.create_custom_train_transform(**kwargs)