# File: smartcash/handlers/dataset/facades/visualization_facade.py
# Author: Alfrida Sabar
# Deskripsi: Facade untuk visualisasi dataset menggunakan utils/visualization

from typing import Dict, List, Optional, Any, Tuple

from smartcash.utils.observer import EventTopics, notify
from smartcash.handlers.dataset.facades.dataset_base_facade import DatasetBaseFacade
from smartcash.utils.visualization.data import DataVisualizationHelper


class VisualizationFacade(DatasetBaseFacade):
    """Facade untuk operasi visualisasi dataset menggunakan utils/visualization."""
    
    @property
    def visualizer(self) -> DataVisualizationHelper:
        """Akses ke visualizer."""
        return self._get_component('visualizer', lambda: DataVisualizationHelper(
            output_dir=str(self.data_dir / 'visualizations'),
            logger=self.logger
        ))
    
    def _visualize_with_notify(self, type_name, split, visualize_func, **kwargs):
        """Helper untuk visualisasi dengan notifikasi."""
        notify(EventTopics.UI_UPDATE, self, action="visualize", type=type_name, split=split)
        result = visualize_func(**kwargs)
        notify(EventTopics.UI_UPDATE, self, action="visualize_complete", type=type_name, path=result)
        return result
    
    def visualize_class_distribution(self, split: str = 'train', save_path: Optional[str] = None,
                               top_n: int = 10, figsize: Tuple[int, int] = (12, 8)) -> str:
        """Visualisasikan distribusi kelas."""
        # Dapatkan statistik kelas
        from smartcash.handlers.dataset.facades.dataset_explorer_facade import DatasetExplorerFacade
        explorer = DatasetExplorerFacade(self.config, str(self.data_dir), self.logger)
        class_stats = explorer.get_class_statistics(split)
        
        return self._visualize_with_notify(
            'class_distribution', 
            split,
            self.visualizer.plot_class_distribution,
            class_stats=class_stats,
            title=f"Distribusi Kelas - Split {split}",
            save_path=save_path,
            top_n=top_n,
            figsize=figsize
        )
    
    def visualize_layer_distribution(self, split: str = 'train', save_path: Optional[str] = None,
                              figsize: Tuple[int, int] = (10, 6)) -> str:
        """Visualisasikan distribusi layer."""
        # Dapatkan statistik layer
        from smartcash.handlers.dataset.facades.dataset_explorer_facade import DatasetExplorerFacade
        explorer = DatasetExplorerFacade(self.config, str(self.data_dir), self.logger)
        layer_stats = explorer.get_layer_statistics(split)
        
        return self._visualize_with_notify(
            'layer_distribution', 
            split,
            self.visualizer.plot_layer_distribution,
            layer_stats=layer_stats,
            title=f"Distribusi Layer - Split {split}",
            save_path=save_path,
            figsize=figsize
        )
    
    def visualize_sample_images(self, split: str = 'train', num_samples: int = 9,
                          classes: Optional[List[str]] = None, random_seed: int = 42,
                          figsize: Tuple[int, int] = (15, 15), save_path: Optional[str] = None) -> str:
        """Visualisasikan sampel gambar dengan bounding box."""
        return self._visualize_with_notify(
            'sample_images', 
            split,
            self.visualizer.plot_sample_images,
            data_dir=str(self.data_dir / split),
            num_samples=num_samples,
            classes=classes,
            random_seed=random_seed,
            figsize=figsize,
            save_path=save_path
        )
        
    def visualize_augmentation_comparison(self, image_path: str,
                                   augmentation_types: List[str] = ['original', 'lighting', 'geometric', 'combined'],
                                   save_path: Optional[str] = None, figsize: Tuple[int, int] = (15, 10)) -> str:
        """Visualisasikan perbandingan berbagai jenis augmentasi pada gambar."""
        return self._visualize_with_notify(
            'augmentation_comparison', 
            None,
            self.visualizer.plot_augmentation_comparison,
            image_path=image_path,
            augmentation_types=augmentation_types,
            save_path=save_path,
            figsize=figsize
        )