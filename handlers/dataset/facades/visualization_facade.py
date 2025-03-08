# File: smartcash/handlers/dataset/facades/visualization_facade.py
# Author: Alfrida Sabar
# Deskripsi: Facade khusus untuk operasi visualisasi dataset

from typing import Dict, List, Optional, Any, Tuple

from smartcash.handlers.dataset.facades.dataset_base_facade import DatasetBaseFacade
from smartcash.handlers.dataset.visualizations.distribution_visualizer import DistributionVisualizer
from smartcash.handlers.dataset.visualizations.sample_visualizer import SampleVisualizer


class VisualizationFacade(DatasetBaseFacade):
    """
    Facade yang menyediakan akses ke operasi visualisasi dataset.
    """
    
    @property
    def distribution_visualizer(self) -> DistributionVisualizer:
        """Akses ke komponen distribution_visualizer dengan lazy initialization."""
        return self._get_component('distribution_visualizer', lambda: DistributionVisualizer(
            data_dir=str(self.data_dir),
            logger=self.logger
        ))
    
    @property
    def sample_visualizer(self) -> SampleVisualizer:
        """Akses ke komponen sample_visualizer dengan lazy initialization."""
        return self._get_component('sample_visualizer', lambda: SampleVisualizer(
            data_dir=str(self.data_dir),
            logger=self.logger
        ))
    
    # ===== Metode dari DistributionVisualizer =====
    
    def visualize_class_distribution(
        self, 
        split: str = 'train', 
        save_path: Optional[str] = None,
        top_n: int = 10,
        figsize: Tuple[int, int] = (12, 8)
    ) -> str:
        """
        Visualisasikan distribusi kelas dalam dataset.
        
        Args:
            split: Split dataset ('train', 'valid', 'test')
            save_path: Path untuk menyimpan visualisasi (opsional)
            top_n: Jumlah kelas teratas untuk ditampilkan
            figsize: Ukuran figur
        
        Returns:
            Path ke file visualisasi yang disimpan
        """
        return self.distribution_visualizer.visualize_class_distribution(
            split=split, 
            save_path=save_path, 
            top_n=top_n, 
            figsize=figsize
        )
    
    def visualize_layer_distribution(
        self, 
        split: str = 'train', 
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 6)
    ) -> str:
        """
        Visualisasikan distribusi layer dalam dataset.
        
        Args:
            split: Split dataset ('train', 'valid', 'test')
            save_path: Path untuk menyimpan visualisasi (opsional)
            figsize: Ukuran figur
        
        Returns:
            Path ke file visualisasi yang disimpan
        """
        return self.distribution_visualizer.visualize_layer_distribution(
            split=split, 
            save_path=save_path, 
            figsize=figsize
        )
    
    # ===== Metode dari SampleVisualizer =====
    
    def visualize_sample_images(
        self,
        split: str = 'train',
        num_samples: int = 9,
        classes: Optional[List[str]] = None,
        random_seed: int = 42,
        figsize: Tuple[int, int] = (15, 15),
        save_path: Optional[str] = None
    ) -> str:
        """
        Visualisasikan sampel gambar dari dataset dengan bounding box.
        
        Args:
            split: Split dataset ('train', 'val', 'test')
            num_samples: Jumlah sampel yang akan ditampilkan
            classes: Filter kelas tertentu (opsional)
            random_seed: Seed untuk random state
            figsize: Ukuran figur
            save_path: Path untuk menyimpan visualisasi (opsional)
            
        Returns:
            Path ke file visualisasi yang disimpan
        """
        return self.sample_visualizer.visualize_sample_images(
            split=split,
            num_samples=num_samples,
            classes=classes,
            random_seed=random_seed,
            figsize=figsize,
            save_path=save_path
        )
    
    def visualize_augmentation_comparison(
        self,
        image_path: str,
        augmentation_types: List[str] = ['original', 'lighting', 'geometric', 'combined'],
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (15, 10)
    ) -> str:
        """
        Visualisasikan perbandingan berbagai jenis augmentasi pada gambar.
        
        Args:
            image_path: Path ke gambar yang akan diaugmentasi
            augmentation_types: Jenis augmentasi yang akan divisualisasikan
            save_path: Path untuk menyimpan visualisasi (opsional)
            figsize: Ukuran figur
            
        Returns:
            Path ke file visualisasi yang disimpan
        """
        return self.sample_visualizer.visualize_augmentation_comparison(
            image_path=image_path,
            augmentation_types=augmentation_types,
            save_path=save_path,
            figsize=figsize
        )