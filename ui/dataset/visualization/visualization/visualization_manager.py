"""
File: smartcash/ui/dataset/augmentation/visualization/visualization_manager.py
Deskripsi: Manager untuk visualisasi augmentasi dataset
"""

import os
import threading
from typing import Dict, List, Any, Optional, Callable
from IPython.display import display, clear_output
import ipywidgets as widgets

from smartcash.common.logger import get_logger
from smartcash.common.config.manager import get_config_manager
from smartcash.ui.dataset.augmentation.visualization.components.visualization_components import AugmentationVisualizationComponents
from smartcash.ui.dataset.augmentation.visualization.handlers.sample_visualization_handler import SampleVisualizationHandler
from smartcash.ui.dataset.augmentation.visualization.handlers.compare_visualization_handler import CompareVisualizationHandler


class AugmentationVisualizationManager:
    """Manager untuk visualisasi augmentasi dataset dengan pendekatan singleton"""
    
    _instance = None
    _lock = threading.Lock()
    
    @classmethod
    def get_instance(cls, config: Dict = None, logger=None):
        """
        Dapatkan instance singleton AugmentationVisualizationManager.
        
        Args:
            config: Konfigurasi aplikasi
            logger: Logger kustom (opsional)
            
        Returns:
            Instance AugmentationVisualizationManager
        """
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(config, logger)
            return cls._instance
    
    def __init__(self, config: Dict = None, logger=None):
        """
        Inisialisasi AugmentationVisualizationManager.
        
        Args:
            config: Konfigurasi aplikasi
            logger: Logger kustom (opsional)
        """
        self.logger = logger or get_logger("augmentation_visualization_manager")
        
        # Dapatkan konfigurasi
        self.config_manager = get_config_manager()
        self.config = config or self.config_manager.get_module_config('augmentation')
        
        # Inisialisasi komponen dan handler
        self.ui_components = AugmentationVisualizationComponents(self.config, self.logger)
        self.sample_handler = SampleVisualizationHandler(self.config, self.logger)
        self.compare_handler = CompareVisualizationHandler(self.config, self.logger)
        
        # Register handler untuk tombol
        self._register_handlers()
        
        self.logger.info("🎨 AugmentationVisualizationManager siap digunakan")
        
    def _register_handlers(self):
        """Register handler untuk tombol visualisasi."""
        self.ui_components.register_handlers(
            on_visualize_samples=self._on_visualize_samples,
            on_visualize_variations=self._on_visualize_variations,
            on_visualize_compare=self._on_visualize_compare,
            on_visualize_impact=self._on_visualize_impact
        )
        
    def _on_visualize_samples(self, button):
        """
        Handler untuk tombol visualisasi sampel.
        
        Args:
            button: Tombol yang diklik
        """
        # Dapatkan parameter dari UI
        aug_type = self.ui_components.aug_type_dropdown.value
        split = self.ui_components.split_dropdown.value
        num_samples = self.ui_components.sample_count_slider.value
        data_dir = self.ui_components.data_dir_text.value
        
        # Tampilkan status
        self.ui_components.show_status(f"Memvisualisasikan {num_samples} sampel augmentasi '{aug_type}' dari split {split}...", 'info')
        
        # Jalankan visualisasi di thread terpisah
        def run_visualization():
            try:
                # Visualisasikan sampel
                result = self.sample_handler.visualize_augmentation_samples(
                    data_dir=data_dir,
                    aug_types=[aug_type],
                    split=split,
                    num_samples=num_samples
                )
                
                # Tampilkan hasil
                if result['status'] == 'success':
                    # Tampilkan figure
                    fig = result['results'][0]['figure']
                    self.ui_components.show_figure(fig, self.ui_components.sample_output)
                    
                    # Tampilkan status sukses
                    self.ui_components.show_status(f"Berhasil memvisualisasikan {num_samples} sampel augmentasi '{aug_type}'", 'success')
                else:
                    # Tampilkan status error
                    self.ui_components.show_status(f"Gagal memvisualisasikan sampel: {result['message']}", 'error')
            except Exception as e:
                # Tampilkan status error
                self.ui_components.show_status(f"Error saat memvisualisasikan sampel: {str(e)}", 'error')
                self.logger.error(f"Error saat memvisualisasikan sampel: {str(e)}")
        
        # Jalankan di thread terpisah
        threading.Thread(target=run_visualization).start()
        
    def _on_visualize_variations(self, button):
        """
        Handler untuk tombol visualisasi variasi.
        
        Args:
            button: Tombol yang diklik
        """
        # Dapatkan parameter dari UI
        aug_type = self.ui_components.aug_type_dropdown.value
        split = self.ui_components.split_dropdown.value
        data_dir = self.ui_components.data_dir_text.value
        
        # Tampilkan status
        self.ui_components.show_status(f"Memvisualisasikan variasi augmentasi '{aug_type}' dari split {split}...", 'info')
        
        # Jalankan visualisasi di thread terpisah
        def run_visualization():
            try:
                # Visualisasikan variasi
                result = self.sample_handler.visualize_augmentation_variations(
                    data_dir=data_dir,
                    aug_type=aug_type,
                    split=split
                )
                
                # Tampilkan hasil
                if result['status'] == 'success':
                    # Tampilkan figure
                    fig = result['figure']
                    self.ui_components.show_figure(fig, self.ui_components.sample_output)
                    
                    # Tampilkan status sukses
                    self.ui_components.show_status(f"Berhasil memvisualisasikan variasi augmentasi '{aug_type}'", 'success')
                else:
                    # Tampilkan status error
                    self.ui_components.show_status(f"Gagal memvisualisasikan variasi: {result['message']}", 'error')
            except Exception as e:
                # Tampilkan status error
                self.ui_components.show_status(f"Error saat memvisualisasikan variasi: {str(e)}", 'error')
                self.logger.error(f"Error saat memvisualisasikan variasi: {str(e)}")
        
        # Jalankan di thread terpisah
        threading.Thread(target=run_visualization).start()
        
    def _on_visualize_compare(self, button):
        """
        Handler untuk tombol visualisasi perbandingan.
        
        Args:
            button: Tombol yang diklik
        """
        # Dapatkan parameter dari UI
        aug_type = self.ui_components.aug_type_dropdown.value
        split = self.ui_components.split_dropdown.value
        num_samples = self.ui_components.sample_count_slider.value
        data_dir = self.ui_components.data_dir_text.value
        preprocessed_dir = self.ui_components.preprocessed_dir_text.value
        
        # Tampilkan status
        self.ui_components.show_status(f"Memvisualisasikan perbandingan preprocess vs augmentasi '{aug_type}' dari split {split}...", 'info')
        
        # Jalankan visualisasi di thread terpisah
        def run_visualization():
            try:
                # Visualisasikan perbandingan
                result = self.compare_handler.visualize_preprocess_vs_augmentation(
                    data_dir=data_dir,
                    preprocessed_dir=preprocessed_dir,
                    aug_type=aug_type,
                    split=split,
                    num_samples=num_samples
                )
                
                # Tampilkan hasil
                if result['status'] == 'success':
                    # Tampilkan figure
                    fig = result['figure']
                    self.ui_components.show_figure(fig, self.ui_components.compare_output)
                    
                    # Tampilkan status sukses
                    self.ui_components.show_status(f"Berhasil memvisualisasikan perbandingan preprocess vs augmentasi '{aug_type}'", 'success')
                else:
                    # Tampilkan status error
                    self.ui_components.show_status(f"Gagal memvisualisasikan perbandingan: {result['message']}", 'error')
            except Exception as e:
                # Tampilkan status error
                self.ui_components.show_status(f"Error saat memvisualisasikan perbandingan: {str(e)}", 'error')
                self.logger.error(f"Error saat memvisualisasikan perbandingan: {str(e)}")
        
        # Jalankan di thread terpisah
        threading.Thread(target=run_visualization).start()
        
    def _on_visualize_impact(self, button):
        """
        Handler untuk tombol visualisasi dampak.
        
        Args:
            button: Tombol yang diklik
        """
        # Dapatkan parameter dari UI
        split = self.ui_components.split_dropdown.value
        data_dir = self.ui_components.data_dir_text.value
        preprocessed_dir = self.ui_components.preprocessed_dir_text.value
        
        # Tampilkan status
        self.ui_components.show_status(f"Memvisualisasikan dampak berbagai jenis augmentasi dari split {split}...", 'info')
        
        # Jalankan visualisasi di thread terpisah
        def run_visualization():
            try:
                # Visualisasikan dampak
                result = self.compare_handler.visualize_augmentation_impact(
                    data_dir=data_dir,
                    preprocessed_dir=preprocessed_dir,
                    aug_types=['combined', 'position', 'lighting'],
                    split=split
                )
                
                # Tampilkan hasil
                if result['status'] == 'success':
                    # Tampilkan figure
                    fig = result['figure']
                    self.ui_components.show_figure(fig, self.ui_components.compare_output)
                    
                    # Tampilkan status sukses
                    self.ui_components.show_status(f"Berhasil memvisualisasikan dampak berbagai jenis augmentasi", 'success')
                else:
                    # Tampilkan status error
                    self.ui_components.show_status(f"Gagal memvisualisasikan dampak: {result['message']}", 'error')
            except Exception as e:
                # Tampilkan status error
                self.ui_components.show_status(f"Error saat memvisualisasikan dampak: {str(e)}", 'error')
                self.logger.error(f"Error saat memvisualisasikan dampak: {str(e)}")
        
        # Jalankan di thread terpisah
        threading.Thread(target=run_visualization).start()
        
    def get_visualization_ui(self) -> widgets.VBox:
        """
        Dapatkan UI visualisasi augmentasi.
        
        Returns:
            Widget VBox berisi UI visualisasi
        """
        return self.ui_components.create_visualization_ui()
        
    def display_visualization_ui(self) -> None:
        """Tampilkan UI visualisasi augmentasi."""
        ui = self.get_visualization_ui()
        display(ui)
