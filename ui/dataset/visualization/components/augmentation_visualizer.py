"""
File: smartcash/ui/dataset/visualization/components/augmentation_visualizer.py
Deskripsi: Komponen untuk memvisualisasikan hasil augmentasi data
"""

import os
import random
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import ipywidgets as widgets
from IPython.display import display, clear_output
import cv2

from smartcash.common.logger import get_logger
from smartcash.utils.augmentation.augmentation_manager import AugmentationManager
from smartcash.ui.dataset.visualization.utils import create_augmentation_comparison

logger = get_logger(__name__)

class AugmentationVisualizer:
    """Komponen untuk memvisualisasikan hasil augmentasi data"""
    
    def __init__(self, dataset_path: str, config: Optional[Dict[str, Any]] = None):
        """Inisialisasi visualizer augmentasi
        
        Args:
            dataset_path: Path ke dataset
            config: Konfigurasi visualizer
        """
        self.dataset_path = dataset_path
        self.config = config or {}
        self.aug_manager = AugmentationManager(config_path=None)
        self.current_image = None
        self.current_image_path = None
        self.augmentation_results = []
        self.ui_components = {}
        
        # Default augmentation config
        self.aug_config = {
            'position': True,
            'lighting': True,
            'combined': True,
            'extreme_rotation': False
        }
    
    def load_random_image(self, split: str = 'train') -> bool:
        """Muat gambar acak dari dataset
        
        Args:
            split: Split dataset (train/val/test)
            
        Returns:
            bool: True jika berhasil, False jika gagal
        """
        try:
            # Cari direktori gambar
            img_dir = os.path.join(self.dataset_path, 'images', split)
            if not os.path.exists(img_dir):
                logger.error(f"Direktori gambar tidak ditemukan: {img_dir}")
                return False
                
            # Dapatkan daftar file gambar
            img_files = [f for f in os.listdir(img_dir) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            if not img_files:
                logger.error(f"Tidak ada gambar yang ditemukan di {img_dir}")
                return False
                
            # Pilih gambar acak
            selected_img = random.choice(img_files)
            self.current_image_path = os.path.join(img_dir, selected_img)
            
            # Baca gambar
            self.current_image = cv2.imread(self.current_image_path)
            if self.current_image is None:
                logger.error(f"Gagal membaca gambar: {self.current_image_path}")
                return False
                
            # Konversi ke RGB
            self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
            
            # Reset hasil augmentasi
            self.augmentation_results = []
            
            return True
            
        except Exception as e:
            logger.error(f"Gagal memuat gambar acak: {e}")
            return False
    
    def apply_augmentations(self) -> bool:
        """Terapkan augmentasi pada gambar saat ini
        
        Returns:
            bool: True jika berhasil, False jika gagal
        """
        if self.current_image is None or self.current_image_path is None:
            logger.warning("Tidak ada gambar yang dimuat")
            return False
            
        try:
            # Reset hasil sebelumnya
            self.augmentation_results = []
            
            # Simpan gambar asli
            self.augmentation_results.append({
                'name': 'Original',
                'image': self.current_image
            })
            
            # Terapkan setiap jenis augmentasi yang diaktifkan
            if self.aug_config.get('position', True):
                augmented = self.aug_manager.apply_augmentation(
                    self.current_image_path, 
                    augmentation_type='position'
                )
                if augmented is not None:
                    self.augmentation_results.append({
                        'name': 'Position Augmentation',
                        'image': cv2.cvtColor(augmented, cv2.COLOR_BGR2RGB)
                    })
            
            if self.aug_config.get('lighting', True):
                augmented = self.aug_manager.apply_augmentation(
                    self.current_image_path,
                    augmentation_type='lighting'
                )
                if augmented is not None:
                    self.augmentation_results.append({
                        'name': 'Lighting Augmentation',
                        'image': cv2.cvtColor(augmented, cv2.COLOR_BGR2RGB)
                    })
            
            if self.aug_config.get('combined', True):
                augmented = self.aug_manager.apply_augmentation(
                    self.current_image_path,
                    augmentation_type='combined'
                )
                if augmented is not None:
                    self.augmentation_results.append({
                        'name': 'Combined Augmentation',
                        'image': cv2.cvtColor(augmented, cv2.COLOR_BGR2RGB)
                    })
            
            if self.aug_config.get('extreme_rotation', False):
                augmented = self.aug_manager.apply_augmentation(
                    self.current_image_path,
                    augmentation_type='extreme_rotation'
                )
                if augmented is not None:
                    self.augmentation_results.append({
                        'name': 'Extreme Rotation',
                        'image': cv2.cvtColor(augmented, cv2.COLOR_BGR2RGB)
                    })
            
            return True
            
        except Exception as e:
            logger.error(f"Gagal menerapkan augmentasi: {e}")
            return False
    
    def _create_controls(self) -> widgets.Widget:
        """Buat kontrol UI untuk visualizer"""
        # Tombol untuk memuat gambar acak
        self.load_btn = widgets.Button(
            description='Muat Gambar Acak',
            button_style='primary',
            icon='refresh'
        )
        self.load_btn.on_click(self._on_load_clicked)
        
        # Pilihan split dataset
        self.split_dropdown = widgets.Dropdown(
            options=['train', 'val', 'test'],
            value='train',
            description='Split:',
            disabled=False
        )
        
        # Checkbox untuk memilih jenis augmentasi
        self.aug_checkboxes = {}
        for aug_type in ['position', 'lighting', 'combined', 'extreme_rotation']:
            self.aug_checkboxes[aug_type] = widgets.Checkbox(
                value=self.aug_config.get(aug_type, aug_type != 'extreme_rotation'),
                description=aug_type.replace('_', ' ').title(),
                disabled=False
            )
            
        # Tombol untuk menerapkan augmentasi
        self.apply_btn = widgets.Button(
            description='Terapkan Augmentasi',
            button_style='info',
            icon='magic'
        )
        self.apply_btn.on_click(self._on_apply_clicked)
        
        # Kelompokkan kontrol
        aug_controls = [
            widgets.HTML('<b>Jenis Augmentasi:</b>'),
            widgets.HBox([v for v in self.aug_checkboxes.values()])
        ]
        
        controls = widgets.VBox([
            widgets.HBox([self.load_btn, self.split_dropdown]),
            *aug_controls,
            self.apply_btn
        ], layout=widgets.Layout(border='1px solid #e0e0e0', padding='10px'))
        
        return controls
    
    def _create_display_area(self) -> widgets.Widget:
        """Buat area tampilan untuk gambar dan hasil augmentasi"""
        self.output_area = widgets.Output()
        return self.output_area
    
    def _create_ui(self) -> widgets.Widget:
        """Buat UI lengkap"""
        controls = self._create_controls()
        display_area = self._create_display_area()
        
        # Buat tab untuk tampilan yang berbeda
        tabs = widgets.Tab()
        tabs.children = [
            widgets.VBox([
                widgets.HTML('<h3>Perbandingan Augmentasi</h3>'),
                display_area
            ])
        ]
        tabs.set_title(0, 'Visualisasi')
        
        # Container utama
        main_container = widgets.VBox([
            widgets.HTML('<h2>Visualisasi Augmentasi Data</h2>'),
            controls,
            tabs
        ])
        
        return main_container
    
    def _on_load_clicked(self, btn) -> None:
        """Handler untuk tombol muat gambar"""
        with self.output_area:
            clear_output(wait=True)
            print("Memuat gambar acak...")
            
            if self.load_random_image(split=self.split_dropdown.value):
                self._display_current_image()
            else:
                print("Gagal memuat gambar. Silakan coba lagi.")
    
    def _on_apply_clicked(self, btn) -> None:
        """Handler untuk tombol terapkan augmentasi"""
        if self.current_image is None:
            with self.output_area:
                print("Tidak ada gambar yang dimuat. Silakan muat gambar terlebih dahulu.")
            return
            
        # Perbarui konfigurasi augmentasi
        for aug_type, checkbox in self.aug_checkboxes.items():
            self.aug_config[aug_type] = checkbox.value
        
        with self.output_area:
            clear_output(wait=True)
            print("Menerapkan augmentasi...")
            
            if self.apply_augmentations():
                self._display_augmentation_results()
            else:
                print("Gagal menerapkan augmentasi. Silakan coba lagi.")
    
    def _display_current_image(self) -> None:
        """Tampilkan gambar saat ini"""
        if self.current_image is None:
            return
            
        with self.output_area:
            clear_output()
            
            # Tampilkan informasi gambar
            height, width = self.current_image.shape[:2]
            info_html = f"""
            <div style='margin: 10px 0; padding: 10px; background: #f5f5f5; border-radius: 5px;'>
                <b>Informasi Gambar:</b><br>
                - Dimensi: {width} x {height} (lebar x tinggi)<br>
                - Format: {os.path.splitext(self.current_image_path or '')[1]}<br>
                - Path: {self.current_image_path}
            </div>
            """
            display(widgets.HTML(info_html))
            
            # Tampilkan gambar
            display(widgets.Image(
                value=cv2.imencode('.jpg', self.current_image)[1].tobytes(),
                format='jpg',
                width=min(800, width)
            ))
    
    def _display_augmentation_results(self) -> None:
        """Tampilkan hasil augmentasi"""
        if not self.augmentation_results:
            with self.output_area:
                print("Tidak ada hasil augmentasi yang tersedia.")
            return
            
        with self.output_area:
            clear_output()
            
            # Tampilkan gambar asli dan hasil augmentasi
            fig = create_augmentation_comparison(
                self.current_image,
                [r for r in self.augmentation_results if r['name'] != 'Original']
            )
            
            if fig is not None:
                display(fig)
            else:
                print("Gagal membuat visualisasi hasil augmentasi.")
    
    def display(self) -> None:
        """Tampilkan visualizer"""
        if not hasattr(self, 'main_container') or self.main_container is None:
            self.main_container = self._create_ui()
        display(self.main_container)
    
    def get_ui_components(self) -> Dict[str, Any]:
        """Dapatkan komponen UI
        
        Returns:
            Dict berisi komponen UI
        """
        if not hasattr(self, 'main_container') or self.main_container is None:
            self.main_container = self._create_ui()
            
        return {
            'main_container': self.main_container,
            'controls': self._create_controls(),
            'output_area': self._create_display_area()
        }
