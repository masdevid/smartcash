"""
File: smartcash/ui/dataset/visualization/visualization_manager.py
Deskripsi: Manager untuk visualisasi dataset yang mengelola semua business logic
"""

from typing import Dict, Any, Optional
import ipywidgets as widgets
from IPython.display import display, clear_output
import threading
import os
from datetime import datetime

from smartcash.ui.utils.constants import ICONS
from smartcash.ui.utils.alert_utils import create_status_indicator
from smartcash.common.logger import get_logger
from smartcash.common.config import ConfigManager, get_config_manager
from smartcash.dataset.services.service_factory import get_dataset_service
from smartcash.ui.dataset.visualization.components.dataset_stats_cards import (
    create_dataset_stats_cards, create_preprocessing_stats_cards, create_augmentation_stats_cards
)
from smartcash.ui.dataset.visualization.components.visualization_tabs import create_visualization_tabs
from smartcash.ui.dataset.visualization.handlers.tab_handlers import setup_tab_handlers
from smartcash.ui.utils.loading_indicator import create_loading_indicator, LoadingIndicator

logger = get_logger(__name__)

class VisualizationManager:
    """Manager untuk visualisasi dataset yang mengelola semua business logic"""
    
    def __init__(self, loading_indicator: Optional[LoadingIndicator] = None):
        """
        Inisialisasi manager visualisasi dataset.
        
        Args:
            loading_indicator: Indikator loading opsional
        """
        self.loading_indicator = loading_indicator
        self.ui_components = {}
        self.config_manager = ConfigManager()
        self.dataset_path = None
        
        # Cek dan inisialisasi konfigurasi dataset
        self.initialize_config()
        
        # Inisialisasi dataset service setelah memastikan konfigurasi ada
        try:
            self.dataset_service = get_dataset_service(service_name='visualization')
        except Exception as e:
            logger.warning(f"{ICONS.get('warning', '⚠️')} Gagal menginisialisasi dataset service: {str(e)}")
            self.dataset_service = None
            
    def initialize_config(self):
        """
        Inisialisasi konfigurasi dataset dari ConfigManager
        """
        try:
            # Cek apakah berjalan di Colab
            def is_in_colab():
                try:
                    import google.colab
                    return True
                except ImportError:
                    return False
            
            # Coba dapatkan path dataset
            dataset_config = self.config_manager.get_module_config('dataset_config')
            
            # Jika di Colab, coba gunakan path dari Google Drive
            if is_in_colab():
                try:
                    from smartcash.common.constants.paths import DRIVE_DATASET_PATH, COLAB_DATASET_PATH
                    
                    # Cek apakah Google Drive sudah di-mount
                    if os.path.exists('/content/drive/MyDrive'):
                        self.dataset_path = DRIVE_DATASET_PATH
                        logger.info(f"💾 Menggunakan dataset path dari Google Drive: {self.dataset_path}")
                    else:
                        # Jika Drive belum di-mount, gunakan path Colab lokal
                        self.dataset_path = COLAB_DATASET_PATH
                        logger.info(f"💾 Menggunakan dataset path lokal Colab: {self.dataset_path}")
                except ImportError:
                    # Jika tidak bisa import paths, gunakan path default
                    self.dataset_path = "/content/data"
                    logger.info(f"💾 Menggunakan default path Colab: {self.dataset_path}")
            # Jika tidak di Colab, gunakan path dari konfigurasi
            elif dataset_config and 'dataset_path' in dataset_config and dataset_config['dataset_path']:
                self.dataset_path = dataset_config['dataset_path']
                logger.info(f"💾 Menggunakan dataset path dari konfigurasi: {self.dataset_path}")
            else:
                # Fallback ke path default
                self.dataset_path = "data"
                logger.warning(f"⚠️ Dataset path tidak ditemukan dalam konfigurasi, menggunakan default: {self.dataset_path}")
            
            # Validasi path dataset
            if os.path.exists(self.dataset_path):
                logger.info(f"✅ Dataset path valid: {self.dataset_path}")
                return True
            else:
                logger.warning(f"⚠️ Dataset path tidak ditemukan: {self.dataset_path}")
                
                # Coba cari direktori data di lokasi lain
                possible_paths = [
                    "data",
                    "/content/data",
                    "/content/smartcash/data",
                    "/content/drive/MyDrive/smartcash/data"
                ]
                
                for path in possible_paths:
                    if os.path.exists(path):
                        self.dataset_path = path
                        logger.info(f"🔍 Menemukan dataset path alternatif: {self.dataset_path}")
                        return True
                
                self.dataset_path = None
                return False
        except Exception as e:
            logger.error(f"⚠️ Gagal menginisialisasi dataset service: {str(e)}")
            self.dataset_path = None
            return False
        
    def initialize(self) -> Dict[str, Any]:
        """
        Inisialisasi UI dan komponen visualisasi.
        
        Returns:
            Dictionary berisi komponen UI
        """
        try:
            # Update loading indicator jika ada
            if self.loading_indicator:
                self.loading_indicator.update(10, "Mempersiapkan komponen visualisasi...")
                
            # Buat container utama dengan layout minimalis
            main_container = widgets.VBox([], layout=widgets.Layout(width='100%', padding='10px'))
            
            # Buat status panel untuk menampilkan pesan status
            status_panel = widgets.Output(layout=widgets.Layout(width='100%', margin='5px 0'))
            
            # Buat dashboard cards container
            dashboard_cards_container = widgets.VBox([], layout=widgets.Layout(width='100%', margin='10px 0'))
            
            # Buat container untuk kartu dataset stats
            dataset_stats_cards = widgets.Output(layout=widgets.Layout(width='100%', margin='10px 0'))
            processing_cards_container = widgets.HBox([
                widgets.Output(layout=widgets.Layout(width='50%')),  # preprocessing_stats_cards
                widgets.Output(layout=widgets.Layout(width='50%'))   # augmentation_stats_cards
            ], layout=widgets.Layout(width='100%', margin='10px 0'))
            
            # Tambahkan kartu ke dashboard container
            dashboard_cards_container.children = [
                widgets.HTML("<h3 style='margin: 10px 0; color: #0d47a1;'>📊 Dashboard Dataset</h3>"),
                status_panel,
                dataset_stats_cards,
                processing_cards_container
            ]
            
            # Buat refresh button untuk dashboard
            refresh_button = widgets.Button(
                description='Refresh Dashboard',
                icon='sync',
                button_style='info',
                layout=widgets.Layout(width='auto', margin='10px 0')
            )
            
            # Tambahkan refresh button ke dashboard container
            dashboard_cards_container.children = list(dashboard_cards_container.children) + [refresh_button]
            
            # Update loading indicator jika ada
            if self.loading_indicator:
                self.loading_indicator.update(30, "Mempersiapkan tab visualisasi...")
                
            # Buat tab untuk visualisasi lainnya
            visualization_tabs_components = create_visualization_tabs()
            visualization_tabs = visualization_tabs_components['tab']
            
            # Tambahkan dashboard cards dan tabs ke container utama
            main_container.children = [dashboard_cards_container, visualization_tabs]
            
            # Buat dictionary untuk komponen UI
            preprocessing_stats_cards = processing_cards_container.children[0]
            augmentation_stats_cards = processing_cards_container.children[1]
            
            self.ui_components = {
                'main_container': main_container,
                'status_panel': status_panel,
                'dashboard_cards_container': dashboard_cards_container,
                'dataset_stats_cards': dataset_stats_cards,
                'processing_cards_container': processing_cards_container,
                'preprocessing_stats_cards': preprocessing_stats_cards,
                'augmentation_stats_cards': augmentation_stats_cards,
                'visualization_tabs': visualization_tabs,
                'refresh_button': refresh_button,
                **visualization_tabs_components
            }
            
            # Setup handlers untuk visualisasi
            if self.loading_indicator:
                self.loading_indicator.update(50, "Menyiapkan handlers visualisasi...")
                
            # Setup handler untuk refresh button
            refresh_button.on_click(lambda b: self.update_dashboard())
            
            # Setup handlers untuk tab visualisasi
            self.ui_components = setup_tab_handlers(self.ui_components)
            
            # Update dashboard dengan data awal
            if self.loading_indicator:
                self.loading_indicator.update(70, "Memperbarui dashboard...")
                
            self.update_dashboard()
            
            # Selesai inisialisasi
            if self.loading_indicator:
                self.loading_indicator.complete("Visualisasi dataset berhasil dimuat")
                
            return self.ui_components
            
        except Exception as e:
            error_message = f"Error saat inisialisasi visualisasi: {str(e)}"
            logger.error(f"{ICONS.get('error', '❌')} {error_message}")
            
            if self.loading_indicator:
                self.loading_indicator.error(error_message)
                
            # Buat container minimal dengan pesan error
            error_container = widgets.VBox([
                widgets.HTML(f"<div style='color: red; padding: 10px;'>{ICONS.get('error', '❌')} {error_message}</div>")
            ])
            
            return {'main_container': error_container, 'error': str(e)}
    
    def _get_dummy_stats(self):
        """
        Mendapatkan data dummy untuk statistik dataset
        
        Returns:
            Dictionary berisi statistik dataset dummy
        """
        return {
            'split': {
                'train': {'images': 1400, 'labels': 1400},
                'val': {'images': 300, 'labels': 300},
                'test': {'images': 300, 'labels': 300}
            },
            'preprocessing': {
                'train_processed': 1000,
                'val_processed': 200,
                'test_processed': 200,
                'total_processed': 1400
            },
            'augmentation': {
                'train_augmented': 700,
                'val_augmented': 150,
                'test_augmented': 150,
                'total_augmented': 1000
            }
        }
        
    def _update_dataset_stats_cards(self, stats):
        """
        Perbarui dataset stats cards dengan data terbaru
        
        Args:
            stats: Dictionary berisi statistik dataset
        """
        try:
            # Ambil komponen UI
            dataset_stats_cards = self.ui_components.get('dataset_stats_cards')
            
            if dataset_stats_cards:
                dataset_stats_cards.clear_output()
                with dataset_stats_cards:
                    from smartcash.ui.dataset.visualization.components.dataset_stats_cards import create_dataset_stats_cards
                    display(create_dataset_stats_cards(stats))
        except Exception as e:
            logger.error(f"{ICONS.get('error', '❌')} Error saat memperbarui dataset stats cards: {str(e)}")
    
    def _update_preprocessing_stats_cards(self, stats):
        """
        Perbarui preprocessing stats cards dengan data terbaru
        
        Args:
            stats: Dictionary berisi statistik preprocessing
        """
        try:
            # Ambil komponen UI
            preprocessing_stats_cards = self.ui_components.get('preprocessing_stats_cards')
            
            if preprocessing_stats_cards:
                preprocessing_stats_cards.clear_output()
                with preprocessing_stats_cards:
                    from smartcash.ui.dataset.visualization.components.dataset_stats_cards import create_preprocessing_stats_cards
                    display(create_preprocessing_stats_cards(stats))
        except Exception as e:
            logger.error(f"{ICONS.get('error', '❌')} Error saat memperbarui preprocessing stats cards: {str(e)}")
    
    def _update_augmentation_stats_cards(self, stats):
        """
        Perbarui augmentation stats cards dengan data terbaru
        
        Args:
            stats: Dictionary berisi statistik augmentasi
        """
        try:
            # Ambil komponen UI
            augmentation_stats_cards = self.ui_components.get('augmentation_stats_cards')
            
            if augmentation_stats_cards:
                augmentation_stats_cards.clear_output()
                with augmentation_stats_cards:
                    from smartcash.ui.dataset.visualization.components.dataset_stats_cards import create_augmentation_stats_cards
                    display(create_augmentation_stats_cards(stats))
        except Exception as e:
            logger.error(f"{ICONS.get('error', '❌')} Error saat memperbarui augmentation stats cards: {str(e)}")
    
    def update_dashboard(self):
        """
        Perbarui dashboard dengan data terbaru
        """
        try:
            # Inisialisasi status panel
            self.ui_components['status_panel'].clear_output()
            with self.ui_components['status_panel']:
                status_text = widgets.HTML("<div style='padding: 10px; background-color: #e3f2fd; border-radius: 5px;'>"
                                         f"<p><b>{ICONS.get('info', 'ℹ️')} Status:</b> Memuat data dataset...</p></div>")
                display(status_text)
            
            # Inisialisasi variabel
            using_dummy_data = False
            stats = {}
            
            # Cek apakah dataset path tersedia
            if not self.dataset_path or not os.path.exists(self.dataset_path):
                error_message = "Dataset path tidak ditemukan dalam konfigurasi"
                if self.dataset_path:
                    error_message = f"Dataset path '{self.dataset_path}' tidak ditemukan"
                    
                logger.warning(f"{ICONS.get('warning', '⚠️')} {error_message}")
                using_dummy_data = True
            else:
                # Perbarui loading indicator jika ada
                if self.loading_indicator:
                    self.loading_indicator.update(40, "Mengambil statistik dataset...")
                    
                # Dapatkan statistik dataset jika dataset_service tersedia
                if self.dataset_service:
                    stats = self.dataset_service.get_dataset_stats()
                else:
                    logger.warning(f"{ICONS.get('warning', '⚠️')} Dataset service tidak tersedia")
                    using_dummy_data = True
            
            # Gunakan data dummy jika diperlukan
            if using_dummy_data:
                stats = self._get_dummy_stats()
            
            # Perbarui loading indicator jika ada
            if self.loading_indicator:
                self.loading_indicator.update(60, "Memperbarui dashboard...")
            
            # Perbarui dataset stats cards
            self._update_dataset_stats_cards(stats)
            
            # Perbarui preprocessing stats cards
            self._update_preprocessing_stats_cards(stats)
            
            # Perbarui augmentation stats cards
            self._update_augmentation_stats_cards(stats)
            
            # Perbarui loading indicator jika ada
            if self.loading_indicator:
                self.loading_indicator.update(90, "Memperbarui statistik augmentasi...")
            
            # Perbarui status panel
            self.ui_components['status_panel'].clear_output()
            with self.ui_components['status_panel']:
                if using_dummy_data:
                    status_html = f"<div style='padding: 10px; background-color: #fff3e0; border-radius: 5px;'>"
                    status_html += f"<p><b>{ICONS.get('warning', '⚠️')} Perhatian:</b> Menggunakan data dummy karena dataset tidak tersedia.</p>"
                    status_html += f"<p>Path dataset tidak valid</p>"
                    status_html += "</div>"
                else:
                    status_html = f"<div style='padding: 10px; background-color: #e8f5e9; border-radius: 5px;'>"
                    status_html += f"<p><b>{ICONS.get('success', '✅')} Status:</b> Data dataset berhasil dimuat.</p>"
                    status_html += f"<p>Dataset path: {self.dataset_path}</p>"
                    status_html += "</div>"
                
                display(widgets.HTML(status_html))
            
            # Perbarui loading indicator jika ada
            if self.loading_indicator:
                self.loading_indicator.update(100, "Dashboard berhasil diperbarui")
                
            return True
        except Exception as e:
            logger.error(f"{ICONS.get('error', '❌')} Error saat memperbarui dashboard: {str(e)}")
            
            # Perbarui status panel dengan error
            self.ui_components['status_panel'].clear_output()
            with self.ui_components['status_panel']:
                status_html = f"<div style='padding: 10px; background-color: #ffebee; border-radius: 5px;'>"
                status_html += f"<p><b>{ICONS.get('error', '❌')} Error:</b> {str(e)}</p>"
                status_html += "</div>"
                display(widgets.HTML(status_html))
            
            return False

# Fungsi helper untuk mendapatkan instance VisualizationManager
def get_visualization_manager(loading_indicator: Optional[LoadingIndicator] = None) -> VisualizationManager:
    """
    Dapatkan instance VisualizationManager.
    
    Args:
        loading_indicator: Indikator loading opsional
        
    Returns:
        Instance VisualizationManager
    """
    return VisualizationManager(loading_indicator)
