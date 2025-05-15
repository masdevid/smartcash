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
from smartcash.common.config.manager import ConfigManager
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
        
        # Cek dan buat konfigurasi default jika perlu
        self._initialize_config()
        
        # Inisialisasi dataset service setelah memastikan konfigurasi ada
        try:
            self.dataset_service = get_dataset_service(service_name='visualization')
        except Exception as e:
            logger.warning(f"{ICONS.get('warning', '⚠️')} Gagal menginisialisasi dataset service: {str(e)}")
            self.dataset_service = None
            
    def _initialize_config(self):
        """
        Inisialisasi konfigurasi dataset jika belum ada.
        """
        dataset_config = self.config_manager.get_module_config('dataset_config')
        
        if not dataset_config or 'dataset_path' not in dataset_config:
            # Buat konfigurasi default jika tidak ada
            default_config = {
                'dataset_path': '',
                'dataset_name': 'Dataset Belum Diinisialisasi',
                'split_ratio': {'train': 0.7, 'val': 0.15, 'test': 0.15}
            }
            
            # Simpan konfigurasi default
            self.config_manager.save_module_config('dataset_config', default_config)
            logger.info(f"{ICONS.get('info', 'ℹ️')} Dibuat konfigurasi dataset default karena tidak ditemukan")
        
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
            preprocessing_stats_cards = widgets.Output(layout=widgets.Layout(width='100%', margin='10px 0'))
            augmentation_stats_cards = widgets.Output(layout=widgets.Layout(width='100%', margin='10px 0'))
            
            # Tambahkan kartu ke dashboard container
            dashboard_cards_container.children = [
                widgets.HTML("<h3 style='margin: 10px 0; color: #0d47a1;'>📊 Dashboard Dataset</h3>"),
                status_panel,
                dataset_stats_cards,
                preprocessing_stats_cards,
                augmentation_stats_cards
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
            self.ui_components = {
                'main_container': main_container,
                'status_panel': status_panel,
                'dashboard_cards_container': dashboard_cards_container,
                'dataset_stats_cards': dataset_stats_cards,
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
    
    def update_dashboard(self) -> None:
        """Update dashboard dengan data terbaru."""
        try:
            # Ambil komponen UI
            status_panel = self.ui_components.get('status_panel')
            dataset_stats_cards = self.ui_components.get('dataset_stats_cards')
            preprocessing_stats_cards = self.ui_components.get('preprocessing_stats_cards')
            augmentation_stats_cards = self.ui_components.get('augmentation_stats_cards')
            
            # Tampilkan loading indicator jika ada
            if self.loading_indicator:
                self.loading_indicator.update(25, "Memperbarui data dashboard...")
            elif status_panel:
                status_panel.clear_output(wait=True)
                with status_panel:
                    display(create_status_indicator("info", f"{ICONS.get('loading', '⏳')} Memperbarui dashboard..."))
            
            # Inisialisasi variabel untuk menandai apakah menggunakan data dummy
            using_dummy_data = False
            error_message = ""
            stats = {}
            
            # Coba dapatkan dataset path dari konfigurasi
            try:
                dataset_config = self.config_manager.get_module_config('dataset_config')
                dataset_path = dataset_config.get('dataset_path', None)
                
                if not dataset_path or not os.path.exists(dataset_path):
                    if not dataset_path:
                        error_message = "Dataset path tidak ditemukan dalam konfigurasi"
                    else:
                        error_message = f"Dataset path '{dataset_path}' tidak ditemukan"
                        
                    logger.warning(f"{ICONS.get('warning', '⚠️')} {error_message}")
                    using_dummy_data = True
                else:
                    # Perbarui loading indicator jika ada
                    if self.loading_indicator:
                        self.loading_indicator.update(40, "Mengambil statistik dataset...")
                        
                    # Dapatkan statistik dataset jika dataset_service tersedia
                    if self.dataset_service:
                        stats = self.dataset_service.get_dataset_statistics(dataset_path)
                    else:
                        logger.warning(f"{ICONS.get('warning', '⚠️')} Dataset service tidak tersedia, menggunakan data dummy")
                        using_dummy_data = True
            except Exception as e:
                error_message = str(e)
                logger.warning(f"{ICONS.get('warning', '⚠️')} Error saat mengakses dataset: {error_message}")
                using_dummy_data = True
            
            # Jika menggunakan data dummy
            if using_dummy_data:
                # Perbarui loading indicator jika ada
                if self.loading_indicator:
                    self.loading_indicator.update(40, "Menyiapkan data dummy...")
                
                # Gunakan data dummy
                stats = {
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
                
                # Tampilkan pesan warning di status panel
                if status_panel:
                    status_panel.clear_output(wait=True)
                    with status_panel:
                        display(widgets.HTML(f"""
                        <div style="background-color: #fff3cd; padding: 10px; border-radius: 5px; margin-bottom: 10px; border-left: 5px solid #ffc107;">
                            <p style="margin: 0;"><strong>{ICONS.get('warning', '⚠️')} Status Data:</strong> Menggunakan data dummy untuk visualisasi. Silakan inisialisasi dataset untuk melihat data aktual.</p>
                        </div>
                        """))
            
            # Perbarui loading indicator jika ada
            if self.loading_indicator:
                self.loading_indicator.update(60, "Memperbarui statistik dataset...")
            
            # Update dataset stats cards
            if dataset_stats_cards:
                with dataset_stats_cards:
                    clear_output(wait=True)
                    display(create_dataset_stats_cards(stats))
            
            # Perbarui loading indicator jika ada
            if self.loading_indicator:
                self.loading_indicator.update(75, "Memperbarui statistik preprocessing...")
            
            # Update preprocessing stats cards
            if preprocessing_stats_cards:
                with preprocessing_stats_cards:
                    clear_output(wait=True)
                    display(create_preprocessing_stats_cards(stats))
            
            # Perbarui loading indicator jika ada
            if self.loading_indicator:
                self.loading_indicator.update(90, "Memperbarui statistik augmentasi...")
            
            # Update augmentation stats cards
            if augmentation_stats_cards:
                with augmentation_stats_cards:
                    clear_output(wait=True)
                    display(create_augmentation_stats_cards(stats))
            
            # Tampilkan pesan sukses
            success_message = "Dashboard berhasil diperbarui" + (" dengan data dummy" if using_dummy_data else "")
            if self.loading_indicator:
                self.loading_indicator.complete(success_message)
            elif status_panel and not using_dummy_data:
                status_panel.clear_output(wait=True)
                with status_panel:
                    display(create_status_indicator("success", f"{ICONS.get('success', '✅')} {success_message}"))
            
            # Log dengan timestamp
            current_time = datetime.now().strftime("%H:%M:%S")
            logger.info(f"{ICONS.get('info', 'ℹ️')} [{current_time}] 📊 {success_message}")
        
        except Exception as e:
            error_message = f"Error saat memperbarui dashboard: {str(e)}"
            logger.error(f"{ICONS.get('error', '❌')} {error_message}")
            
            # Tampilkan pesan error
            if self.loading_indicator:
                self.loading_indicator.error(error_message)
            elif status_panel:
                status_panel.clear_output(wait=True)
                with status_panel:
                    display(create_status_indicator("error", f"{ICONS.get('error', '❌')} {error_message}"))

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
