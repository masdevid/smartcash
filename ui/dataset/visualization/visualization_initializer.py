"""
File: smartcash/ui/dataset/visualization/visualization_initializer.py
Deskripsi: Inisialisasi UI untuk visualisasi dataset dengan pendekatan minimalis
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
from smartcash.ui.utils.loading_indicator import create_loading_indicator, LoadingIndicator

logger = get_logger(__name__)

def initialize_visualization_ui(loading_indicator: Optional[LoadingIndicator] = None) -> Dict[str, Any]:
    """
    Inisialisasi UI untuk visualisasi dataset dengan pendekatan minimalis.
    
    Args:
        loading_indicator: Indikator loading opsional untuk menampilkan progress
        
    Returns:
        Dictionary berisi komponen UI
    """
    try:
        # Update loading indicator jika ada
        if loading_indicator:
            loading_indicator.update(25, "Mempersiapkan komponen visualisasi...")
            
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
        
        # Buat tab untuk visualisasi lainnya
        visualization_tabs = widgets.Tab(layout=widgets.Layout(width='100%', margin='20px 0 10px 0'))
        visualization_tabs.children = [widgets.VBox([widgets.HTML("<p>Tab visualisasi lainnya akan ditampilkan di sini</p>")])]
        visualization_tabs.set_title(0, 'Visualisasi Lainnya')
        
        # Tambahkan dashboard cards dan tabs ke container utama
        main_container.children = [dashboard_cards_container, visualization_tabs]
        
        # Buat dictionary untuk komponen UI
        ui_components = {
            'main_container': main_container,
            'status_panel': status_panel,
            'dashboard_cards_container': dashboard_cards_container,
            'dataset_stats_cards': dataset_stats_cards,
            'preprocessing_stats_cards': preprocessing_stats_cards,
            'augmentation_stats_cards': augmentation_stats_cards,
            'visualization_tabs': visualization_tabs,
            'refresh_button': refresh_button
        }
        
        # Update dashboard cards secara asinkron untuk menghindari UI freeze
        if loading_indicator:
            loading_indicator.update(50, "Memperbarui dashboard...")
            
            # Gunakan threading untuk memperbarui dashboard secara asinkron
            def update_dashboard_async():
                try:
                    update_dashboard_cards(ui_components, loading_indicator)
                except Exception as e:
                    logger.error(f"Error saat memperbarui dashboard: {str(e)}")
                    if loading_indicator:
                        loading_indicator.error(f"Error saat memperbarui dashboard: {str(e)}")
            
            thread = threading.Thread(target=update_dashboard_async, daemon=True)
            thread.start()
        else:
            # Jika tidak ada loading indicator, update langsung
            update_dashboard_cards(ui_components)
        
        # Setup handler untuk tombol refresh
        refresh_button.on_click(lambda b: update_dashboard_cards(ui_components))
        
        return ui_components
        
    except Exception as e:
        logger.error(f"{ICONS.get('error', '❌')} Error saat inisialisasi visualisasi dataset: {str(e)}")
        
        # Update loading indicator jika ada
        if loading_indicator:
            loading_indicator.error(f"Error: {str(e)}")
        
        # Buat container minimal untuk menampilkan error
        error_container = widgets.VBox([
            widgets.HTML(f"<div style='color: #d9534f; padding: 10px; border-left: 5px solid #d9534f;'>"
                        f"<h3>{ICONS.get('error', '❌')} Error saat inisialisasi visualisasi dataset</h3>"
                        f"<p>{str(e)}</p></div>")
        ])
        
        return {'error_container': error_container, 'main_container': error_container}


def update_dashboard_cards(ui_components: Dict[str, Any], loading_indicator: Optional[LoadingIndicator] = None) -> None:
    """
    Update dashboard cards dengan data terbaru.
    
    Args:
        ui_components: Dictionary berisi komponen UI
        loading_indicator: Indikator loading opsional
    """
    try:
        # Ambil komponen UI
        status_panel = ui_components.get('status_panel')
        dataset_stats_cards = ui_components.get('dataset_stats_cards')
        preprocessing_stats_cards = ui_components.get('preprocessing_stats_cards')
        augmentation_stats_cards = ui_components.get('augmentation_stats_cards')
        
        # Tampilkan loading indicator jika ada
        if loading_indicator:
            loading_indicator.update(25, "Memperbarui data dashboard...")
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
            # Ambil dataset service
            dataset_service = get_dataset_service(service_name='visualization')
            
            config_manager = ConfigManager()
            dataset_config = config_manager.get_module_config('dataset_config')
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
                if loading_indicator:
                    loading_indicator.update(40, "Mengambil statistik dataset...")
                    
                # Dapatkan statistik dataset
                stats = dataset_service.get_dataset_statistics(dataset_path)
        except Exception as e:
            error_message = str(e)
            logger.warning(f"{ICONS.get('warning', '⚠️')} Error saat mengakses dataset: {error_message}")
            using_dummy_data = True
        
        # Jika menggunakan data dummy
        if using_dummy_data:
            # Perbarui loading indicator jika ada
            if loading_indicator:
                loading_indicator.update(40, "Menyiapkan data dummy...")
            
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
        if loading_indicator:
            loading_indicator.update(60, "Memperbarui statistik dataset...")
        
        # Update dataset stats cards
        if dataset_stats_cards:
            with dataset_stats_cards:
                clear_output(wait=True)
                display(create_dataset_stats_cards(stats))
        
        # Perbarui loading indicator jika ada
        if loading_indicator:
            loading_indicator.update(75, "Memperbarui statistik preprocessing...")
        
        # Update preprocessing stats cards
        if preprocessing_stats_cards:
            with preprocessing_stats_cards:
                clear_output(wait=True)
                display(create_preprocessing_stats_cards(stats))
        
        # Perbarui loading indicator jika ada
        if loading_indicator:
            loading_indicator.update(90, "Memperbarui statistik augmentasi...")
        
        # Update augmentation stats cards
        if augmentation_stats_cards:
            with augmentation_stats_cards:
                clear_output(wait=True)
                display(create_augmentation_stats_cards(stats))
        
        # Tampilkan pesan sukses
        success_message = "Dashboard berhasil diperbarui" + (" dengan data dummy" if using_dummy_data else "")
        if loading_indicator:
            loading_indicator.complete(success_message)
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
        if loading_indicator:
            loading_indicator.error(error_message)
        elif status_panel:
            status_panel.clear_output(wait=True)
            with status_panel:
                display(create_status_indicator("error", f"{ICONS.get('error', '❌')} {error_message}"))
