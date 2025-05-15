"""
File: smartcash/ui/dataset/visualization/handlers/dashboard_handler.py
Deskripsi: Handler untuk dashboard visualisasi dataset dengan pendekatan minimalis
"""

import os
from typing import Dict, Any, Optional
import ipywidgets as widgets
from IPython.display import display, clear_output
import threading
from datetime import datetime

from smartcash.ui.utils.constants import ICONS
from smartcash.ui.utils.alert_utils import create_status_indicator
from smartcash.common.logger import get_logger
from smartcash.common.config.manager import ConfigManager
from smartcash.dataset.services.service_factory import get_dataset_service
from smartcash.ui.dataset.visualization.components.dashboard_cards import (
    create_preprocessing_cards, create_augmentation_cards, update_preprocessing_cards, update_augmentation_cards
)
from smartcash.ui.utils.loading_indicator import create_loading_indicator, LoadingIndicator

logger = get_logger(__name__)


def create_dashboard_handler() -> Dict[str, Any]:
    """
    Buat handler untuk dashboard visualisasi dataset.
    
    Returns:
        Dictionary berisi komponen UI dan handler
    """
    # Buat container untuk dashboard
    container = widgets.VBox([], layout=widgets.Layout(width='100%'))
    
    # Buat panel status
    status_panel = widgets.Output(layout=widgets.Layout(width='100%', margin='5px 0'))
    
    # Buat loading indicator
    loading_indicator = create_loading_indicator(
        message="Mempersiapkan dashboard", 
        is_indeterminate=False,
        auto_hide=True
    )
    
    # Buat container untuk kartu preprocessing dan augmentasi
    preprocessing_cards = widgets.Output(layout=widgets.Layout(width='100%'))
    augmentation_cards = widgets.Output(layout=widgets.Layout(width='100%'))
    
    # Buat tombol refresh
    refresh_button = widgets.Button(
        description='Refresh Dashboard',
        icon='sync',
        button_style='info',
        layout=widgets.Layout(width='auto', margin='10px 0')
    )
    
    # Tambahkan komponen ke container
    container.children = [
        status_panel,
        preprocessing_cards,
        augmentation_cards,
        refresh_button
    ]
    
    # Setup handler untuk tombol refresh
    refresh_button.on_click(
        lambda b: update_dashboard_cards({
            'dashboard_handler': {
                'status': status_panel,
                'preprocessing_cards': preprocessing_cards,
                'augmentation_cards': augmentation_cards,
                'loading_indicator': loading_indicator,
                'refresh_button': refresh_button
            }
        })
    )
    
    # Buat dictionary untuk handler
    handler = {
        'container': container,
        'status': status_panel,
        'preprocessing_cards': preprocessing_cards,
        'augmentation_cards': augmentation_cards,
        'loading_indicator': loading_indicator,
        'refresh_button': refresh_button
    }
    
    return handler


def update_dashboard_cards(ui_components: Dict[str, Any]) -> None:
    """
    Update dashboard cards dengan data terbaru.
    
    Args:
        ui_components: Dictionary berisi komponen UI
    """
    try:
        # Ambil komponen UI
        dashboard_handler = ui_components.get('dashboard_handler', {})
        status_panel = dashboard_handler.get('status')
        preprocessing_cards = dashboard_handler.get('preprocessing_cards')
        augmentation_cards = dashboard_handler.get('augmentation_cards')
        loading_indicator = dashboard_handler.get('loading_indicator')
        
        # Ambil dataset service
        dataset_service = get_dataset_service(service_name='visualization')
        
        # Tampilkan loading indicator jika ada
        if loading_indicator:
            loading_indicator.update(25, "Memperbarui data dashboard...")
        elif status_panel:
            status_panel.clear_output(wait=True)
            with status_panel:
                display(create_status_indicator("info", f"{ICONS.get('loading', '⏳')} Memperbarui dashboard..."))
        
        # Coba dapatkan dataset path dari konfigurasi
        try:
            config_manager = ConfigManager()
            dataset_config = config_manager.get_config('dataset_config')
            dataset_path = dataset_config.get('dataset_path', None)
            
            if not dataset_path:
                raise ValueError("Dataset path tidak ditemukan dalam konfigurasi")
                
            # Perbarui loading indicator jika ada
            if loading_indicator:
                loading_indicator.update(50, "Mengambil statistik dataset...")
                
            # Dapatkan statistik dataset
            stats = dataset_service.get_dataset_statistics(dataset_path)
            
            # Perbarui loading indicator jika ada
            if loading_indicator:
                loading_indicator.update(75, "Memperbarui kartu dashboard...")
            
            # Update preprocessing cards
            if preprocessing_cards:
                with preprocessing_cards:
                    clear_output(wait=True)
                    display(create_preprocessing_cards(stats.get('preprocessing', {})))
            
            # Update augmentation cards
            if augmentation_cards:
                with augmentation_cards:
                    clear_output(wait=True)
                    display(create_augmentation_cards(stats.get('augmentation', {})))
            
            # Tampilkan pesan sukses
            if loading_indicator:
                loading_indicator.complete("Dashboard berhasil diperbarui")
            elif status_panel:
                status_panel.clear_output(wait=True)
                with status_panel:
                    display(create_status_indicator("success", f"{ICONS.get('success', '✅')} Dashboard berhasil diperbarui"))
            
            # Log dengan timestamp
            current_time = datetime.now().strftime("%H:%M:%S")
            logger.info(f"{ICONS.get('info', 'ℹ️')} [{current_time}] 📊 Dashboard berhasil diperbarui")
            
        except Exception as e:
            logger.warning(f"{ICONS.get('warning', '⚠️')} Dataset path tidak ditemukan, menggunakan data dummy: {str(e)}")
            
            # Perbarui loading indicator jika ada
            if loading_indicator:
                loading_indicator.update(50, "Menyiapkan data dummy...")
            
            # Gunakan data dummy jika dataset path tidak ditemukan
            dummy_stats = {
                'preprocessing': {
                    'processed_images': 2000,
                    'filtered_images': 2000,
                    'normalized_images': 2000
                },
                'augmentation': {
                    'augmented_images': 2000,
                    'augmentations_created': 2000,
                    'augmentation_types': 5
                }
            }
            
            # Perbarui loading indicator jika ada
            if loading_indicator:
                loading_indicator.update(75, "Memperbarui kartu dashboard dengan data dummy...")
            
            # Update preprocessing cards dengan data dummy
            if preprocessing_cards:
                with preprocessing_cards:
                    clear_output(wait=True)
                    display(create_preprocessing_cards(dummy_stats.get('preprocessing', {})))
            
            # Update augmentation cards dengan data dummy
            if augmentation_cards:
                with augmentation_cards:
                    clear_output(wait=True)
                    display(create_augmentation_cards(dummy_stats.get('augmentation', {})))
            
            # Tampilkan pesan warning
            warning_message = "Menggunakan data dummy untuk visualisasi. Silakan inisialisasi dataset untuk melihat data aktual."
            
            if status_panel:
                status_panel.clear_output(wait=True)
                with status_panel:
                    display(widgets.HTML(f"""
                    <div style="background-color: #fff3cd; padding: 10px; border-radius: 5px; margin-bottom: 10px; border-left: 5px solid #ffc107;">
                        <p style="margin: 0;"><strong>{ICONS.get('warning', '⚠️')} Status Data:</strong> {warning_message}</p>
                    </div>
                    """))
            
            if loading_indicator:
                loading_indicator.complete("Dashboard berhasil diperbarui dengan data dummy")
            
            # Log dengan timestamp
            current_time = datetime.now().strftime("%H:%M:%S")
            logger.info(f"{ICONS.get('info', 'ℹ️')} [{current_time}] 📊 Dashboard berhasil diperbarui dengan data dummy")
    
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
