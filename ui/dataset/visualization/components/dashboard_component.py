"""
File: smartcash/ui/dataset/visualization/components/dashboard_component.py
Deskripsi: Komponen utama untuk dashboard visualisasi dataset
"""

import ipywidgets as widgets
from typing import Dict, Any, List, Optional

from smartcash.ui.utils.constants import ICONS
from smartcash.ui.utils.header_utils import create_header
from smartcash.ui.utils.alert_utils import create_status_indicator
from smartcash.ui.dataset.visualization.components.dashboard_cards import (
    create_split_cards, create_preprocessing_cards, create_augmentation_cards
)
from smartcash.ui.dataset.visualization.components.visualization_tabs import create_visualization_tabs


def create_dashboard_component() -> Dict[str, Any]:
    """
    Membuat komponen dashboard untuk visualisasi dataset.
    
    Returns:
        Dictionary berisi komponen UI yang telah dibuat
    """
    # Container utama
    main_container = widgets.VBox([])
    
    # Status panel
    status = widgets.Output()
    
    # Progress bar
    progress_bar = widgets.IntProgress(
        value=0,
        min=0,
        max=100,
        description='Progress:',
        bar_style='info',
        orientation='horizontal',
        layout=widgets.Layout(width='100%', visibility='hidden')
    )
    
    # Buat header dengan create_header
    header = create_header(
        title="Visualisasi Dataset",
        description="Dashboard visualisasi dan analisis dataset untuk deteksi mata uang",
        icon=ICONS.get('dashboard', '📊')
    )
    
    # Container untuk cards
    cards_section = widgets.VBox([])
    
    # Judul section cards
    cards_title = widgets.HTML(
        value="<h3 style='margin-bottom: 10px;'>Dashboard Statistik Dataset</h3>"
    )
    
    # Container untuk status dashboard
    dashboard_status = widgets.Output()
    
    # Container untuk split cards
    split_cards_container = widgets.Output()
    
    # Container untuk cards preprocessing dan augmentation
    processing_cards_row = widgets.HBox([])
    
    # Container untuk preprocessing cards
    preprocessing_container = widgets.VBox([])
    preprocessing_title = widgets.HTML(
        value="<h4 style='margin-bottom: 5px;'>Preprocessing</h4>"
    )
    preprocessing_cards = widgets.Output()
    preprocessing_container.children = [preprocessing_title, preprocessing_cards]
    
    # Container untuk augmentation cards
    augmentation_container = widgets.VBox([])
    augmentation_title = widgets.HTML(
        value="<h4 style='margin-bottom: 5px;'>Augmentasi</h4>"
    )
    augmentation_cards = widgets.Output()
    augmentation_container.children = [augmentation_title, augmentation_cards]
    
    # Atur layout untuk preprocessing dan augmentation containers
    preprocessing_container.layout = widgets.Layout(
        width='48%',
        margin='0 1% 0 0',
        padding='10px',
        border='1px solid #eee',
        border_radius='5px'
    )
    
    augmentation_container.layout = widgets.Layout(
        width='48%',
        margin='0 0 0 1%',
        padding='10px',
        border='1px solid #eee',
        border_radius='5px'
    )
    
    # Tambahkan containers ke processing_cards_row
    processing_cards_row.children = [preprocessing_container, augmentation_container]
    
    # Tambahkan semua komponen ke cards_section
    cards_section.children = [cards_title, dashboard_status, processing_cards_row]
    
    # Buat komponen tab visualisasi
    visualization_components = create_visualization_tabs()
    visualization_tab = visualization_components['tab']
    
    # Tombol refresh
    refresh_button = widgets.Button(
        description=f"{ICONS.get('refresh', '🔄')} Refresh Data",
        button_style='primary',
        tooltip='Refresh data visualisasi'
    )
    
    # Tambahkan komponen ke container utama
    main_container.children = [
        header,
        status,
        refresh_button,
        progress_bar,
        cards_section,
        widgets.HTML("<hr style='margin: 20px 0;'>"),
        visualization_tab
    ]
    
    # Kumpulkan semua komponen dalam dictionary
    ui_components = {
        'main_container': main_container,
        'status': status,
        'progress_bar': progress_bar,
        'refresh_button': refresh_button,
        'cards_section': cards_section,
        'dashboard_status': dashboard_status,
        'split_cards_container': split_cards_container,
        'preprocessing_output': preprocessing_cards,
        'augmentation_output': augmentation_cards,
        'preprocessing_cards': preprocessing_cards,
        'augmentation_cards': augmentation_cards,
        'visualization_components': visualization_components
    }
    
    return ui_components
