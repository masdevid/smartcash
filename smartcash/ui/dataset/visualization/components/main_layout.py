"""
File: smartcash/ui/dataset/visualization/components/main_layout.py
Deskripsi: Komponen layout utama untuk visualisasi dataset
"""

from typing import Dict, Any
import ipywidgets as widgets

from smartcash.common.logger import get_logger
from smartcash.ui.utils.constants import ICONS
from smartcash.ui.components.header import create_header
from smartcash.ui.components.tabs import create_tabs
from smartcash.ui.utils.alert_utils import create_status_indicator

logger = get_logger(__name__)

def create_visualization_layout() -> Dict[str, Any]:
    """
    Buat layout untuk visualisasi dataset.
    
    Returns:
        Dictionary berisi komponen UI
    """
    try:
        # Buat container utama
        main_container = widgets.VBox(layout=widgets.Layout(width='100%'))
        
        # Buat header
        header = create_header("Visualisasi Dataset", "Analisis dan visualisasi distribusi dataset")
        
        # Buat status panel
        status = widgets.Output(layout=widgets.Layout(width='100%', min_height='50px'))
        with status:
            display(create_status_indicator("info", "Siap memvisualisasikan dataset"))
        
        # Buat progress bar
        progress_bar = widgets.IntProgress(
            value=0,
            min=0,
            max=100,
            description='Loading:',
            bar_style='info',
            orientation='horizontal',
            layout=widgets.Layout(width='100%', visibility='hidden')
        )
        
        # Buat refresh button
        refresh_button = widgets.Button(
            description='Refresh Data',
            icon='sync',
            button_style='primary',
            tooltip='Refresh data visualisasi',
            layout=widgets.Layout(width='150px')
        )
        
        # Buat dashboard cards container
        dashboard_container = widgets.VBox([
            widgets.HBox([refresh_button], layout=widgets.Layout(justify_content='flex-end')),
            widgets.HBox([
                widgets.VBox([
                    widgets.HTML("<h3>Statistik Split Dataset</h3>"),
                    widgets.Output(layout=widgets.Layout(width='100%'))
                ], layout=widgets.Layout(width='100%')),
            ], layout=widgets.Layout(width='100%')),
            widgets.HBox([
                widgets.VBox([
                    widgets.HTML("<h3>Preprocessing</h3>"),
                    widgets.Output(layout=widgets.Layout(width='100%'))
                ], layout=widgets.Layout(width='33%')),
                widgets.VBox([
                    widgets.HTML("<h3>Augmentasi</h3>"),
                    widgets.Output(layout=widgets.Layout(width='100%'))
                ], layout=widgets.Layout(width='33%')),
            ], layout=widgets.Layout(width='100%'))
        ])
        
        # Buat tab visualisasi
        distribution_tab = {
            'button': widgets.Button(description='Tampilkan Distribusi Kelas', button_style='info'),
            'output': widgets.Output(layout=widgets.Layout(width='100%', min_height='400px'))
        }
        
        split_tab = {
            'button': widgets.Button(description='Tampilkan Distribusi Split', button_style='info'),
            'output': widgets.Output(layout=widgets.Layout(width='100%', min_height='400px'))
        }
        
        layer_tab = {
            'button': widgets.Button(description='Tampilkan Distribusi Layer', button_style='info'),
            'output': widgets.Output(layout=widgets.Layout(width='100%', min_height='400px'))
        }
        
        heatmap_tab = {
            'button': widgets.Button(description='Tampilkan Heatmap Deteksi', button_style='info'),
            'output': widgets.Output(layout=widgets.Layout(width='100%', min_height='400px'))
        }
        
        bbox_tab = {
            'button': widgets.Button(description='Tampilkan Analisis Bounding Box', button_style='info'),
            'output': widgets.Output(layout=widgets.Layout(width='100%', min_height='400px'))
        }
        
        # Buat tab content
        distribution_content = widgets.VBox([
            distribution_tab['button'],
            distribution_tab['output']
        ])
        
        split_content = widgets.VBox([
            split_tab['button'],
            split_tab['output']
        ])
        
        layer_content = widgets.VBox([
            layer_tab['button'],
            layer_tab['output']
        ])
        
        heatmap_content = widgets.VBox([
            heatmap_tab['button'],
            heatmap_tab['output']
        ])
        
        bbox_content = widgets.VBox([
            bbox_tab['button'],
            bbox_tab['output']
        ])
        
        # Buat tabs
        tabs = create_tabs([
            ('Dashboard', dashboard_container),
            ('Distribusi Kelas', distribution_content),
            ('Distribusi Split', split_content),
            ('Distribusi Layer', layer_content),
            ('Analisis Bounding Box', bbox_content),
            ('Heatmap Deteksi', heatmap_content)
        ])
        
        # Susun komponen
        main_container.children = [
            header,
            status,
            progress_bar,
            tabs
        ]
        
        # Kumpulkan semua komponen dalam dictionary
        ui_components = {
            'main_container': main_container,
            'header': header,
            'status': status,
            'progress_bar': progress_bar,
            'refresh_button': refresh_button,
            'tabs': tabs,
            'split_cards_container': dashboard_container.children[1].children[0].children[1],
            'preprocessing_cards': dashboard_container.children[2].children[0].children[1],
            'augmentation_cards': dashboard_container.children[2].children[1].children[1],
            'visualization_components': {
                'distribution_tab': distribution_tab,
                'split_tab': split_tab,
                'layer_tab': layer_tab,
                'heatmap_tab': heatmap_tab,
                'bbox_tab': bbox_tab
            }
        }
        
        return ui_components
    
    except Exception as e:
        logger.error(f"{ICONS.get('error', '❌')} Error saat membuat layout visualisasi: {str(e)}")
        return {} 