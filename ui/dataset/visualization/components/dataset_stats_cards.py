"""
File: smartcash/ui/dataset/visualization/components/dataset_stats_cards.py
Deskripsi: Komponen untuk menampilkan statistik dataset dengan visualisasi interaktif
"""

import ipywidgets as widgets
from IPython.display import display
from typing import Dict, Any, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

from smartcash.ui.utils.constants import COLORS
from smartcash.common.logger import get_logger

logger = get_logger(__name__)

class DatasetStatsCards:
    """
    Komponen untuk menampilkan kartu statistik dataset
    """

    def __init__(self):
        # Style untuk kartu statistik
        self.card_style = {
            'border': '1px solid #e0e0e0',
            'border_radius': '8px',
            'padding': '15px',
            'margin': '5px',
            'background': '#ffffff',
            'box_shadow': '0 2px 4px rgba(0,0,0,0.1)'
        }

    def create_card(self, title: str, value: Any, icon: str) -> widgets.VBox:
        """Buat kartu statistik individual"""
        return widgets.VBox([
            widgets.HTML(f"<div style='font-size: 14px; font-weight: bold;'>{icon} {title}</div>"),
            widgets.HTML(f"<div style='font-size: 24px; font-weight: bold; margin-top: 5px;'>{value}</div>")
        ], layout=widgets.Layout(**self.card_style))

    def display(self, stats: Dict[str, Any]) -> widgets.HBox:
        """Tampilkan kartu statistik dalam layout horizontal"""
        # Buat kartu untuk setiap statistik
        cards = [
            self.create_card("Total Gambar", stats.get('total_images', 0), "🖼️"),
            self.create_card("Gambar Diaugmentasi", stats.get('augmented', 0), "🔄"),
            self.create_card("File Diproses", stats.get('preprocessed', 0), "📄"),
            self.create_card("Status Validasi", stats.get('validation_rate', '0%'), "✅")
        ]
        
        # Tampilkan dalam HBox
        container = widgets.HBox(
            cards,
            layout=widgets.Layout(
                display='flex',
                flex_flow='row wrap',
                justify_content='space-between',
                width='100%'
            )
        )
        display(container)
