"""
File: smartcash/ui/dataset/visualization/visualization_components.py
Deskripsi: Komponen utama untuk visualisasi dataset
"""

from typing import Dict, Any, Optional, List, Callable
import ipywidgets as widgets
import pandas as pd
from IPython.display import display

from smartcash.ui.initializers.visualization_initializer import (
    VisualizationInitializer,
    create_line_plot,
    create_scatter_plot,
    create_hist_plot,
    create_distribution_viz,
    create_correlation_viz
)
from smartcash.ui.components import (
    create_header,
    create_tab_widget as create_tab,
    create_info_accordion,
    create_section_title
)
from smartcash.ui.utils.constants import COLORS, ICONS


class VisualizationUI:
    """Kelas utama untuk UI visualisasi dataset"""
    
    def __init__(self, data: pd.DataFrame, title: str = "Data Visualization"):
        """Inisialisasi UI visualisasi dengan data dan judul"""
        self.data = data
        self.title = title
        self.components = {}
        self.plots = {}
        self._setup_ui()
    
    def _setup_ui(self) -> None:
        """Setup komponen UI utama"""
        # Header
        self.components['header'] = create_header(
            title=self.title,
            description="Visualisasi interaktif untuk analisis dataset",
            icon=ICONS.get('chart', '📊')
        )
        
        # Tabs untuk berbagai jenis visualisasi
        self._setup_tabs()
        
        # Container utama
        self.components['main_container'] = widgets.VBox([
            self.components['header'],
            self.components['tabs']
        ], layout=widgets.Layout(width='100%'))
    
    def _setup_tabs(self) -> None:
        """Setup tab untuk berbagai jenis visualisasi"""
        # Tab Overview
        overview_tab = self._create_overview_tab()
        
        # Tab Distribusi
        distribution_tab = self._create_distribution_tab()
        
        # Tab Korelasi
        correlation_tab = self._create_correlation_tab()
        
        # Buat tab container
        self.components['tabs'] = create_tab({
            '📊 Overview': overview_tab,
            '📈 Distribusi': distribution_tab,
            '🔗 Korelasi': correlation_tab
        })
    
    def _create_overview_tab(self) -> widgets.Widget:
        """Buat tab overview dengan ringkasan data"""
        # Ringkasan statistik
        stats = self.data.describe().round(2)
        stats_output = widgets.Output()
        with stats_output:
            display(stats)
        
        # Plot line sederhana
        plot_output = widgets.Output()
        with plot_output:
            viz = VisualizationInitializer("Data Overview")
            viz.initialize(lambda: self._create_default_plot())
        
        return widgets.VBox([
            create_section_title("Statistik Deskriptif", ICONS.get('stats', '📊')),
            stats_output,
            create_section_title("Visualisasi Data", ICONS.get('chart', '📈')),
            plot_output
        ])
    
    def _create_distribution_tab(self) -> widgets.Widget:
        """Buat tab distribusi"""
        # Pilih kolom numerik
        numeric_cols = self.data.select_dtypes(include=['number']).columns.tolist()
        
        if not numeric_cols:
            return widgets.HTML("<p>Tidak ada kolom numerik yang tersedia untuk visualisasi distribusi.</p>")
        
        # Buat visualisasi distribusi
        output = widgets.Output()
        with output:
            create_distribution_viz("Distribusi Data", self.data, columns=numeric_cols[:3])
        
        return output
    
    def _create_correlation_tab(self) -> widgets.Widget:
        """Buat tab korelasi"""
        # Pilih kolom numerik
        numeric_cols = self.data.select_dtypes(include=['number']).columns.tolist()
        
        if len(numeric_cols) < 2:
            return widgets.HTML("<p>Diperuhkan minimal 2 kolom numerik untuk visualisasi korelasi.</p>")
        
        # Buat visualisasi korelasi
        output = widgets.Output()
        with output:
            create_correlation_viz("Korelasi Data", self.data[numeric_cols])
        
        return output
    
    def _create_default_plot(self) -> None:
        """Buat plot default untuk data"""
        try:
            if len(self.data) > 0:
                numeric_cols = self.data.select_dtypes(include=['number']).columns
                if len(numeric_cols) >= 2:
                    # Coba buat line plot dari 2 kolom numerik pertama
                    return create_line_plot("Data Plot", self.data, 
                                         x=numeric_cols[0], y=numeric_cols[1])
            
            # Fallback ke histogram jika tidak memungkinkan
            if len(numeric_cols) > 0:
                return create_hist_plot("Data Distribution", self.data, 
                                     column=numeric_cols[0])
            
            # Jika tidak ada kolom numerik, tampilkan pesan
            print("Tidak ada data numerik yang tersedia untuk divisualisasikan")
            
        except Exception as e:
            print(f"Error membuat plot: {str(e)}")
    
    def show(self) -> None:
        """Tampilkan UI visualisasi"""
        display(self.components['main_container'])
        return self.components['main_container']


def create_visualization_ui(data: pd.DataFrame, title: str = "Data Visualization") -> VisualizationUI:
    """
    Factory function untuk membuat UI visualisasi
    
    Args:
        data: DataFrame berisi data yang akan divisualisasikan
        title: Judul untuk UI
        
    Returns:
        Instance dari VisualizationUI
    """
    return VisualizationUI(data, title)
