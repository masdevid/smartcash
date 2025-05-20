"""
File: smartcash/dataset/visualization/helpers/chart_helper.py
Deskripsi: Utilitas untuk pembuatan dan styling berbagai jenis chart
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import Dict, List, Optional, Any, Tuple, Union
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from smartcash.common.logger import get_logger
from smartcash.common.visualization.helpers.color_helper import ColorHelper
from smartcash.common.visualization.helpers.annotation_helper import AnnotationHelper
from smartcash.common.visualization.helpers.style_helper import StyleHelper
from smartcash.common.visualization.core.visualization_base import VisualizationBase

class ChartHelper(VisualizationBase):
    """Helper untuk pembuatan dan styling berbagai jenis chart."""
    
    def __init__(self, style: str = 'seaborn-v0_8-whitegrid', logger=None):
        """
        Inisialisasi ChartHelper.
        
        Args:
            style: Style plot matplotlib
            logger: Logger kustom (opsional)
        """
        self.style = style
        self.logger = logger or get_logger()
        
        # Helper lain yang dibutuhkan
        self.color_helper = ColorHelper(logger)
        self.annotation_helper = AnnotationHelper(logger)
        self.style_helper = StyleHelper(logger)
        
        # Setup style
        self.style_helper.set_style(style)
        self.logger.info(f"🎨 ChartHelper diinisialisasi dengan style: {self.style}")
    
    def create_bar_chart(
        self, 
        data: Dict[str, Union[int, float]], 
        title: str = "", 
        horizontal: bool = False,
        figsize: Tuple[int, int] = (10, 6),
        color_palette: str = 'viridis',
        show_values: bool = True,
        sort_values: bool = True,
        top_n: Optional[int] = None,
        grid: bool = True,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None
    ) -> Tuple[Figure, Axes]:
        """
        Buat bar chart dari data.
        
        Args:
            data: Dictionary berisi data {kategori: nilai}
            title: Judul chart
            horizontal: Apakah bar horizontal
            figsize: Ukuran figure (width, height)
            color_palette: Nama palet warna
            show_values: Apakah menampilkan nilai di atas bar
            sort_values: Apakah mengurutkan nilai
            top_n: Jumlah item teratas yang ditampilkan
            grid: Apakah menampilkan grid
            xlabel: Label sumbu x
            ylabel: Label sumbu y
            
        Returns:
            Tuple (Figure, Axes)
        """
        # Create figure and axes
        fig, ax = plt.subplots(figsize=figsize)
        
        # Sort data if required
        if sort_values:
            data = dict(sorted(data.items(), key=lambda x: x[1], reverse=True))
            
        # Limit to top_n items if specified
        if top_n and len(data) > top_n:
            top_items = list(data.items())[:top_n]
            others_sum = sum(value for _, value in list(data.items())[top_n:])
            data = dict(top_items)
            if others_sum > 0:
                data['Lainnya'] = others_sum
        
        # Get colors
        colors = self.color_helper.get_color_palette(len(data), color_palette)
        
        # Create bar chart
        if horizontal:
            bars = ax.barh(list(data.keys()), list(data.values()), color=colors)
        else:
            bars = ax.bar(list(data.keys()), list(data.values()), color=colors)
        
        # Add value labels if requested
        if show_values:
            self.annotation_helper.add_bar_annotations(ax, bars, horizontal=horizontal)
        
        # Set title and labels
        self.style_helper.set_title_and_labels(ax, title, xlabel, ylabel)
        
        # Configure x-ticks
        if not horizontal and len(data) > 6:
            plt.xticks(rotation=45, ha='right')
        
        # Add grid
        if grid:
            ax.grid(axis='y' if not horizontal else 'x', linestyle='--', alpha=0.7)
                
        plt.tight_layout()
        return fig, ax
    
    def create_pie_chart(
        self, 
        data: Dict[str, Union[int, float]], 
        title: str = "",
        figsize: Tuple[int, int] = (8, 8),
        color_palette: str = 'Set2',
        show_values: bool = True,
        show_percents: bool = True,
        start_angle: int = 90,
        explode: Optional[List[float]] = None,
        sort_values: bool = True,
        top_n: Optional[int] = None,
        legend_loc: str = 'best'
    ) -> Tuple[Figure, Axes]:
        """
        Buat pie chart dari data.
        
        Args:
            data: Dictionary berisi data {kategori: nilai}
            title: Judul chart
            figsize: Ukuran figure (width, height)
            color_palette: Nama palet warna
            show_values: Apakah menampilkan nilai dalam label
            show_percents: Apakah menampilkan persentase dalam label
            start_angle: Sudut mulai (derajat)
            explode: List offset per wedge (opsional)
            sort_values: Apakah mengurutkan nilai
            top_n: Jumlah item teratas yang ditampilkan
            legend_loc: Lokasi legenda
            
        Returns:
            Tuple (Figure, Axes)
        """
        # Create figure and axes
        fig, ax = plt.subplots(figsize=figsize)
        
        # Sort data if required
        if sort_values:
            data = dict(sorted(data.items(), key=lambda x: x[1], reverse=True))
            
        # Limit to top_n items if specified
        if top_n and len(data) > top_n:
            top_items = list(data.items())[:top_n]
            others_sum = sum(value for _, value in list(data.items())[top_n:])
            data = dict(top_items)
            if others_sum > 0:
                data['Lainnya'] = others_sum
        
        # Get colors
        colors = self.color_helper.get_color_palette(len(data), color_palette)
        
        # Prepare autopct function based on show_values and show_percents
        autopct = self.annotation_helper.get_pie_autopct_func(data, show_values, show_percents)
        
        # Create pie chart
        wedges, texts, autotexts = ax.pie(
            list(data.values()), 
            labels=None,  # We'll use legend instead
            autopct=autopct,
            startangle=start_angle,
            explode=explode,
            colors=colors,
            shadow=False
        )
        
        # Style autotexts
        if autopct:
            for autotext in autotexts:
                autotext.set_fontsize(10)
                autotext.set_fontweight('bold')
        
        # Add legend
        ax.legend(wedges, list(data.keys()), 
                 title="Kategori",
                 loc=legend_loc,
                 bbox_to_anchor=(1, 0, 0.5, 1) if legend_loc == 'center left' else None)
        
        # Set title
        self.style_helper.set_title_and_labels(ax, title)
        
        # Equal aspect ratio ensures that pie is drawn as a circle
        ax.set_aspect('equal')
        
        plt.tight_layout()
        return fig, ax
    
    def create_line_chart(
        self, 
        data: Dict[str, List[Union[int, float]]],
        x_values: List[Any], 
        title: str = "",
        figsize: Tuple[int, int] = (10, 6),
        color_palette: str = 'tab10',
        markers: Optional[List[str]] = None,
        line_styles: Optional[List[str]] = None,
        show_points: bool = True,
        grid: bool = True,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        legend_loc: str = 'best'
    ) -> Tuple[Figure, Axes]:
        """
        Buat line chart dari data.
        
        Args:
            data: Dictionary berisi {series_name: [values]}
            x_values: List nilai untuk sumbu x
            title: Judul chart
            figsize: Ukuran figure (width, height)
            color_palette: Nama palet warna
            markers: List marker untuk setiap series
            line_styles: List style garis untuk setiap series
            show_points: Apakah menampilkan titik pada garis
            grid: Apakah menampilkan grid
            xlabel: Label sumbu x
            ylabel: Label sumbu y
            legend_loc: Lokasi legenda
            
        Returns:
            Tuple (Figure, Axes)
        """
        # Create figure and axes
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get colors
        colors = self.color_helper.get_color_palette(len(data), color_palette)
        
        # Default markers and line styles
        if markers is None:
            markers = ['o', 's', '^', 'd', 'v', '<', '>', 'p', '*', 'h'] * 3
        if line_styles is None:
            line_styles = ['-', '--', '-.', ':'] * 5
        
        # Plot each series
        for i, (series_name, values) in enumerate(data.items()):
            marker = markers[i % len(markers)] if show_points else None
            line_style = line_styles[i % len(line_styles)]
            ax.plot(x_values, values, 
                   label=series_name, 
                   color=colors[i],
                   marker=marker,
                   linestyle=line_style,
                   linewidth=2,
                   markersize=6)
        
        # Set title and labels
        self.style_helper.set_title_and_labels(ax, title, xlabel, ylabel)
        
        # Add grid
        if grid:
            ax.grid(linestyle='--', alpha=0.7)
        
        # Add legend
        ax.legend(loc=legend_loc)
        
        plt.tight_layout()
        return fig, ax
    
    def create_heatmap(
        self, 
        data: np.ndarray,
        row_labels: List[str],
        col_labels: List[str],
        title: str = "",
        figsize: Tuple[int, int] = (10, 8),
        color_map: str = 'viridis',
        show_values: bool = True,
        value_format: str = '.1f',
        cbar_label: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None
    ) -> Tuple[Figure, Axes]:
        """
        Buat heatmap dari data matrix.
        
        Args:
            data: Matrix numpy 2D
            row_labels: Label untuk baris
            col_labels: Label untuk kolom
            title: Judul heatmap
            figsize: Ukuran figure (width, height)
            color_map: Nama colormap
            show_values: Apakah menampilkan nilai dalam cell
            value_format: Format nilai (.1f, .2f, d, etc.)
            cbar_label: Label untuk colorbar
            xlabel: Label sumbu x
            ylabel: Label sumbu y
            
        Returns:
            Tuple (Figure, Axes)
        """
        # Create figure and axes
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        heatmap = sns.heatmap(
            data, 
            annot=show_values,
            fmt=value_format,
            cmap=color_map,
            cbar_kws={'label': cbar_label} if cbar_label else {},
            ax=ax
        )
        
        # Set labels
        ax.set_xticklabels(col_labels)
        ax.set_yticklabels(row_labels)
        
        # Rotate x labels if there are too many
        if len(col_labels) > 6:
            plt.xticks(rotation=45, ha='right')
        
        # Set title and labels
        self.style_helper.set_title_and_labels(ax, title, xlabel, ylabel)
        
        plt.tight_layout()
        return fig, ax
    
    def create_stacked_bar_chart(
        self, 
        data: Dict[str, Dict[str, Union[int, float]]],
        title: str = "",
        figsize: Tuple[int, int] = (10, 6),
        color_palette: str = 'tab10',
        horizontal: bool = False,
        show_values: bool = False,
        grid: bool = True,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        legend_loc: str = 'best'
    ) -> Tuple[Figure, Axes]:
        """
        Buat stacked bar chart dari data.
        
        Args:
            data: Nested dict {category: {stack_name: value}}
            title: Judul chart
            figsize: Ukuran figure (width, height)
            color_palette: Nama palet warna
            horizontal: Apakah bar horizontal
            show_values: Apakah menampilkan nilai dalam bar
            grid: Apakah menampilkan grid
            xlabel: Label sumbu x
            ylabel: Label sumbu y
            legend_loc: Lokasi legenda
            
        Returns:
            Tuple (Figure, Axes)
        """
        # Create figure and axes
        fig, ax = plt.subplots(figsize=figsize)
        
        # Extract categories and stack names
        categories = list(data.keys())
        stack_names = list(set().union(*[d.keys() for d in data.values()]))
        
        # Get colors
        colors = self.color_helper.get_color_palette(len(stack_names), color_palette)
        
        # Prepare data for plotting
        x = np.arange(len(categories))
        width = 0.8
        
        # Plot each stack
        bottom = np.zeros(len(categories))
        for i, stack in enumerate(stack_names):
            values = [data[cat].get(stack, 0) for cat in categories]
            
            if horizontal:
                bars = ax.barh(x, values, width, left=bottom, label=stack, color=colors[i])
                bottom += values
            else:
                bars = ax.bar(x, values, width, bottom=bottom, label=stack, color=colors[i])
                bottom += values
                
            # Add value labels if requested
            if show_values:
                self.annotation_helper.add_stacked_bar_annotations(ax, bars, values, horizontal)
        
        # Set x-axis properties
        if horizontal:
            ax.set_yticks(x)
            ax.set_yticklabels(categories)
        else:
            ax.set_xticks(x)
            ax.set_xticklabels(categories)
            if len(categories) > 6:
                plt.xticks(rotation=45, ha='right')
        
        # Set title and labels
        self.style_helper.set_title_and_labels(ax, title, xlabel, ylabel)
        
        # Add grid
        if grid:
            ax.grid(axis='x' if horizontal else 'y', linestyle='--', alpha=0.7)
        
        # Add legend
        ax.legend(loc=legend_loc)
        
        plt.tight_layout()
        return fig, ax