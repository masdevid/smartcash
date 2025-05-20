"""
File: smartcash/common/visualization/helpers/annotation_helper.py
Deskripsi: Utilitas untuk menambahkan anotasi pada visualisasi dengan inheritance dari VisualizationBase
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from matplotlib.axes import Axes
from matplotlib.container import BarContainer

from smartcash.common.logger import get_logger
from smartcash.common.visualization.core.visualization_base import VisualizationBase


class AnnotationHelper(VisualizationBase):
    """Helper untuk menambahkan dan mengelola anotasi pada visualisasi."""
    
    def __init__(self, logger=None):
        """
        Inisialisasi AnnotationHelper.
        
        Args:
            logger: Logger kustom (opsional)
        """
        super().__init__()
        self.logger = logger or get_logger()
        self.logger.info("📝 AnnotationHelper diinisialisasi")
    
    def add_bar_annotations(
        self, 
        ax: Axes, 
        bars: BarContainer,
        horizontal: bool = False,
        fontsize: int = 10,
        fontweight: str = 'bold',
        color: Optional[str] = None,
        format_str: str = '{}'
    ) -> None:
        """
        Tambahkan anotasi nilai pada bar chart.
        
        Args:
            ax: Axes matplotlib
            bars: Container bars dari ax.bar() atau ax.barh()
            horizontal: Apakah bar chart horizontal
            fontsize: Ukuran font anotasi
            fontweight: Ketebalan font (normal, bold)
            color: Warna teks (opsional)
            format_str: Format string untuk nilai (default: {})
        """
        # Tentukan nilai maksimum untuk penempatan teks
        if horizontal:
            max_value = max([bar.get_width() for bar in bars])
            offset = max_value * 0.01
        else:
            max_value = max([bar.get_height() for bar in bars])
            offset = max_value * 0.01
        
        # Tambahkan anotasi untuk setiap bar
        for bar in bars:
            if horizontal:
                value = bar.get_width()
                x = value + offset
                y = bar.get_y() + bar.get_height() / 2
                ha, va = 'left', 'center'
            else:
                value = bar.get_height()
                x = bar.get_x() + bar.get_width() / 2
                y = value + offset
                ha, va = 'center', 'bottom'
            
            # Format nilai
            text = format_str.format(value)
            
            # Tambahkan teks
            ax.text(
                x, y, text,
                ha=ha, va=va,
                fontsize=fontsize,
                fontweight=fontweight,
                color=color
            )
    
    def add_stacked_bar_annotations(
        self, 
        ax: Axes, 
        bars: BarContainer,
        values: List[Union[int, float]],
        horizontal: bool = False,
        fontsize: int = 9,
        threshold: float = 0.05,
        format_str: str = '{}'
    ) -> None:
        """
        Tambahkan anotasi untuk stacked bar chart.
        
        Args:
            ax: Axes matplotlib
            bars: Container bars dari ax.bar() atau ax.barh()
            values: List nilai untuk bar
            horizontal: Apakah bar chart horizontal
            fontsize: Ukuran font anotasi
            threshold: Batas minimum proporsi untuk menampilkan label
            format_str: Format string untuk nilai
        """
        # Dapatkan nilai total untuk menentukan proporsi
        total = sum(values)
        if total == 0:
            return
            
        # Tambahkan anotasi untuk setiap bar yang cukup besar
        for i, bar in enumerate(bars):
            value = values[i]
            proportion = value / total
            
            # Skip anotasi jika bar terlalu kecil
            if proportion < threshold:
                continue
                
            # Tentukan posisi teks
            if horizontal:
                width = bar.get_width()
                height = bar.get_height()
                x = bar.get_x() + width / 2
                y = bar.get_y() + height / 2
            else:
                width = bar.get_width()
                height = bar.get_height()
                x = bar.get_x() + width / 2
                y = bar.get_y() + height / 2
            
            # Format nilai
            text = format_str.format(value)
            
            # Tambahkan teks
            ax.text(
                x, y, text,
                ha='center', va='center',
                fontsize=fontsize,
                fontweight='bold',
                color='white'
            )
    
    def add_line_annotations(
        self, 
        ax: Axes, 
        x_values: List[Any],
        y_values: List[Union[int, float]],
        labels: Optional[List[str]] = None,
        fontsize: int = 9,
        offset: Tuple[float, float] = (0, 5),
        every_nth: int = 1,
        format_str: str = '{}'
    ) -> None:
        """
        Tambahkan anotasi untuk line chart.
        
        Args:
            ax: Axes matplotlib
            x_values: List nilai untuk sumbu x
            y_values: List nilai untuk sumbu y
            labels: List label untuk titik (opsional)
            fontsize: Ukuran font anotasi
            offset: Offset (x, y) untuk posisi teks
            every_nth: Frekuensi anotasi (setiap n titik)
            format_str: Format string untuk nilai
        """
        # Jika labels tidak disediakan, gunakan nilai y
        if labels is None:
            labels = [format_str.format(y) for y in y_values]
            
        # Tambahkan anotasi setiap n titik
        for i in range(0, len(x_values), every_nth):
            if i < len(y_values):
                x = x_values[i]
                y = y_values[i]
                label = labels[i]
                
                ax.annotate(
                    label,
                    (x, y),
                    xytext=(offset[0], offset[1]),
                    textcoords='offset points',
                    fontsize=fontsize,
                    ha='center'
                )
    
    def create_legend(
        self, 
        ax: Axes, 
        labels: List[str],
        colors: List[str],
        title: Optional[str] = None,
        loc: str = 'best',
        frameon: bool = True,
        fontsize: int = 10
    ) -> None:
        """
        Buat legenda kustom.
        
        Args:
            ax: Axes matplotlib
            labels: List label untuk item
            colors: List warna untuk item
            title: Judul legenda (opsional)
            loc: Lokasi legenda
            frameon: Apakah menampilkan border legenda
            fontsize: Ukuran font legenda
        """
        # Buat patches untuk legenda
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color, label=label)
                          for label, color in zip(labels, colors)]
        
        # Tambahkan legenda
        ax.legend(
            handles=legend_elements,
            title=title,
            loc=loc,
            frameon=frameon,
            fontsize=fontsize
        )
    
    def add_data_labels(
        self, 
        ax: Axes, 
        x_values: List[Any],
        y_values: List[Union[int, float]],
        labels: Optional[List[str]] = None,
        fontsize: int = 9,
        rotation: int = 0,
        ha: str = 'center',
        va: str = 'bottom'
    ) -> None:
        """
        Tambahkan label data di bawah sumbu x.
        
        Args:
            ax: Axes matplotlib
            x_values: List nilai untuk sumbu x
            y_values: List nilai untuk sumbu y
            labels: List label untuk ditampilkan (opsional)
            fontsize: Ukuran font label
            rotation: Rotasi label (derajat)
            ha: Horizontal alignment
            va: Vertical alignment
        """
        # Jika labels tidak disediakan, gunakan nilai x
        if labels is None:
            labels = [str(x) for x in x_values]
            
        # Set posisi label
        ax.set_xticks(x_values)
        ax.set_xticklabels(labels, fontsize=fontsize, rotation=rotation, ha=ha, va=va)
    
    def get_pie_autopct_func(
        self, 
        data: Dict[str, Union[int, float]],
        show_values: bool = True,
        show_percents: bool = True
    ) -> Optional[Union[str, Callable]]:
        """
        Dapatkan fungsi format untuk pie chart.
        
        Args:
            data: Dictionary data untuk pie chart {kategori: nilai}
            show_values: Apakah menampilkan nilai absolut
            show_percents: Apakah menampilkan persentase
            
        Returns:
            Format string atau fungsi format, None jika tidak ada yang ditampilkan
        """
        if not show_values and not show_percents:
            return None
            
        if show_values and show_percents:
            # Buat fungsi kustom
            def autopct_func(pct):
                total = sum(data.values())
                val = int(round(pct*total/100.0))
                return f'{pct:.1f}%\n({val})'
            return autopct_func
        elif show_values:
            # Buat fungsi yang hanya menampilkan nilai
            def val_only(pct):
                total = sum(data.values())
                val = int(round(pct*total/100.0))
                return f'{val}'
            return val_only
        else:  # show_percents only
            return '%1.1f%%'
    
    def add_text_box(
        self, 
        ax: Axes, 
        text: str,
        x: float = 0.5,
        y: float = 0.5,
        fontsize: int = 10,
        ha: str = 'center',
        va: str = 'center',
        bbox: Optional[Dict] = None
    ) -> None:
        """
        Tambahkan text box pada chart.
        
        Args:
            ax: Axes matplotlib
            text: Text untuk ditampilkan
            x, y: Posisi (0-1) dalam axes
            fontsize: Ukuran font
            ha, va: Horizontal dan vertical alignment
            bbox: Dictionary properti box (opsional)
        """
        # Siapkan default bbox style jika tidak disediakan
        if bbox is None:
            bbox = dict(boxstyle="round,pad=0.5", facecolor='#f5f5f5', alpha=0.5)
            
        # Tambahkan teks
        ax.text(
            x, y, text,
            transform=ax.transAxes,
            ha=ha, va=va,
            fontsize=fontsize,
            bbox=bbox
        )
    
    def add_annotated_heatmap(
        self, 
        ax: Axes, 
        data: np.ndarray,
        text_format: str = "{:.1f}",
        threshold: Optional[float] = None,
        cmap: Optional[Any] = None
    ) -> None:
        """
        Tambahkan anotasi nilai pada heatmap.
        
        Args:
            ax: Axes matplotlib
            data: Array 2D dengan data
            text_format: Format untuk nilai
            threshold: Batas untuk memilih warna teks (None = otomatis)
            cmap: Colormap untuk menentukan warna teks
        """
        # Ukuran data
        height, width = data.shape
        
        # Tentukan threshold otomatis jika tidak disediakan
        if threshold is None:
            threshold = (data.max() + data.min()) / 2
            
        # Tambahkan teks untuk setiap cell
        for i in range(height):
            for j in range(width):
                value = data[i, j]
                
                # Format nilai
                text = text_format.format(value)
                
                # Tentukan warna teks berdasarkan nilai
                if cmap:
                    text_color = 'white' if value > threshold else 'black'
                else:
                    text_color = 'white' if value > threshold else 'black'
                
                # Tambahkan teks
                ax.text(
                    j, i, text,
                    ha='center', va='center',
                    color=text_color,
                    fontweight='bold'
                )