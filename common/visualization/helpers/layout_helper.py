
"""
File: smartcash/dataset/visualization/helpers/layout_helper.py
Deskripsi: Utilitas untuk pengaturan layout dan grid pada visualisasi
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from smartcash.common.logger import get_logger


class LayoutHelper:
    """Helper untuk pengaturan layout dan grid pada visualisasi."""
    
    def __init__(self, logger=None):
        """
        Inisialisasi LayoutHelper.
        
        Args:
            logger: Logger kustom (opsional)
        """
        self.logger = logger or get_logger()
        self.logger.info("📐 LayoutHelper diinisialisasi")
    
    def create_grid_layout(
        self, 
        nrows: int, 
        ncols: int,
        figsize: Tuple[int, int] = (10, 8),
        width_ratios: Optional[List[float]] = None,
        height_ratios: Optional[List[float]] = None,
        sharex: Union[bool, str] = False,
        sharey: Union[bool, str] = False
    ) -> Tuple[Figure, np.ndarray]:
        """
        Buat layout grid dengan GridSpec.
        
        Args:
            nrows: Jumlah baris
            ncols: Jumlah kolom
            figsize: Ukuran figure (width, height)
            width_ratios: Proporsi lebar kolom (opsional)
            height_ratios: Proporsi tinggi baris (opsional)
            sharex: Berbagi sumbu x
            sharey: Berbagi sumbu y
            
        Returns:
            Tuple (Figure, array of Axes)
        """
        # Buat figure dan gridspec
        fig = plt.figure(figsize=figsize)
        spec = gridspec.GridSpec(
            nrows=nrows, 
            ncols=ncols,
            width_ratios=width_ratios,
            height_ratios=height_ratios
        )
        
        # Buat axes untuk setiap cell
        axes = np.empty((nrows, ncols), dtype=object)
        
        for i in range(nrows):
            for j in range(ncols):
                if i == 0 and j == 0:
                    # First axes
                    axes[i, j] = fig.add_subplot(spec[i, j])
                else:
                    # Share axes if requested
                    if sharex == 'all' or (sharex == 'row' and j > 0):
                        sharex_ax = axes[i, 0]
                    elif sharex is True:
                        sharex_ax = axes[0, 0]
                    else:
                        sharex_ax = None
                        
                    if sharey == 'all' or (sharey == 'col' and i > 0):
                        sharey_ax = axes[0, j]
                    elif sharey is True:
                        sharey_ax = axes[0, 0]
                    else:
                        sharey_ax = None
                        
                    axes[i, j] = fig.add_subplot(spec[i, j], sharex=sharex_ax, sharey=sharey_ax)
        
        # Adjust layout untuk tight fit
        fig.tight_layout()
        
        return fig, axes
    
    def create_subplot_mosaic(
        self, 
        mosaic: List[List[str]],
        figsize: Tuple[int, int] = (10, 8),
        empty_sentinel: str = '.'
    ) -> Tuple[Figure, Dict[str, Axes]]:
        """
        Buat layout dengan subplot_mosaic.
        
        Args:
            mosaic: List of lists dengan layout identifier
            figsize: Ukuran figure (width, height)
            empty_sentinel: Karakter untuk cell kosong
            
        Returns:
            Tuple (Figure, dict mapping identifier ke Axes)
        """
        fig, axd = plt.subplot_mosaic(
            mosaic,
            figsize=figsize,
            empty_sentinel=empty_sentinel
        )
        
        # Adjust layout
        fig.tight_layout()
        
        return fig, axd
    
    def create_dashboard_layout(
        self, 
        layout_spec: List[Dict[str, Any]],
        figsize: Tuple[int, int] = (12, 10),
        height_ratios: Optional[List[float]] = None
    ) -> Tuple[Figure, Dict[str, Axes]]:
        """
        Buat layout dashboard kustom dengan GridSpec.
        
        Args:
            layout_spec: List spesifikasi layout
                [{
                    'name': 'panel_name',
                    'position': (row, col),
                    'size': (rowspan, colspan)
                }, ...]
            figsize: Ukuran figure (width, height)
            height_ratios: Proporsi tinggi baris (opsional)
            
        Returns:
            Tuple (Figure, dict mapping name ke Axes)
        """
        # Tentukan ukuran grid
        max_row = max(item['position'][0] + item['size'][0] - 1 for item in layout_spec)
        max_col = max(item['position'][1] + item['size'][1] - 1 for item in layout_spec)
        
        # Buat figure dan gridspec
        fig = plt.figure(figsize=figsize)
        spec = gridspec.GridSpec(
            nrows=max_row + 1,
            ncols=max_col + 1,
            height_ratios=height_ratios
        )
        
        # Buat axes untuk setiap panel
        axd = {}
        
        for item in layout_spec:
            name = item['name']
            row, col = item['position']
            rowspan, colspan = item['size']
            
            axd[name] = fig.add_subplot(spec[row:row+rowspan, col:col+colspan])
        
        # Adjust layout
        fig.tight_layout()
        
        return fig, axd
    
    def adjust_subplots_spacing(
        self, 
        fig: Figure,
        left: Optional[float] = None,
        right: Optional[float] = None,
        bottom: Optional[float] = None,
        top: Optional[float] = None,
        wspace: Optional[float] = None,
        hspace: Optional[float] = None
    ) -> None:
        """
        Sesuaikan spacing antara subplots.
        
        Args:
            fig: Figure matplotlib
            left, right, bottom, top: Margin (0.0-1.0)
            wspace, hspace: Spasi horisontal/vertikal antara subplots
        """
        fig.subplots_adjust(
            left=left,
            right=right,
            bottom=bottom,
            top=top,
            wspace=wspace,
            hspace=hspace
        )
    
    def add_colorbar(
        self, 
        fig: Figure, 
        mappable: Any,
        ax: Optional[Axes] = None,
        location: str = 'right',
        size: str = '5%',
        pad: float = 0.05,
        label: Optional[str] = None
    ) -> Axes:
        """
        Tambahkan colorbar ke figure.
        
        Args:
            fig: Figure matplotlib
            mappable: Mappable object (mis. hasil imshow(), contourf())
            ax: Axes untuk reference (jika None, gunakan semua axes)
            location: Lokasi colorbar ('right', 'left', 'top', 'bottom')
            size: Ukuran colorbar (string)
            pad: Padding antara axes dan colorbar
            label: Label untuk colorbar
            
        Returns:
            Axes colorbar
        """
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        
        # Jika ax tidak ditentukan, gunakan semua axes
        if ax is None:
            if len(fig.axes) == 0:
                self.logger.warning("⚠️ Tidak ada axes dalam figure")
                return None
            ax = fig.axes[0]
        
        # Buat axes untuk colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes(location, size=size, pad=pad)
        
        # Buat colorbar
        cbar = fig.colorbar(mappable, cax=cax)
        
        # Set label jika ada
        if label:
            cbar.set_label(label)
            
        return cax