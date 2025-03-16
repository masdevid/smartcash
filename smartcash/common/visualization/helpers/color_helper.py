"""
File: smartcash/dataset/visualization/helpers/color_helper.py
Deskripsi: Utilitas untuk manajemen warna, palet, dan gradien warna
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from typing import Dict, List, Tuple, Union, Optional, Any

from smartcash.common.logger import get_logger


class ColorHelper:
    """Helper untuk manajemen palet warna dan gradien."""
    
    def __init__(self, logger=None):
        """
        Inisialisasi ColorHelper.
        
        Args:
            logger: Logger kustom (opsional)
        """
        self.logger = logger or get_logger("color_helper")
        
        # Palet warna default untuk berbagai kasus penggunaan
        self.palettes = {
            'categorical': 'tab10',    # Warna kategori berbeda
            'sequential': 'viridis',   # Gradasi untuk nilai berurutan
            'diverging': 'coolwarm',   # Untuk nilai dengan titik tengah (0)
            'emphasis': 'Set2',        # Untuk highlight beberapa kategori
            'pastel': 'Pastel1',       # Warna pastel lembut
            'dark': 'dark:salmon_r'    # Versi gelap untuk latar belakang gelap
        }
        
        # Mapping kategori spesifik yang bisa digunakan aplikasi
        self.semantic_colors = {
            'success': '#4CAF50',  # Hijau
            'warning': '#FFC107',  # Kuning
            'error': '#F44336',    # Merah
            'info': '#2196F3',     # Biru
            'neutral': '#9E9E9E'   # Abu-abu
        }
        
        self.logger.info("🎨 ColorHelper diinisialisasi dengan palet warna default")
    
    def get_color_palette(
        self, 
        n_colors: int, 
        palette_name: Optional[str] = None,
        as_hex: bool = False,
        desat: float = None
    ) -> List[Union[Tuple[float, float, float], str]]:
        """
        Dapatkan palet warna dengan jumlah warna tertentu.
        
        Args:
            n_colors: Jumlah warna yang dibutuhkan
            palette_name: Nama palet warna (seaborn atau matplotlib)
            as_hex: Apakah mengembalikan format hex (#RRGGBB)
            desat: Faktor desaturasi (0-1)
            
        Returns:
            List berisi warna RGB atau hex
        """
        # Gunakan palet default jika tidak disebutkan
        palette_name = palette_name or self.palettes['categorical']
        
        # Gunakan palet seaborn
        try:
            colors = sns.color_palette(palette_name, n_colors, desat=desat)
            
            # Konversi ke hex jika diminta
            if as_hex:
                colors = [mcolors.rgb2hex(c) for c in colors]
                
            return colors
        except Exception as e:
            self.logger.warning(f"⚠️ Error saat mengambil palet '{palette_name}': {str(e)}")
            # Fallback ke palet default
            colors = sns.color_palette(self.palettes['categorical'], n_colors)
            if as_hex:
                colors = [mcolors.rgb2hex(c) for c in colors]
            return colors
    
    def create_color_mapping(
        self, 
        categories: List[str],
        palette: Optional[str] = None,
        as_hex: bool = False
    ) -> Dict[str, Union[Tuple[float, float, float], str]]:
        """
        Buat mapping kategori ke warna.
        
        Args:
            categories: List kategori
            palette: Nama palet warna
            as_hex: Apakah mengembalikan format hex (#RRGGBB)
            
        Returns:
            Dictionary berisi mapping {kategori: warna}
        """
        # Dapatkan palet warna
        colors = self.get_color_palette(len(categories), palette, as_hex)
        
        # Buat mapping
        return dict(zip(categories, colors))
    
    def generate_gradient(
        self, 
        start_color: Union[str, Tuple[float, float, float]],
        end_color: Union[str, Tuple[float, float, float]],
        steps: int,
        as_hex: bool = False
    ) -> List[Union[Tuple[float, float, float], str]]:
        """
        Buat gradien warna dari dua warna.
        
        Args:
            start_color: Warna awal (RGB atau hex)
            end_color: Warna akhir (RGB atau hex)
            steps: Jumlah langkah gradien
            as_hex: Apakah mengembalikan format hex (#RRGGBB)
            
        Returns:
            List berisi warna gradien
        """
        # Konversi ke RGB jika dalam format hex
        if isinstance(start_color, str) and start_color.startswith('#'):
            start_color = mcolors.hex2color(start_color)
            
        if isinstance(end_color, str) and end_color.startswith('#'):
            end_color = mcolors.hex2color(end_color)
            
        # Buat gradien
        gradient = []
        for i in range(steps):
            r = start_color[0] + (end_color[0] - start_color[0]) * (i / (steps - 1))
            g = start_color[1] + (end_color[1] - start_color[1]) * (i / (steps - 1))
            b = start_color[2] + (end_color[2] - start_color[2]) * (i / (steps - 1))
            
            if as_hex:
                gradient.append(mcolors.rgb2hex((r, g, b)))
            else:
                gradient.append((r, g, b))
                
        return gradient
    
    def create_cmap(
        self,
        colors: List[Union[str, Tuple[float, float, float]]],
        name: str = 'custom_cmap'
    ) -> mcolors.LinearSegmentedColormap:
        """
        Buat colormap kustom dari list warna.
        
        Args:
            colors: List warna (RGB atau hex)
            name: Nama colormap
            
        Returns:
            Colormap kustom
        """
        # Konversi hex ke RGB jika perlu
        rgb_colors = []
        for color in colors:
            if isinstance(color, str) and color.startswith('#'):
                rgb_colors.append(mcolors.hex2color(color))
            else:
                rgb_colors.append(color)
        
        # Buat colormap
        return mcolors.LinearSegmentedColormap.from_list(name, rgb_colors)
    
    def get_color_for_value(
        self,
        value: float,
        vmin: float,
        vmax: float,
        cmap_name: str = 'viridis',
        as_hex: bool = False
    ) -> Union[Tuple[float, float, float], str]:
        """
        Dapatkan warna untuk nilai dalam range.
        
        Args:
            value: Nilai yang akan dipetakan ke warna
            vmin: Nilai minimum range
            vmax: Nilai maksimum range
            cmap_name: Nama colormap
            as_hex: Apakah mengembalikan format hex (#RRGGBB)
            
        Returns:
            Warna untuk nilai tersebut
        """
        # Normalisasi nilai ke range 0-1
        if vmax > vmin:
            normalized = (value - vmin) / (vmax - vmin)
        else:
            normalized = 0.5  # Default jika range tidak valid
            
        # Clip nilai ke range 0-1
        normalized = max(0, min(1, normalized))
        
        # Dapatkan colormap
        cmap = plt.get_cmap(cmap_name)
        
        # Dapatkan warna
        color = cmap(normalized)
        
        # Konversi ke hex jika diminta
        if as_hex:
            return mcolors.rgb2hex(color[:3])
        
        return color[:3]  # RGB tuple
    
    def get_semantic_color(self, key: str, as_hex: bool = True) -> Union[str, Tuple[float, float, float]]:
        """
        Dapatkan warna semantik berdasarkan kunci.
        
        Args:
            key: Kunci warna semantik ('success', 'warning', 'error', dll)
            as_hex: Apakah mengembalikan format hex (#RRGGBB)
            
        Returns:
            Warna semantik
        """
        # Dapatkan warna hex dari mapping
        hex_color = self.semantic_colors.get(key, self.semantic_colors['neutral'])
        
        if as_hex:
            return hex_color
        
        # Konversi ke RGB jika diminta
        return mcolors.hex2color(hex_color)