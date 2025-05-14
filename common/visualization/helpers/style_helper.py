"""
File: smartcash/dataset/visualization/helpers/style_helper.py
Deskripsi: Utilitas untuk styling visualisasi dan manajemen tema
"""

import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any, Tuple, Union
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from smartcash.common.logger import get_logger


class StyleHelper:
    """Helper untuk styling visualisasi dan manajemen tema."""
    
    def __init__(self, logger=None):
        """
        Inisialisasi StyleHelper.
        
        Args:
            logger: Logger kustom (opsional)
        """
        self.logger = logger or get_logger("style_helper")
        
        # Style presets untuk reusable styling
        self.style_presets = {
            'default': {
                'style': 'seaborn-v0_8-whitegrid',
                'title_size': 14,
                'title_weight': 'bold',
                'label_size': 12,
                'tick_size': 10,
                'legend_size': 10,
                'grid': True,
                'grid_alpha': 0.3
            },
            'presentation': {
                'style': 'seaborn-v0_8-talk',
                'title_size': 16,
                'title_weight': 'bold',
                'label_size': 14,
                'tick_size': 12,
                'legend_size': 12,
                'grid': True,
                'grid_alpha': 0.3
            },
            'paper': {
                'style': 'seaborn-v0_8-white',
                'title_size': 12,
                'title_weight': 'bold',
                'label_size': 10,
                'tick_size': 9,
                'legend_size': 9,
                'grid': False,
                'grid_alpha': 0.15
            },
            'dark': {
                'style': 'dark_background',
                'title_size': 14,
                'title_weight': 'bold',
                'label_size': 12,
                'tick_size': 10,
                'legend_size': 10,
                'grid': True,
                'grid_alpha': 0.3
            }
        }
        
        # Tema default
        self.current_style = self.style_presets['default'].copy()
        
        self.logger.info("🎨 StyleHelper diinisialisasi dengan style default")
    
    def set_style(
        self, 
        style: Union[str, Dict[str, Any]] = 'default',
        custom_params: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Set style untuk visualisasi.
        
        Args:
            style: Nama preset, style matplotlib, atau dictionary parameter
            custom_params: Parameter tambahan untuk override
        """
        # Jika string, cek apakah preset atau style matplotlib
        if isinstance(style, str):
            if style in self.style_presets:
                # Gunakan style preset
                self.current_style = self.style_presets[style].copy()
                
                # Terapkan style matplotlib
                plt.style.use(self.current_style['style'])
                
                self.logger.info(f"🎨 Menggunakan style preset: {style}")
            else:
                # Anggap sebagai style matplotlib
                try:
                    plt.style.use(style)
                    self.current_style['style'] = style
                    self.logger.info(f"🎨 Menggunakan style matplotlib: {style}")
                except Exception as e:
                    self.logger.warning(f"⚠️ Style '{style}' tidak valid: {str(e)}")
                    # Gunakan default jika error
                    plt.style.use(self.style_presets['default']['style'])
                    self.current_style = self.style_presets['default'].copy()
                    
        elif isinstance(style, dict):
            # Update current style dengan dict yang diberikan
            self.current_style.update(style)
            
            # Terapkan style matplotlib jika ada
            if 'style' in style:
                try:
                    plt.style.use(self.current_style['style'])
                except Exception as e:
                    self.logger.warning(f"⚠️ Style '{style}' tidak valid: {str(e)}")
        
        # Update dengan custom params jika disediakan
        if custom_params:
            self.current_style.update(custom_params)
    
    def apply_style_to_figure(self, fig: Figure) -> None:
        """
        Terapkan style saat ini ke figure.
        
        Args:
            fig: Figure matplotlib
        """
        for ax in fig.axes:
            self.apply_style_to_axes(ax)
        
        fig.tight_layout()
    
    def apply_style_to_axes(self, ax: Axes) -> None:
        """
        Terapkan style saat ini ke axes.
        
        Args:
            ax: Axes matplotlib
        """
        # Set properties berdasarkan current style
        if ax.get_title():
            ax.set_title(
                ax.get_title(), 
                fontsize=self.current_style['title_size'],
                fontweight=self.current_style['title_weight']
            )
        
        if ax.get_xlabel():
            ax.set_xlabel(ax.get_xlabel(), fontsize=self.current_style['label_size'])
        
        if ax.get_ylabel():
            ax.set_ylabel(ax.get_ylabel(), fontsize=self.current_style['label_size'])
        
        # Set tick params
        ax.tick_params(labelsize=self.current_style['tick_size'])
        
        # Set grid
        ax.grid(self.current_style['grid'], alpha=self.current_style['grid_alpha'])
        
        # Update legend jika ada
        if ax.get_legend():
            ax.legend(fontsize=self.current_style['legend_size'])
    
    def set_title_and_labels(
        self, 
        ax: Axes, 
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None
    ) -> None:
        """
        Set judul dan label dengan style saat ini.
        
        Args:
            ax: Axes matplotlib
            title: Judul (opsional)
            xlabel: Label sumbu x (opsional)
            ylabel: Label sumbu y (opsional)
        """
        if title:
            ax.set_title(
                title, 
                fontsize=self.current_style['title_size'],
                fontweight=self.current_style['title_weight']
            )
            
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=self.current_style['label_size'])
            
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=self.current_style['label_size'])
    
    def get_current_style_params(self) -> Dict[str, Any]:
        """
        Dapatkan parameter style saat ini.
        
        Returns:
            Dictionary parameter style
        """
        return self.current_style.copy()
    
    def register_custom_style(self, name: str, params: Dict[str, Any]) -> None:
        """
        Daftarkan style kustom.
        
        Args:
            name: Nama untuk style
            params: Parameter style
        """
        if name in self.style_presets:
            self.logger.warning(f"⚠️ Menimpa style preset yang sudah ada: {name}")
            
        self.style_presets[name] = params.copy()
        self.logger.info(f"✅ Style kustom '{name}' berhasil didaftarkan")

