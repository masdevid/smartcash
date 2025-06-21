# smartcash/common/visualization/core/visualization_base.py
"""
Kelas dasar untuk semua komponen visualisasi
"""

import matplotlib.pyplot as plt
import os
from typing import Optional, Tuple, Any

class VisualizationBase:
    def set_plot_style(self, style: str = 'default') -> None:
        """Set style untuk matplotlib plots"""
        plt.style.use(style)
    
    def save_figure(self, fig: plt.Figure, filepath: Optional[str] = None, 
                   dpi: int = 300) -> str:
        """Simpan figure matplotlib"""
        if filepath:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
            return filepath
        return ""
    
    def create_output_directory(self, output_dir: str) -> str:
        """Buat direktori output"""
        os.makedirs(output_dir, exist_ok=True)
        return output_dir