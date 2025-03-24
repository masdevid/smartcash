"""
File: smartcash/ui/helpers/plot_base.py
Deskripsi: Kelas dasar dan utilitas umum untuk plotting distribusi kelas
"""

import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Any, Optional
from smartcash.ui.utils.constants import COLORS

def prepare_class_dataframe(class_distribution: Dict[str, int]) -> pd.DataFrame:
    """
    Buat DataFrame dari distribusi kelas untuk plotting.
    
    Args:
        class_distribution: Dictionary {class_id: jumlah_instance}
        
    Returns:
        DataFrame siap untuk plotting
    """
    df = pd.DataFrame({
        'Kelas': list(class_distribution.keys()),
        'Jumlah': list(class_distribution.values())
    }).sort_values('Kelas')
    return df

def create_figure(figsize=(10, 6)):
    """
    Buat figure dan axes dengan ukuran default.
    
    Args:
        figsize: Ukuran figure (width, height)
        
    Returns:
        Tuple (figure, axes)
    """
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax

def add_styling(ax, title, xlabel='Kelas', ylabel='Jumlah Instance'):
    """
    Tambahkan styling standard ke axes.
    
    Args:
        ax: matplotlib Axes
        title: Judul plot
        xlabel: Label sumbu X
        ylabel: Label sumbu Y
    """
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis='y', linestyle='--', alpha=0.7)