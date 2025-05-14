"""
File: smartcash/ui/helpers/plot_comparison.py
Deskripsi: Utilitas untuk plotting perbandingan distribusi kelas
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict

from smartcash.ui.utils.constants import COLORS
from smartcash.ui.charts.plot_base import create_figure, add_styling

def plot_class_distribution_comparison(
    original_distribution: Dict[str, int],
    augmented_distribution: Dict[str, int],
    title: str = "Perbandingan Distribusi Kelas",
    display_numbers: bool = True
) -> plt.Figure:
    """
    Plot perbandingan distribusi kelas asli dan augmentasi.
    
    Args:
        original_distribution: Dictionary {class_id: jumlah_instance} data asli
        augmented_distribution: Dictionary {class_id: jumlah_instance} data augmentasi
        title: Judul plot
        display_numbers: Tampilkan angka di atas bar
        
    Returns:
        Matplotlib Figure
    """
    # Gabungkan semua kelas
    all_classes = sorted(set(list(original_distribution.keys()) + list(augmented_distribution.keys())))
    
    # Buat DataFrame
    df = pd.DataFrame({
        'Kelas': all_classes,
        'Asli': [original_distribution.get(cls, 0) for cls in all_classes],
        'Augmentasi': [augmented_distribution.get(cls, 0) for cls in all_classes]
    })
    
    # Buat plot
    fig, ax = create_figure(figsize=(12, 6))
    
    # Posisi bar
    x = np.arange(len(all_classes))
    width = 0.35
    
    # Plot bars
    bars1 = ax.bar(x - width/2, df['Asli'], width, label='Data Asli', color=COLORS.get('primary', '#3498db'))
    bars2 = ax.bar(x + width/2, df['Augmentasi'], width, label='Data Augmentasi', color=COLORS.get('warning', '#f39c12'))
    
    # Tambahkan angka di atas bar
    if display_numbers:
        for bars, offset in [(bars1, -width/2), (bars2, width/2)]:
            for i, bar in enumerate(bars):
                height = bar.get_height()
                if height > 0:
                    ax.annotate(f'{int(height)}',
                                xy=(i + offset, height),
                                xytext=(0, 3),  # 3 points vertical offset
                                textcoords="offset points",
                                ha='center', va='bottom',
                                fontweight='bold')
    
    # Styling
    add_styling(ax, title)
    ax.set_xticks(x)
    ax.set_xticklabels(df['Kelas'])
    ax.legend()
    
    plt.tight_layout()
    return fig