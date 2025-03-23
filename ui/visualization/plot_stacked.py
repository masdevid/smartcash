"""
File: smartcash/ui/helpers/plot_stacked.py
Deskripsi: Utilitas untuk plotting distribusi kelas dengan stacked bar
"""

import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict

from smartcash.ui.utils.constants import COLORS
from smartcash.ui.helpers.plot_base import create_figure, add_styling

def plot_class_distribution_stacked(
    original_distribution: Dict[str, int],
    augmented_distribution: Dict[str, int],
    title: str = "Distribusi Kelas (Total)",
    display_numbers: bool = True
) -> plt.Figure:
    """
    Plot distribusi kelas dengan stacked bar (asli + augmentasi).
    
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
    
    # Hitung total
    df['Total'] = df['Asli'] + df['Augmentasi']
    
    # Buat plot
    fig, ax = create_figure(figsize=(12, 6))
    
    # Plot stacked bars
    bars1 = ax.bar(df['Kelas'].astype(str), df['Asli'], label='Data Asli', color=COLORS.get('primary', '#3498db'))
    bars2 = ax.bar(df['Kelas'].astype(str), df['Augmentasi'], bottom=df['Asli'], 
                  label='Data Augmentasi', color=COLORS.get('warning', '#f39c12'))
    
    # Tambahkan angka total di atas bar
    if display_numbers:
        for i, (orig, aug) in enumerate(zip(df['Asli'], df['Augmentasi'])):
            total = orig + aug
            # Only show number if there's a non-zero value
            if total > 0:
                ax.annotate(f'{int(total)}',
                            xy=(i, total),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom',
                            fontweight='bold')
    
    # Styling
    add_styling(ax, title)
    ax.legend()
    
    plt.tight_layout()
    return fig