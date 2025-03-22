"""
File: smartcash/ui/helpers/plot_single.py
Deskripsi: Utilitas untuk plotting distribusi kelas tunggal
"""

import matplotlib.pyplot as plt
from typing import Dict

from smartcash.ui.utils.constants import COLORS
from smartcash.ui.dataset.visualization_helpers.plot_base import (
    prepare_class_dataframe, create_figure, add_styling
)

def plot_class_distribution(
    class_distribution: Dict[str, int],
    title: str = "Distribusi Kelas",
    display_numbers: bool = True,
    color: str = None
) -> plt.Figure:
    """
    Plot distribusi kelas sebagai bar chart.
    
    Args:
        class_distribution: Dictionary {class_id: jumlah_instance}
        title: Judul plot
        display_numbers: Tampilkan angka di atas bar
        color: Warna bar
        
    Returns:
        Matplotlib Figure
    """
    # Convert ke DataFrame untuk plotting
    df = prepare_class_dataframe(class_distribution)
    
    # Buat plot
    fig, ax = create_figure()
    bars = ax.bar(df['Kelas'].astype(str), df['Jumlah'], color=color or COLORS.get('primary', '#3498db'))
    
    # Tambahkan angka di atas bar
    if display_numbers:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{int(height)}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontweight='bold')
    
    # Styling
    add_styling(ax, title)
    
    plt.tight_layout()
    return fig