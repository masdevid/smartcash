"""
File: smartcash/ui/components/shared/metrics.py
Deskripsi: Komponen UI untuk menampilkan metrik dengan styling konsisten
"""

import ipywidgets as widgets
from typing import Union, Optional, Dict, Any
import pandas as pd
from IPython.display import display, HTML
import matplotlib.pyplot as plt

from smartcash.ui.utils.constants import COLORS

def create_metric_display(label: str, 
                         value: Union[int, float, str],
                         unit: Optional[str] = None,
                         is_good: Optional[bool] = None) -> widgets.HTML:
    """
    Buat display metrik dengan style konsisten.
    
    Args:
        label: Label metrik
        value: Nilai metrik (angka atau string)
        unit: Unit opsional (misalnya %, detik, dll)
        is_good: Flag opsional untuk indikasi positif/negatif
        
    Returns:
        Widget HTML berisi display metrik
    """
    # Tentukan warna berdasarkan nilai is_good
    if is_good is None:
        color = COLORS['dark']  # Neutral
    elif is_good:
        color = COLORS['success']  # Green for good
    else:
        color = COLORS['danger']  # Red for bad
    
    # Format nilai
    if isinstance(value, float):
        formatted_value = f"{value:.4f}"
    else:
        formatted_value = str(value)
        
    # Tambahkan unit jika ada
    if unit:
        formatted_value = f"{formatted_value} {unit}"
    
    # Buat HTML
    metric_html = f"""
    <div style="margin: 10px 5px; padding: 8px; background-color: {COLORS['light']}; 
                border-radius: 5px; text-align: center; min-width: 120px;">
        <div style="font-size: 0.9em; color: {COLORS['muted']};">{label}</div>
        <div style="font-size: 1.3em; font-weight: bold; color: {color};">{formatted_value}</div>
    </div>
    """
    
    return widgets.HTML(value=metric_html)

def create_result_table(
    data: Dict[str, Any],
    title: str = 'Results',
    highlight_max: bool = True
) -> None:
    """
    Menampilkan table hasil dengan highlighting.
    
    Args:
        data: Dictionary data untuk tabel
        title: Judul tabel
        highlight_max: Highlight nilai maksimum di setiap kolom
    """
    # Konversi ke DataFrame
    df = pd.DataFrame(data)
    
    # Display judul
    display(HTML(f"<h3>{title}</h3>"))
    
    # Display tabel dengan styling
    if highlight_max:
        display(df.style.highlight_max(axis=0, color='lightgreen'))
    else:
        display(df)

def plot_statistics(
    data: pd.DataFrame, 
    title: str, 
    kind: str = 'bar', 
    figsize=(10, 6),
    **kwargs
) -> None:
    """
    Plot statistik data.
    
    Args:
        data: DataFrame berisi data
        title: Judul plot
        kind: Jenis plot (bar, line, dll)
        figsize: Ukuran gambar
        **kwargs: Parameter tambahan untuk plot
    """
    plt.figure(figsize=figsize)
    
    data.plot(kind=kind, **kwargs)
    
    plt.title(title)
    plt.tight_layout()
    plt.show()

def styled_html(content: str, bg_color: str = "#f8f9fa", text_color: str = "#2c3e50", 
              border_color: Optional[str] = None, padding: int = 10, margin: int = 10) -> widgets.HTML:
    """
    Buat HTML dengan styling kustom.
    
    Args:
        content: Konten HTML
        bg_color: Warna background
        text_color: Warna teks
        border_color: Warna border (opsional)
        padding: Padding dalam piksel
        margin: Margin dalam piksel
        
    Returns:
        Widget HTML dengan styling kustom
    """
    border_style = f"border-left: 4px solid {border_color}; " if border_color else ""
    
    return widgets.HTML(f"""
    <div style="background-color: {bg_color}; color: {text_color}; 
                {border_style}padding: {padding}px; margin: {margin}px 0; 
                border-radius: 4px;">
        {content}
    </div>
    """)