"""
File: smartcash/ui/components/dataset_distribution.py
Deskripsi: Komponen shared untuk menampilkan distribusi dataset dalam bentuk pie chart
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional, List, Tuple
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
import numpy as np
from smartcash.ui.utils.constants import ICONS, COLORS

def create_dataset_distribution(
    title: str = "Distribusi Dataset",
    description: str = "Visualisasi distribusi dataset dalam bentuk pie chart",
    distribution: Dict[str, int] = None,
    width: str = "100%",
    height: str = "400px",
    icon: str = "pie_chart",
    with_dummy_data: bool = False,
    colors: Optional[List[str]] = None,
    explode: Optional[List[float]] = None,
    show_values: bool = True,
    show_percentages: bool = True
) -> Dict[str, Any]:
    """
    Buat komponen distribusi dataset dalam bentuk pie chart yang dapat digunakan di berbagai modul.
    
    Args:
        title: Judul komponen
        description: Deskripsi komponen
        distribution: Dictionary berisi distribusi {label: count}
        width: Lebar komponen
        height: Tinggi komponen
        icon: Ikon untuk judul
        with_dummy_data: Tampilkan data dummy jika distribution tidak tersedia
        colors: List warna untuk pie chart
        explode: List nilai explode untuk pie chart
        show_values: Tampilkan nilai pada pie chart
        show_percentages: Tampilkan persentase pada pie chart
        
    Returns:
        Dictionary berisi komponen distribusi dataset
    """
    # Default distribution jika tidak disediakan
    if distribution is None and with_dummy_data:
        distribution = {
            "Train": 700,
            "Validation": 150,
            "Test": 150
        }
    
    # Tambahkan ikon jika tersedia
    display_title = title
    if icon and icon in ICONS:
        display_title = f"{ICONS[icon]} {title}"
    
    # Buat header untuk komponen
    header = widgets.HTML(
        value=f"<h4 style='margin-top: 5px; margin-bottom: 10px; color: {COLORS.get('dark', '#333')};'>{display_title}</h4>"
    )
    
    # Buat deskripsi jika ada
    description_widget = None
    if description:
        description_widget = widgets.HTML(
            value=f"<div style='margin-bottom: 10px; color: {COLORS.get('secondary', '#666')};'>{description}</div>"
        )
    
    # Buat output untuk pie chart
    chart_output = widgets.Output(
        layout=widgets.Layout(
            width=width,
            height=height,
            padding='10px'
        )
    )
    
    # Fungsi untuk membuat pie chart
    def _create_pie_chart(data=None):
        with chart_output:
            clear_output(wait=True)
            
            if data is None and distribution is None:
                if with_dummy_data:
                    data = {
                        "Train": 700,
                        "Validation": 150,
                        "Test": 150
                    }
                else:
                    display(widgets.HTML(
                        value=f"<div style='padding: 20px; text-align: center; color: {COLORS.get('secondary', '#666')};'>{ICONS.get('warning', '⚠️')} Tidak ada data distribusi yang tersedia.</div>"
                    ))
                    return
            
            chart_data = data if data is not None else distribution
            
            if not chart_data:
                display(widgets.HTML(
                    value=f"<div style='padding: 20px; text-align: center; color: {COLORS.get('secondary', '#666')};'>{ICONS.get('warning', '⚠️')} Tidak ada data distribusi yang tersedia.</div>"
                ))
                return
            
            try:
                # Siapkan data untuk pie chart
                labels = list(chart_data.keys())
                values = list(chart_data.values())
                
                # Siapkan warna
                chart_colors = colors
                if chart_colors is None:
                    chart_colors = plt.cm.tab10(np.arange(len(labels)) % 10)
                
                # Siapkan explode
                chart_explode = explode
                if chart_explode is None:
                    chart_explode = [0.05] * len(labels)
                
                # Buat pie chart
                plt.figure(figsize=(8, 8))
                wedges, texts, autotexts = plt.pie(
                    values, 
                    labels=None,  # Akan ditambahkan di legend
                    autopct='%1.1f%%' if show_percentages else None,
                    startangle=90,
                    shadow=True,
                    colors=chart_colors,
                    explode=chart_explode,
                    textprops={'color': 'white', 'weight': 'bold'}
                )
                
                # Tambahkan legend
                legend_labels = labels
                if show_values:
                    legend_labels = [f"{label} ({value})" for label, value in zip(labels, values)]
                
                plt.legend(
                    wedges, 
                    legend_labels,
                    title="Kategori",
                    loc="center left",
                    bbox_to_anchor=(1, 0, 0.5, 1)
                )
                
                plt.title(title)
                plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
                plt.tight_layout()
                plt.show()
                
                # Tambahkan pesan dummy jika perlu
                if with_dummy_data and data is None and distribution is None:
                    display(widgets.HTML(
                        value=f"<div style='margin-top: 10px; padding: 10px; background-color: {COLORS.get('warning', '#f39c12')}20; border-left: 4px solid {COLORS.get('warning', '#f39c12')}; border-radius: 4px;'>{ICONS.get('warning', '⚠️')} <b>Data Dummy:</b> Visualisasi ini menggunakan data dummy karena data aktual tidak tersedia.</div>"
                    ))
            except Exception as e:
                print(f"{ICONS.get('error', '❌')} Error saat membuat pie chart: {str(e)}")
                
                # Tampilkan data dummy jika error
                if with_dummy_data:
                    try:
                        dummy_data = {
                            "Train": 700,
                            "Validation": 150,
                            "Test": 150
                        }
                        
                        labels = list(dummy_data.keys())
                        values = list(dummy_data.values())
                        
                        plt.figure(figsize=(8, 8))
                        wedges, texts, autotexts = plt.pie(
                            values, 
                            labels=None,
                            autopct='%1.1f%%' if show_percentages else None,
                            startangle=90,
                            shadow=True,
                            textprops={'color': 'white', 'weight': 'bold'}
                        )
                        
                        legend_labels = labels
                        if show_values:
                            legend_labels = [f"{label} ({value})" for label, value in zip(labels, values)]
                        
                        plt.legend(
                            wedges, 
                            legend_labels,
                            title="Kategori",
                            loc="center left",
                            bbox_to_anchor=(1, 0, 0.5, 1)
                        )
                        
                        plt.title(f"{title} (Data Dummy)")
                        plt.axis('equal')
                        plt.tight_layout()
                        plt.show()
                        
                        display(widgets.HTML(
                            value=f"<div style='margin-top: 10px; padding: 10px; background-color: {COLORS.get('warning', '#f39c12')}20; border-left: 4px solid {COLORS.get('warning', '#f39c12')}; border-radius: 4px;'>{ICONS.get('warning', '⚠️')} <b>Data Dummy:</b> Visualisasi ini menggunakan data dummy karena terjadi error: {str(e)}</div>"
                        ))
                    except Exception as e2:
                        print(f"{ICONS.get('error', '❌')} Error saat membuat pie chart dummy: {str(e2)}")
    
    # Inisialisasi pie chart
    _create_pie_chart()
    
    # Fungsi untuk memperbarui pie chart
    def update_distribution(data=None):
        """
        Perbarui pie chart dengan data baru.
        
        Args:
            data: Data untuk pie chart {label: count}
        """
        _create_pie_chart(data)
    
    # Buat container untuk komponen
    widgets_list = [header]
    if description_widget:
        widgets_list.append(description_widget)
    widgets_list.append(chart_output)
    
    container = widgets.VBox(
        widgets_list,
        layout=widgets.Layout(
            margin='10px 0px',
            padding='10px',
            border='1px solid #eee',
            border_radius='4px',
            width=width
        )
    )
    
    return {
        'container': container,
        'chart_output': chart_output,
        'header': header,
        'update_distribution': update_distribution
    }
