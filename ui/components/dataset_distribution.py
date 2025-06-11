"""
File: smartcash/ui/components/dataset_distribution.py
Deskripsi: Komponen shared untuk menampilkan distribusi dataset dengan one-liner style
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional, List
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
import numpy as np
from smartcash.ui.utils.constants import ICONS, COLORS

def create_dataset_distribution(title: str = "Distribusi Dataset", description: str = "Visualisasi distribusi dataset dalam bentuk pie chart",
                               distribution: Dict[str, int] = None, width: str = "100%", height: str = "400px",
                               icon: str = "pie_chart", with_dummy_data: bool = False, colors: Optional[List[str]] = None,
                               explode: Optional[List[float]] = None, show_values: bool = True, show_percentages: bool = True) -> Dict[str, Any]:
    """Buat komponen distribusi dataset dengan one-liner style."""
    distribution = distribution or ({"Train": 700, "Validation": 150, "Test": 150} if with_dummy_data else None)
    display_title = f"{ICONS.get(icon, '')} {title}" if icon and icon in ICONS else title
    
    header = widgets.HTML(f"<h4 style='margin-top: 5px; margin-bottom: 10px; color: {COLORS.get('dark', '#333')};'>{display_title}</h4>")
    description_widget = widgets.HTML(f"<div style='margin-bottom: 10px; color: {COLORS.get('secondary', '#666')};'>{description}</div>") if description else None
    chart_output = widgets.Output(layout=widgets.Layout(width=width, height=height, padding='10px'))
    
    def _create_pie_chart(data=None):
        with chart_output:
            clear_output(wait=True)
            chart_data = data or distribution or ({"Train": 700, "Validation": 150, "Test": 150} if with_dummy_data else None)
            
            if not chart_data:
                return display(widgets.HTML(f"<div style='padding: 20px; text-align: center; color: {COLORS.get('secondary', '#666')};'>{ICONS.get('warning', '⚠️')} Tidak ada data distribusi yang tersedia.</div>"))
            
            try:
                labels, values = list(chart_data.keys()), list(chart_data.values())
                chart_colors = colors or plt.cm.tab10(np.arange(len(labels)) % 10)
                chart_explode = explode or [0.05] * len(labels)
                
                plt.figure(figsize=(8, 8))
                wedges, texts, autotexts = plt.pie(values, labels=None, autopct='%1.1f%%' if show_percentages else None,
                                                  startangle=90, shadow=True, colors=chart_colors, explode=chart_explode,
                                                  textprops={'color': 'white', 'weight': 'bold'})
                
                legend_labels = [f"{label} ({value})" for label, value in zip(labels, values)] if show_values else labels
                plt.legend(wedges, legend_labels, title="Kategori", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
                plt.title(title), plt.axis('equal'), plt.tight_layout(), plt.show()
                
                (with_dummy_data and data is None and distribution is None) and display(widgets.HTML(
                    f"<div style='margin-top: 10px; padding: 10px; background-color: {COLORS.get('warning', '#f39c12')}20; border-left: 4px solid {COLORS.get('warning', '#f39c12')}; border-radius: 4px;'>{ICONS.get('warning', '⚠️')} <b>Data Dummy:</b> Visualisasi ini menggunakan data dummy karena data aktual tidak tersedia.</div>"))
            
            except Exception as e:
                print(f"{ICONS.get('error', '❌')} Error saat membuat pie chart: {str(e)}")
                if with_dummy_data:
                    try:
                        dummy_data = {"Train": 700, "Validation": 150, "Test": 150}
                        labels, values = list(dummy_data.keys()), list(dummy_data.values())
                        plt.figure(figsize=(8, 8))
                        wedges, texts, autotexts = plt.pie(values, labels=None, autopct='%1.1f%%' if show_percentages else None,
                                                          startangle=90, shadow=True, textprops={'color': 'white', 'weight': 'bold'})
                        legend_labels = [f"{label} ({value})" for label, value in zip(labels, values)] if show_values else labels
                        plt.legend(wedges, legend_labels, title="Kategori", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
                        plt.title(f"{title} (Data Dummy)"), plt.axis('equal'), plt.tight_layout(), plt.show()
                        display(widgets.HTML(f"<div style='margin-top: 10px; padding: 10px; background-color: {COLORS.get('warning', '#f39c12')}20; border-left: 4px solid {COLORS.get('warning', '#f39c12')}; border-radius: 4px;'>{ICONS.get('warning', '⚠️')} <b>Data Dummy:</b> Visualisasi ini menggunakan data dummy karena terjadi error: {str(e)}</div>"))
                    except Exception as e2:
                        print(f"{ICONS.get('error', '❌')} Error saat membuat pie chart dummy: {str(e2)}")
    
    _create_pie_chart()
    widgets_list = [header] + ([description_widget] if description_widget else []) + [chart_output]
    container = widgets.VBox(widgets_list, layout=widgets.Layout(margin='10px 0px', padding='10px', border='1px solid #eee', border_radius='4px', width=width))
    
    return {'container': container, 'chart_output': chart_output, 'header': header, 'update_distribution': _create_pie_chart}
