"""
File: smartcash/ui/components/visualization_tabs.py
Deskripsi: Komponen shared untuk visualisasi dataset dalam bentuk tab
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional, List, Tuple
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
import numpy as np
from smartcash.ui.utils.constants import ICONS, COLORS
from smartcash.ui.components.tab_factory import create_tab_widget

def create_visualization_tabs(
    title: str = "Visualisasi Dataset",
    description: str = "Visualisasi distribusi data pada dataset",
    width: str = "100%",
    height: str = "400px",
    icon: str = "chart",
    with_dummy_data: bool = False,
    dataset_info: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Buat komponen visualisasi dataset dalam bentuk tab yang dapat digunakan di berbagai modul.
    
    Args:
        title: Judul komponen
        description: Deskripsi komponen
        width: Lebar komponen
        height: Tinggi komponen
        icon: Ikon untuk judul
        with_dummy_data: Tampilkan data dummy jika dataset_info tidak tersedia
        dataset_info: Informasi dataset untuk visualisasi
        
    Returns:
        Dictionary berisi komponen visualisasi dataset
    """
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
    
    # Buat output untuk setiap tab
    class_distribution_output = widgets.Output(
        layout=widgets.Layout(
            width=width,
            height=height,
            padding='10px'
        )
    )
    
    split_distribution_output = widgets.Output(
        layout=widgets.Layout(
            width=width,
            height=height,
            padding='10px'
        )
    )
    
    layer_distribution_output = widgets.Output(
        layout=widgets.Layout(
            width=width,
            height=height,
            padding='10px'
        )
    )
    
    heatmap_output = widgets.Output(
        layout=widgets.Layout(
            width=width,
            height=height,
            padding='10px'
        )
    )
    
    # Buat tab untuk visualisasi
    tab_items = [
        ("Distribusi Kelas", class_distribution_output),
        ("Distribusi Split", split_distribution_output),
        ("Distribusi Layer", layer_distribution_output),
        ("Heatmap", heatmap_output)
    ]
    
    tabs = create_tab_widget(tab_items)
    
    # Fungsi untuk membuat visualisasi dummy
    def _create_dummy_class_distribution():
        with class_distribution_output:
            clear_output(wait=True)
            plt.figure(figsize=(10, 6))
            classes = ['Rp1000', 'Rp2000', 'Rp5000', 'Rp10000', 'Rp20000', 'Rp50000', 'Rp100000']
            counts = [120, 80, 150, 200, 100, 90, 110]
            colors = plt.cm.viridis(np.linspace(0, 1, len(classes)))
            
            plt.bar(classes, counts, color=colors)
            plt.title('Distribusi Kelas (Data Dummy)')
            plt.xlabel('Kelas')
            plt.ylabel('Jumlah')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
            
            # Tambahkan pesan dummy
            display(widgets.HTML(
                value=f"<div style='margin-top: 10px; padding: 10px; background-color: {COLORS.get('warning', '#f39c12')}20; border-left: 4px solid {COLORS.get('warning', '#f39c12')}; border-radius: 4px;'>{ICONS.get('warning', '⚠️')} <b>Data Dummy:</b> Visualisasi ini menggunakan data dummy karena dataset aktual tidak tersedia.</div>"
            ))
    
    def _create_dummy_split_distribution():
        with split_distribution_output:
            clear_output(wait=True)
            plt.figure(figsize=(10, 6))
            splits = ['Train', 'Validation', 'Test']
            counts = [700, 150, 150]
            colors = [COLORS.get('primary', '#3498db'), COLORS.get('warning', '#f39c12'), COLORS.get('info', '#2980b9')]
            
            plt.bar(splits, counts, color=colors)
            plt.title('Distribusi Split (Data Dummy)')
            plt.xlabel('Split')
            plt.ylabel('Jumlah')
            plt.tight_layout()
            plt.show()
            
            # Tambahkan pesan dummy
            display(widgets.HTML(
                value=f"<div style='margin-top: 10px; padding: 10px; background-color: {COLORS.get('warning', '#f39c12')}20; border-left: 4px solid {COLORS.get('warning', '#f39c12')}; border-radius: 4px;'>{ICONS.get('warning', '⚠️')} <b>Data Dummy:</b> Visualisasi ini menggunakan data dummy karena dataset aktual tidak tersedia.</div>"
            ))
    
    def _create_dummy_layer_distribution():
        with layer_distribution_output:
            clear_output(wait=True)
            plt.figure(figsize=(10, 6))
            layers = ['Original', 'Preprocessed', 'Augmented']
            train = [500, 450, 700]
            val = [100, 90, 150]
            test = [100, 90, 150]
            
            x = np.arange(len(layers))
            width = 0.25
            
            plt.bar(x - width, train, width, label='Train', color=COLORS.get('primary', '#3498db'))
            plt.bar(x, val, width, label='Validation', color=COLORS.get('warning', '#f39c12'))
            plt.bar(x + width, test, width, label='Test', color=COLORS.get('info', '#2980b9'))
            
            plt.title('Distribusi Layer per Split (Data Dummy)')
            plt.xlabel('Layer')
            plt.ylabel('Jumlah')
            plt.xticks(x, layers)
            plt.legend()
            plt.tight_layout()
            plt.show()
            
            # Tambahkan pesan dummy
            display(widgets.HTML(
                value=f"<div style='margin-top: 10px; padding: 10px; background-color: {COLORS.get('warning', '#f39c12')}20; border-left: 4px solid {COLORS.get('warning', '#f39c12')}; border-radius: 4px;'>{ICONS.get('warning', '⚠️')} <b>Data Dummy:</b> Visualisasi ini menggunakan data dummy karena dataset aktual tidak tersedia.</div>"
            ))
    
    def _create_dummy_heatmap():
        with heatmap_output:
            clear_output(wait=True)
            plt.figure(figsize=(10, 6))
            
            # Buat data dummy untuk heatmap
            classes = ['Rp1000', 'Rp2000', 'Rp5000', 'Rp10000', 'Rp20000', 'Rp50000', 'Rp100000']
            splits = ['Train', 'Validation', 'Test']
            
            data = np.random.randint(10, 100, size=(len(splits), len(classes)))
            
            plt.imshow(data, cmap='viridis')
            plt.colorbar(label='Jumlah')
            plt.title('Heatmap Distribusi Kelas per Split (Data Dummy)')
            plt.xlabel('Kelas')
            plt.ylabel('Split')
            plt.xticks(np.arange(len(classes)), classes, rotation=45)
            plt.yticks(np.arange(len(splits)), splits)
            
            # Tambahkan nilai ke setiap sel
            for i in range(len(splits)):
                for j in range(len(classes)):
                    plt.text(j, i, data[i, j], ha='center', va='center', color='white')
            
            plt.tight_layout()
            plt.show()
            
            # Tambahkan pesan dummy
            display(widgets.HTML(
                value=f"<div style='margin-top: 10px; padding: 10px; background-color: {COLORS.get('warning', '#f39c12')}20; border-left: 4px solid {COLORS.get('warning', '#f39c12')}; border-radius: 4px;'>{ICONS.get('warning', '⚠️')} <b>Data Dummy:</b> Visualisasi ini menggunakan data dummy karena dataset aktual tidak tersedia.</div>"
            ))
    
    # Fungsi untuk membuat visualisasi dari dataset_info
    def _create_class_distribution(data):
        with class_distribution_output:
            clear_output(wait=True)
            if not data or 'class_distribution' not in data:
                _create_dummy_class_distribution()
                return
            
            try:
                class_dist = data['class_distribution']
                classes = list(class_dist.keys())
                counts = list(class_dist.values())
                
                plt.figure(figsize=(10, 6))
                colors = plt.cm.viridis(np.linspace(0, 1, len(classes)))
                
                plt.bar(classes, counts, color=colors)
                plt.title('Distribusi Kelas')
                plt.xlabel('Kelas')
                plt.ylabel('Jumlah')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.show()
            except Exception as e:
                print(f"{ICONS.get('error', '❌')} Error saat membuat visualisasi distribusi kelas: {str(e)}")
                _create_dummy_class_distribution()
    
    def _create_split_distribution(data):
        with split_distribution_output:
            clear_output(wait=True)
            if not data or 'split_distribution' not in data:
                _create_dummy_split_distribution()
                return
            
            try:
                split_dist = data['split_distribution']
                splits = list(split_dist.keys())
                counts = list(split_dist.values())
                
                plt.figure(figsize=(10, 6))
                colors = [COLORS.get('primary', '#3498db'), COLORS.get('warning', '#f39c12'), COLORS.get('info', '#2980b9')]
                
                plt.bar(splits, counts, color=colors[:len(splits)])
                plt.title('Distribusi Split')
                plt.xlabel('Split')
                plt.ylabel('Jumlah')
                plt.tight_layout()
                plt.show()
            except Exception as e:
                print(f"{ICONS.get('error', '❌')} Error saat membuat visualisasi distribusi split: {str(e)}")
                _create_dummy_split_distribution()
    
    def _create_layer_distribution(data):
        with layer_distribution_output:
            clear_output(wait=True)
            if not data or 'layer_distribution' not in data:
                _create_dummy_layer_distribution()
                return
            
            try:
                layer_dist = data['layer_distribution']
                layers = list(layer_dist.keys())
                splits = list(layer_dist[layers[0]].keys())
                
                x = np.arange(len(layers))
                width = 0.8 / len(splits)
                
                plt.figure(figsize=(10, 6))
                
                for i, split in enumerate(splits):
                    values = [layer_dist[layer][split] for layer in layers]
                    offset = (i - len(splits) / 2 + 0.5) * width
                    plt.bar(x + offset, values, width, label=split, 
                           color=[COLORS.get('primary', '#3498db'), COLORS.get('warning', '#f39c12'), 
                                 COLORS.get('info', '#2980b9')][i % 3])
                
                plt.title('Distribusi Layer per Split')
                plt.xlabel('Layer')
                plt.ylabel('Jumlah')
                plt.xticks(x, layers)
                plt.legend()
                plt.tight_layout()
                plt.show()
            except Exception as e:
                print(f"{ICONS.get('error', '❌')} Error saat membuat visualisasi distribusi layer: {str(e)}")
                _create_dummy_layer_distribution()
    
    def _create_heatmap(data):
        with heatmap_output:
            clear_output(wait=True)
            if not data or 'class_split_matrix' not in data:
                _create_dummy_heatmap()
                return
            
            try:
                matrix = data['class_split_matrix']
                classes = list(matrix['classes'])
                splits = list(matrix['splits'])
                values = np.array(matrix['values'])
                
                plt.figure(figsize=(10, 6))
                plt.imshow(values, cmap='viridis')
                plt.colorbar(label='Jumlah')
                plt.title('Heatmap Distribusi Kelas per Split')
                plt.xlabel('Kelas')
                plt.ylabel('Split')
                plt.xticks(np.arange(len(classes)), classes, rotation=45)
                plt.yticks(np.arange(len(splits)), splits)
                
                # Tambahkan nilai ke setiap sel
                for i in range(len(splits)):
                    for j in range(len(classes)):
                        plt.text(j, i, values[i, j], ha='center', va='center', color='white')
                
                plt.tight_layout()
                plt.show()
            except Exception as e:
                print(f"{ICONS.get('error', '❌')} Error saat membuat visualisasi heatmap: {str(e)}")
                _create_dummy_heatmap()
    
    # Fungsi untuk memperbarui visualisasi
    def update_visualizations(data=None):
        """
        Perbarui visualisasi dengan data baru.
        
        Args:
            data: Data untuk visualisasi
        """
        if data is None and dataset_info is None:
            if with_dummy_data:
                _create_dummy_class_distribution()
                _create_dummy_split_distribution()
                _create_dummy_layer_distribution()
                _create_dummy_heatmap()
            return
        
        viz_data = data if data is not None else dataset_info
        
        _create_class_distribution(viz_data)
        _create_split_distribution(viz_data)
        _create_layer_distribution(viz_data)
        _create_heatmap(viz_data)
    
    # Inisialisasi visualisasi
    if with_dummy_data and dataset_info is None:
        update_visualizations()
    elif dataset_info is not None:
        update_visualizations(dataset_info)
    
    # Buat container untuk komponen
    widgets_list = [header]
    if description_widget:
        widgets_list.append(description_widget)
    widgets_list.append(tabs)
    
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
        'tabs': tabs,
        'header': header,
        'outputs': {
            'class_distribution': class_distribution_output,
            'split_distribution': split_distribution_output,
            'layer_distribution': layer_distribution_output,
            'heatmap': heatmap_output
        },
        'update_visualizations': update_visualizations
    }
