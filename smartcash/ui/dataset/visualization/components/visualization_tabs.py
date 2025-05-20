"""
File: smartcash/ui/dataset/visualization/components/visualization_tabs.py
Deskripsi: Komponen tab untuk visualisasi dataset
"""

import ipywidgets as widgets
from typing import Dict, Any, List, Optional

from smartcash.ui.utils.constants import ICONS
from smartcash.ui.components.tab_factory import create_tab_widget


def create_distribution_tab() -> Dict[str, Any]:
    """
    Membuat tab untuk visualisasi distribusi kelas.
    
    Returns:
        Dictionary berisi komponen tab distribusi kelas
    """
    # Container untuk tab
    container = widgets.VBox([])
    
    # Output untuk visualisasi
    output = widgets.Output()
    
    # Tombol untuk menampilkan visualisasi
    button = widgets.Button(
        description=f"{ICONS.get('chart', '📊')} Tampilkan Distribusi Kelas",
        button_style='info',
        tooltip='Tampilkan distribusi kelas dataset'
    )
    
    # Tambahkan komponen ke container
    container.children = [button, output]
    
    # Kumpulkan komponen dalam dictionary
    components = {
        'container': container,
        'output': output,
        'button': button
    }
    
    return components


def create_split_distribution_tab() -> Dict[str, Any]:
    """
    Membuat tab untuk visualisasi distribusi split.
    
    Returns:
        Dictionary berisi komponen tab distribusi split
    """
    # Container untuk tab
    container = widgets.VBox([])
    
    # Output untuk visualisasi
    output = widgets.Output()
    
    # Tombol untuk menampilkan visualisasi
    button = widgets.Button(
        description=f"{ICONS.get('split', '📋')} Tampilkan Distribusi Split",
        button_style='info',
        tooltip='Tampilkan distribusi split dataset'
    )
    
    # Tambahkan komponen ke container
    container.children = [button, output]
    
    # Kumpulkan komponen dalam dictionary
    components = {
        'container': container,
        'output': output,
        'button': button
    }
    
    return components


def create_layer_distribution_tab() -> Dict[str, Any]:
    """
    Membuat tab untuk visualisasi distribusi layer deteksi.
    
    Returns:
        Dictionary berisi komponen tab distribusi layer
    """
    # Container untuk tab
    container = widgets.VBox([])
    
    # Output untuk visualisasi
    output = widgets.Output()
    
    # Tombol untuk menampilkan visualisasi
    button = widgets.Button(
        description=f"{ICONS.get('layer', '🔍')} Tampilkan Distribusi Layer",
        button_style='info',
        tooltip='Tampilkan distribusi layer deteksi'
    )
    
    # Tambahkan komponen ke container
    container.children = [button, output]
    
    # Kumpulkan komponen dalam dictionary
    components = {
        'container': container,
        'output': output,
        'button': button
    }
    
    return components


def create_preprocessing_samples_tab() -> Dict[str, Any]:
    """
    Membuat tab untuk menampilkan sampel preprocessing.
    
    Returns:
        Dictionary berisi komponen tab sampel preprocessing
    """
    # Container untuk tab
    container = widgets.VBox([])
    
    # Output untuk visualisasi
    output = widgets.Output()
    
    # Tombol untuk menampilkan visualisasi
    button = widgets.Button(
        description=f"{ICONS.get('image', '🖼️')} Tampilkan Sampel Preprocessing",
        button_style='info',
        tooltip='Tampilkan 3 sampel hasil preprocessing'
    )
    
    # Tambahkan komponen ke container
    container.children = [button, output]
    
    # Kumpulkan komponen dalam dictionary
    components = {
        'container': container,
        'output': output,
        'button': button
    }
    
    return components


def create_augmentation_comparison_tab() -> Dict[str, Any]:
    """
    Membuat tab untuk perbandingan gambar asli dengan hasil augmentasi.
    
    Returns:
        Dictionary berisi komponen tab perbandingan augmentasi
    """
    # Container untuk tab
    container = widgets.VBox([])
    
    # Output untuk visualisasi
    output = widgets.Output()
    
    # Tombol untuk menampilkan visualisasi
    button = widgets.Button(
        description=f"{ICONS.get('compare', '⚖️')} Tampilkan Perbandingan Augmentasi",
        button_style='info',
        tooltip='Tampilkan perbandingan gambar asli dengan hasil augmentasi'
    )
    
    # Tambahkan komponen ke container
    container.children = [button, output]
    
    # Kumpulkan komponen dalam dictionary
    components = {
        'container': container,
        'output': output,
        'button': button
    }
    
    return components


def create_visualization_tabs() -> Dict[str, Any]:
    """
    Membuat tab untuk berbagai visualisasi dataset.
    
    Returns:
        Dictionary berisi komponen tab visualisasi
    """
    # Buat komponen untuk setiap tab
    distribution_tab = create_distribution_tab()
    split_tab = create_split_distribution_tab()
    layer_tab = create_layer_distribution_tab()
    preprocessing_samples_tab = create_preprocessing_samples_tab()
    augmentation_comparison_tab = create_augmentation_comparison_tab()
    
    # Buat tab items
    tab_items = [
        ('Distribusi Kelas', distribution_tab['container']),
        ('Distribusi Split', split_tab['container']),
        ('Distribusi Layer', layer_tab['container']),
        ('Sampel Preprocessing', preprocessing_samples_tab['container']),
        ('Perbandingan Augmentasi', augmentation_comparison_tab['container'])
    ]
    
    # Buat tab widget
    tab = create_tab_widget(tab_items)
    
    # Kumpulkan semua komponen dalam dictionary
    components = {
        'tab': tab,
        'distribution_tab': distribution_tab,
        'split_tab': split_tab,
        'layer_tab': layer_tab,
        'preprocessing_samples_tab': preprocessing_samples_tab,
        'augmentation_comparison_tab': augmentation_comparison_tab
    }
    
    return components
