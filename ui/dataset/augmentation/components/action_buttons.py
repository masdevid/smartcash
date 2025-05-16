"""
File: smartcash/ui/dataset/augmentation/components/action_buttons.py
Deskripsi: Komponen tombol aksi untuk augmentasi dataset
"""

import ipywidgets as widgets
from typing import Dict, Any

def create_action_buttons() -> Dict[str, Any]:
    """
    Buat komponen tombol aksi khusus untuk augmentasi dataset.
    
    Returns:
        Dictionary berisi tombol dan containers
    """
    # Buat tombol augmentasi
    augment_button = widgets.Button(
        description="Run Augmentation",
        icon="random",
        button_style="primary",
        tooltip="Jalankan proses augmentasi dataset",
        layout=widgets.Layout(width="auto")
    )
    
    # Buat tombol stop
    stop_button = widgets.Button(
        description="Stop",
        icon="stop",
        button_style="danger",
        tooltip="Hentikan proses augmentasi yang sedang berjalan",
        layout=widgets.Layout(width="auto", display="none")
    )
    
    # Buat tombol reset
    reset_button = widgets.Button(
        description="Reset",
        icon="refresh",
        button_style="warning",
        tooltip="Reset konfigurasi augmentasi ke default",
        layout=widgets.Layout(width="auto")
    )
    
    # Buat tombol save
    save_button = widgets.Button(
        description="Save Config",
        icon="save",
        button_style="info",
        tooltip="Simpan konfigurasi augmentasi saat ini",
        layout=widgets.Layout(width="auto")
    )
    
    # Buat tombol cleanup
    cleanup_button = widgets.Button(
        description="Cleanup",
        icon="trash",
        button_style="warning",
        tooltip="Bersihkan hasil augmentasi",
        layout=widgets.Layout(width="auto", display="none")
    )
    
    # Buat container untuk tombol aksi
    action_container = widgets.HBox(
        [augment_button, stop_button, reset_button, save_button, cleanup_button],
        layout=widgets.Layout(justify_content="flex-start", margin="10px 0")
    )
    
    # Buat tombol visualisasi
    visualize_button = widgets.Button(
        description="Visualize",
        icon="eye",
        button_style="success",
        tooltip="Visualisasikan hasil augmentasi",
        layout=widgets.Layout(width="auto")
    )
    
    # Buat tombol compare
    compare_button = widgets.Button(
        description="Compare",
        icon="columns",
        button_style="info",
        tooltip="Bandingkan gambar asli dan hasil augmentasi",
        layout=widgets.Layout(width="auto")
    )
    
    # Buat tombol distribution
    distribution_button = widgets.Button(
        description="Distribution",
        icon="chart-bar",
        button_style="info",
        tooltip="Tampilkan distribusi kelas setelah augmentasi",
        layout=widgets.Layout(width="auto")
    )
    
    # Buat container untuk tombol visualisasi
    visualization_container = widgets.HBox(
        [visualize_button, compare_button, distribution_button],
        layout=widgets.Layout(justify_content="flex-start", margin="10px 0", display="none")
    )
    
    # Kembalikan dictionary gabungan
    result = {
        'augment_button': augment_button,
        'stop_button': stop_button,
        'reset_button': reset_button,
        'save_button': save_button,
        'cleanup_button': cleanup_button,
        'container': action_container,
        'visualization_buttons': visualization_container,
        'visualize_button': visualize_button,
        'compare_button': compare_button,
        'distribution_button': distribution_button
    }
    
    return result