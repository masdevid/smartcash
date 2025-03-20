"""
File: smartcash/ui/info_boxes/environment.py
Deskripsi: Konten info box untuk konfigurasi environment
"""

import ipywidgets as widgets
from typing import Optional

from smartcash.ui.utils.info_utils import create_info_accordion

def get_environment_info(open_by_default: bool = False) -> widgets.Accordion:
    """
    Dapatkan info box untuk konfigurasi environment.
    
    Args:
        open_by_default: Buka accordion secara default
        
    Returns:
        Widget accordion berisi info konfigurasi environment
    """
    content = """
    <h3 style="margin-top:0">Konfigurasi Environment</h3>
    
    <p>Konfigurasi environment akan memastikan project SmartCash berjalan dengan baik di lingkungan saat ini.</p>
    
    <h4>Google Colab</h4>
    <ul>
        <li>Hubungkan ke Google Drive untuk menyimpan dataset, model, dan hasil</li>
        <li>Data akan disimpan di <code>/content/drive/MyDrive/SmartCash</code></li>
        <li>Symlink akan dibuat untuk <code>data</code>, <code>configs</code>, <code>runs</code>, dll</li>
    </ul>
    
    <h4>Lingkungan Lokal</h4>
    <ul>
        <li>Pastikan struktur direktori telah dibuat</li>
        <li>Direktori utama: <code>data</code>, <code>configs</code>, <code>runs</code>, <code>logs</code></li>
        <li>Subdirektori data: <code>train</code>, <code>valid</code>, <code>test</code></li>
    </ul>
    
    <h4>Tips Konfigurasi</h4>
    <ul>
        <li>Simpan konfigurasi dalam file YAML di direktori <code>configs</code></li>
        <li>Sesuaikan <code>configs/colab_config.yaml</code> untuk Google Colab</li>
        <li>Gunakan <code>configs/base_config.yaml</code> untuk pengaturan umum</li>
        <li>Cache preprocessing dapat mempercepat loading dataset</li>
    </ul>
    """
    
    return create_info_accordion(
        "Informasi Konfigurasi Environment",
        content,
        "info",
        "⚙️",
        open_by_default
    )

def get_drive_info(open_by_default: bool = False) -> widgets.Accordion:
    """
    Dapatkan info box untuk koneksi Google Drive.
    
    Args:
        open_by_default: Buka accordion secara default
        
    Returns:
        Widget accordion berisi info koneksi Drive
    """
    content = """
    <h3 style="margin-top:0">Koneksi Google Drive</h3>
    
    <p>Google Drive digunakan untuk menyimpan data dan model agar tidak hilang saat sesi Colab berakhir.</p>
    
    <h4>Struktur Drive</h4>
    <ul>
        <li><code>SmartCash/data</code> - Dataset (gambar dan label)</li>
        <li><code>SmartCash/configs</code> - File konfigurasi</li>
        <li><code>SmartCash/runs</code> - Hasil training dan checkpoints</li>
        <li><code>SmartCash/logs</code> - Log training dan operasi</li>
        <li><code>SmartCash/exports</code> - Model teroptimasi untuk deployment</li>
    </ul>
    
    <h4>Tips Sinkronisasi</h4>
    <ul>
        <li>Pembuatan symlink otomatis akan mengarahkan direktori lokal ke Drive</li>
        <li>Jika koneksi terputus, hubungkan kembali tanpa khawatir kehilangan data</li>
        <li>Hindari mengganti nama direktori di Drive agar symlinks tetap berfungsi</li>
        <li>Backup secara reguler ke lokasi terpisah untuk keamanan data</li>
    </ul>
    """
    
    return create_info_accordion(
        "Informasi Koneksi Google Drive",
        content,
        "info",
        "📂",
        open_by_default
    )

def get_directory_structure_info(open_by_default: bool = False) -> widgets.Accordion:
    """
    Dapatkan info box untuk struktur direktori.
    
    Args:
        open_by_default: Buka accordion secara default
        
    Returns:
        Widget accordion berisi info struktur direktori
    """
    content = """
    <h3 style="margin-top:0">Struktur Direktori</h3>
    
    <p>SmartCash menggunakan struktur direktori standar untuk organisasi data dan model.</p>
    
    <h4>Direktori Utama</h4>
    <pre>
smartcash/                   # Root project
├── data/                    # Dataset
│   ├── train/               # Training data
│   │   ├── images/          # Training images
│   │   └── labels/          # Training labels (format YOLO)
│   ├── valid/               # Validation data
│   │   ├── images/          # Validation images
│   │   └── labels/          # Validation labels
│   └── test/                # Test data
│       ├── images/          # Test images
│       └── labels/          # Test labels
├── configs/                 # Konfigurasi
│   ├── base_config.yaml     # Konfigurasi dasar
│   └── colab_config.yaml    # Konfigurasi untuk Colab
├── runs/                    # Hasil training
│   └── train/               # Training runs
│       └── weights/         # Model weights
├── logs/                    # Log aplikasi
├── exports/                 # Model teroptimasi
└── checkpoints/             # Checkpoint training
    </pre>
    
    <h4>Tips Pengorganisasian</h4>
    <ul>
        <li>Gunakan <code>data</code> untuk dataset mentah dan terproses</li>
        <li>Simpan konfigurasi di <code>configs</code> dengan format YAML</li>
        <li>Hasil training otomatis akan disimpan di <code>runs/train</code></li>
        <li>Simpan model untuk deployment di <code>exports</code></li>
    </ul>
    """
    
    return create_info_accordion(
        "Informasi Struktur Direktori",
        content,
        "info",
        "🗂️",
        open_by_default
    )