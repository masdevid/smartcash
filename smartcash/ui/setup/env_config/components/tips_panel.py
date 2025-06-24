"""
File: smartcash/ui/setup/env_config/components/tips_panel.py
Deskripsi: Component untuk menampilkan tips dan requirements menggunakan shared tips_panel
"""

import ipywidgets as widgets
from smartcash.ui.components.tips_panel import create_tips_panel

def create_tips_requirements() -> widgets.HTML:
    """
    💡 Buat panel tips dan requirements menggunakan shared component
    
    Returns:
        Tips panel dengan 2 kolom
    """
    # Define tips content dalam format yang sesuai untuk tips_panel
    tips_content = [
        [
            "🚀 GPU Runtime: Aktifkan GPU di Runtime > Change runtime type untuk training yang lebih cepat",
            "💾 Google Drive: Mount Drive untuk penyimpanan persistent dan akses dataset",
            "🔄 Auto-sync: Config akan otomatis tersinkron dari Drive jika tersedia"
        ],
        [
            "📁 Symlinks: Folder akan otomatis ter-link untuk akses data yang efisien",
            "⚡ Restart Runtime: Jika ada error, coba restart runtime dan jalankan ulang",
            "📦 Libraries: PyTorch, OpenCV, Pillow akan diinstall otomatis"
        ]
    ]
    
    return create_tips_panel(
        title="💡 Tips & Requirements",
        tips=tips_content,
        gradient_start="#e3f2fd",  # Light blue
        gradient_end="#fce4ec",    # Light pink
        border_color="#2196f3",
        title_color="#1976d2",
        text_color="#424242",
        columns=2,
        margin="10px 0"
    )