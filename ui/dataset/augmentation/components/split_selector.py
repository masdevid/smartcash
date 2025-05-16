"""
File: smartcash/ui/dataset/augmentation/components/split_selector.py
Deskripsi: Komponen UI untuk pemilihan split dataset yang akan diaugmentasi
"""

import ipywidgets as widgets
from typing import Dict, Any, List, Optional

def create_split_selector() -> widgets.VBox:
    """
    Buat komponen UI untuk pemilihan split dataset.
    
    Returns:
        Widget VBox berisi pemilihan split
    """
    from smartcash.ui.utils.constants import COLORS, ICONS
    
    # Daftar split yang tersedia
    available_splits = ['train', 'valid', 'test']
    
    # Selector untuk split
    split_selector = widgets.RadioButtons(
        options=available_splits,  # Pastikan ini adalah list, bukan tuple
        value='train',
        description='Split:',
        disabled=False,
        layout=widgets.Layout(width='auto')
    )
    
    # Pastikan options tetap berupa list untuk kompatibilitas dengan pengujian
    split_selector.options = available_splits  # Memastikan options tetap list
    
    # Informasi tentang split
    split_info = widgets.HTML(
        f"""
        <div style="padding: 5px; color: {COLORS['dark']};">
            <p><b>{ICONS['info']} Informasi Split:</b></p>
            <ul>
                <li><b>train</b>: Augmentasi pada data training (rekomendasi)</li>
                <li><b>valid</b>: Augmentasi pada data validasi (jarang diperlukan)</li>
                <li><b>test</b>: Augmentasi pada data testing (tidak direkomendasikan)</li>
            </ul>
        </div>
        """
    )
    
    # Container utama
    container = widgets.VBox([
        widgets.HTML(f"<h4 style='color: {COLORS['dark']}; margin-top: 10px; margin-bottom: 5px;'>{ICONS['folder']} Pilih Split Dataset</h4>"),
        widgets.HBox([
            split_selector,
            split_info
        ], layout=widgets.Layout(
            justify_content='space-between',
            align_items='center',
            border='1px solid #ddd',
            padding='10px',
            margin='5px 0 10px 0'
        ))
    ])
    
    return container
