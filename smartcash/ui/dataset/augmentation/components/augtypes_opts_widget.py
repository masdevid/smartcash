"""
File: smartcash/ui/dataset/augmentation/components/augtypes_opts_widget.py
Deskripsi: Augmentation types dan target split widget dengan informasi yang jelas
"""

import ipywidgets as widgets
from typing import Dict, Any

def create_augmentation_types_widget() -> Dict[str, Any]:
    """
    Create augmentation types dan target split selection widget
    
    Returns:
        Dictionary berisi container dan widget mapping
    """
    from smartcash.ui.utils.constants import COLORS, ICONS
    
    # Jenis augmentasi dengan deskripsi yang jelas
    augmentation_types = widgets.SelectMultiple(
        options=[
            ('Combined: Kombinasi posisi dan pencahayaan (direkomendasikan)', 'combined'),
            ('Position: Variasi posisi seperti rotasi, flipping, dan scaling', 'position'),
            ('Lighting: Variasi pencahayaan seperti brightness, contrast dan HSV', 'lighting')
        ],
        value=['combined'],
        description='Jenis Augmentasi:',
        disabled=False,
        layout=widgets.Layout(width='95%', height='100px'),
        style={'description_width': '130px'}
    )
    
    # Target split dengan guidance
    target_split = widgets.Dropdown(
        options=[
            ('Train - Dataset training (direkomendasikan)', 'train'),
            ('Valid - Dataset validasi (jarang diperlukan)', 'valid'),
            ('Test - Dataset testing (tidak direkomendasikan)', 'test')
        ],
        value='train',
        description='Target Split:',
        disabled=False,
        layout=widgets.Layout(width='95%'),
        style={'description_width': '130px'}
    )
    
    # Split information panel
    split_info = widgets.HTML(
        f"""
        <div style="padding: 8px; background-color:#e3f2fd; 
                    border-radius: 4px; margin: 5px 0; font-size: 11px;
                    border: 1px solid #2196f3;">
            <strong style="color:#2196f3">{ICONS.get('info', 'ℹ️')} Informasi Split:</strong><br>
            • <strong style="color:#2196f3">train</strong>: Augmentasi pada data training (rekomendasi)<br>
            • <strong style="color:#2196f3">valid</strong>: Augmentasi pada data validasi (jarang diperlukan)<br>
            • <strong style="color:#2196f3">test</strong>: Augmentasi pada data testing (tidak direkomendasikan)
        </div>
        """,
        layout=widgets.Layout(width='100%', margin='5px 0')
    )
    
    # Types information panel
    types_info = widgets.HTML(
        f"""
        <div style="padding: 8px; background-color:#e3f2fd; 
                    border-radius: 4px; margin: 5px 0; font-size: 11px;
                    border: 1px solid #2196f3;">
            <strong style="color:#2196f3">{ICONS.get('augmentation', '🔄')} Jenis Augmentasi:</strong><br>
            • <strong style="color:#2196f3">Combined</strong>: Gabungan transformasi posisi dan pencahayaan<br>
            • <strong style="color:#2196f3">Position</strong>: Hanya transformasi geometri (rotasi, flip, scale)<br>
            • <strong style="color:#2196f3">Lighting</strong>: Hanya transformasi pencahayaan (HSV, brightness)
        </div>
        """,
        layout=widgets.Layout(width='100%', margin='5px 0')
    )
    
    # Layout dengan 2 kolom
    left_column = widgets.VBox([
        widgets.HTML(f"<h6 style='color: {COLORS.get('dark', '#333')}; margin: 5px 0;'>{ICONS.get('augmentation', '🔄')} Pilih Jenis Augmentasi:</h6>"),
        augmentation_types,
        types_info
    ], layout=widgets.Layout(width='58%', padding='5px'))
    
    right_column = widgets.VBox([
        widgets.HTML(f"<h6 style='color: {COLORS.get('dark', '#333')}; margin: 5px 0;'>{ICONS.get('split', '📂')} Target Split:</h6>"),
        target_split,
        split_info
    ], layout=widgets.Layout(width='40%', padding='5px'))
    
    # Main container dengan responsive layout
    container = widgets.VBox([
        widgets.HTML(f"<h6 style='color: {COLORS.get('dark', '#333')}; margin: 5px 0; font-size: 14px;'>{ICONS.get('settings', '⚙️')} Jenis Augmentasi & Target Split</h6>"),
        widgets.HBox([left_column, right_column], 
                    layout=widgets.Layout(width='100%', justify_content='flex-start'))
    ], layout=widgets.Layout(padding='10px', width='100%'))
    
    return {
        'container': container,
        'widgets': {
            'augmentation_types': augmentation_types,
            'target_split': target_split
        },
        # Validation info
        'validation': {
            'required': ['augmentation_types', 'target_split'],
            'defaults': {
                'augmentation_types': ['combined'],
                'target_split': 'train'
            }
        },
        # Options mapping untuk validation
        'options': {
            'augmentation_types': ['combined', 'position', 'lighting'],
            'target_split': ['train', 'valid', 'test']
        }
    }