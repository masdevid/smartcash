"""
File: smartcash/ui/dataset/augmentation/components/augtypes_opts_widget.py
Deskripsi: Fixed augmentation types widget dengan full width layout dan informasi yang jelas
"""

import ipywidgets as widgets
from typing import Dict, Any

def create_augmentation_types_widget() -> Dict[str, Any]:
    """
    Create augmentation types dan target split selection widget dengan full width layout
    
    Returns:
        Dictionary berisi container dan widget mapping
    """
    from smartcash.ui.utils.constants import COLORS, ICONS
    
    # Jenis augmentasi dengan deskripsi yang jelas - full width
    augmentation_types = widgets.SelectMultiple(
        options=[
            ('🎯 Combined: Kombinasi posisi dan pencahayaan (direkomendasikan)', 'combined'),
            ('📍 Position: Variasi posisi seperti rotasi, flipping, dan scaling', 'position'),
            ('💡 Lighting: Variasi pencahayaan seperti brightness, contrast dan HSV', 'lighting')
        ],
        value=['combined'],
        description='Jenis Augmentasi:',
        disabled=False,
        layout=widgets.Layout(width='auto', height='120px'),
        style={'description_width': '130px'}
    )
    
    # Target split dengan guidance
    target_split = widgets.Dropdown(
        options=[
            ('🎯 Train - Dataset training (direkomendasikan)', 'train'),
            ('📊 Valid - Dataset validasi (jarang diperlukan)', 'valid'),
            ('🧪 Test - Dataset testing (tidak direkomendasikan)', 'test')
        ],
        value='train',
        description='Target Split:',
        disabled=False,
        layout=widgets.Layout(width='auto'),
        style={'description_width': '130px'}
    )
    
    # Types information panel dengan detail
    types_info = widgets.HTML(
        f"""
        <div style="padding: 10px; background-color:#2196f315; 
                    border-radius: 6px; margin: 8px 0; font-size: 12px;
                    border: 1px solid #2196f3;">
            <strong style="color:#2196f3">{ICONS.get('augmentation', '🔄')} Jenis Augmentasi:</strong><br>
            • <strong style="color:#2196f3">Combined</strong>: Gabungan transformasi posisi dan pencahayaan (pipeline penelitian)<br>
            • <strong style="color:#2196f3">Position</strong>: Hanya transformasi geometri (rotasi, flip, scale, translate)<br>
            • <strong style="color:#2196f3">Lighting</strong>: Hanya transformasi pencahayaan (HSV, brightness, contrast)
        </div>
        """,
        layout=widgets.Layout(width='100%', margin='5px 0')
    )
    
    # Split information panel
    split_info = widgets.HTML(
        f"""
        <div style="padding: 10px; background-color:#4caf5015; 
                    border-radius: 6px; margin: 8px 0; font-size: 12px;
                    border: 1px solid #4caf50;">
            <strong style="color:#2e7d32">{ICONS.get('info', 'ℹ️')} Informasi Split:</strong><br>
            • <strong style="color:#2e7d32">train</strong>: Augmentasi pada data training untuk meningkatkan variasi data<br>
            • <strong style="color:#2e7d32">valid</strong>: Augmentasi pada data validasi (hanya jika dataset sangat kecil)<br>
            • <strong style="color:#2e7d32">test</strong>: Augmentasi pada data testing (tidak direkomendasikan untuk evaluasi)
        </div>
        """,
        layout=widgets.Layout(width='100%', margin='5px 0')
    )
    aug_types = widgets.VBox([
        widgets.HTML(f"<h6 style='color: {COLORS.get('dark', '#333')}; margin: 8px 0;'>{ICONS.get('augmentation', '🔄')} Pilih Jenis Augmentasi:</h6>"),
        augmentation_types,
        types_info], layout=widgets.Layout(width='46%'))
    split_selcetion  = widgets.VBox([
        widgets.HTML(f"<h6 style='color: {COLORS.get('dark', '#333')}; margin: 15px 0 8px 0;'>{ICONS.get('split', '📂')} Target Split Dataset:</h6>"),
        target_split,
        split_info
    ], layout=widgets.Layout(width='46%', overflow='hidden', display='flex', flex_flow='column'))
    # FIXED: Layout full width dengan proper spacing
    container = widgets.HBox([
        aug_types,
        split_selcetion
    ], layout=widgets.Layout(
        width='100%',           # FIXED: Full width
        padding='10px',
        margin='5px 0',
        display='flex',
        justify_content="space-between"
    ))
    
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
        # Extended options mapping
        'options': {
            'augmentation_types': ['combined', 'position', 'lighting'],
            'target_split': ['train', 'valid', 'test']
        }
    }