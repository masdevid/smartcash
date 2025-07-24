"""
File: smartcash/ui/dataset/augmentation/components/augtypes_opts_widget.py
Deskripsi: Augmentation types widget yang dioptimasi dengan styling terkonsolidasi
"""

import ipywidgets as widgets
from typing import Dict, Any

def create_augmentation_types_widget() -> Dict[str, Any]:
    """Create compact augmentation types widget dengan styling terkonsolidasi"""
    
    # Augmentation types widget
    augmentation_types = widgets.SelectMultiple(
        options=[
            ('🎯 Gabungan: Posisi + Pencahayaan (Pipeline Riset)', 'combined'),
            ('📍 Posisi: Transformasi geometri (rotasi, balik, skala)', 'position'),
            ('💡 Pencahayaan: Variasi pencahayaan (kecerahan, kontras, HSV)', 'lighting'),
            ('🔄 Geometri: Transformasi lanjutan (perspektif, geser)', 'geometric'),
            ('🎨 Warna: Variasi warna dan saturasi', 'color'),
            ('📡 Derau: Gaussian noise dan motion blur', 'noise')
        ],
        value=['combined'],
        disabled=False,
        layout=widgets.Layout(width='100%', height='120px'),
        style={'description_width': '0'}
    )
    
    # Create info content with simple HTML
    info_content = widgets.HTML("""
    <div style='background: #f0f8f0; padding: 8px; border-radius: 4px; margin: 8px 0; font-size: 12px;'>
        <strong>Jenis Augmentasi:</strong><br>
        • Gabungan: Pipeline riset optimal<br>
        • Posisi: Transformasi geometri + pelestarian bbox<br>
        • Pencahayaan: Transformasi fotometrik pencahayaan<br>
        • Lanjutan: Transformasi geometri, warna, dan derau
    </div>
    """)
    
    # Create container with simple layout
    container = widgets.VBox([
        widgets.HTML("<h6 style='color: #2196f3; margin: 6px 0;'>🔄 Jenis Augmentasi</h6>"),
        augmentation_types,
        info_content
    ], layout=widgets.Layout(width='100%'))
    
    return {
        'container': container,
        'widgets': {
            'augmentation_types': augmentation_types
        },
        'validation': {
            'required': ['augmentation_types'],
            'defaults': {
                'augmentation_types': ['combined']
            },
            'backend_compatible': True
        },
        'options': {
            'augmentation_types': ['combined', 'position', 'lighting', 'geometric', 'color', 'noise'],
            'backend_recommended': {
                'types': ['combined', 'position', 'lighting']
            }
        },
        'backend_mapping': {
            'augmentation_types': 'augmentation.types'
        }
    }