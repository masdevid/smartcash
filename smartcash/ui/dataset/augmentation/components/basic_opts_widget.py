"""
File: smartcash/ui/dataset/augmentation/components/basic_opts_widget.py
Deskripsi: Basic options widget dengan cleanup options integration dan responsive styling
"""

import ipywidgets as widgets
from typing import Dict, Any

def create_basic_options_widget() -> Dict[str, Any]:
    """Create basic options dengan cleanup integration dan responsive styling"""
    
    # Create widgets dengan overflow-safe styling
    widgets_dict = {
        'num_variations': widgets.IntSlider(
            value=2, min=1, max=10, step=1, description='Jumlah Variasi (1-10):',
            continuous_update=False, readout=True, readout_format='d',
            style={'description_width': '110px'}, layout=widgets.Layout(width='auto', max_width='100%')
        ),
        'target_count': widgets.IntSlider(
            value=500, min=100, max=2000, step=50, description='Jumlah Target:',
            continuous_update=False, readout=True, readout_format='d',
            style={'description_width': '110px'}, layout=widgets.Layout(width='auto', max_width='100%')
        ),
        'intensity': widgets.FloatSlider(
            value=0.7, min=0.1, max=1.0, step=0.1, description='Tingkat Intensitas (0.1-1.0):',
            continuous_update=False, readout=True, readout_format='.1f',
            style={'description_width': '110px'}, layout=widgets.Layout(width='auto', max_width='100%')
        ),
        'target_split': widgets.Dropdown(
            options=[
                ('🎯 Latih - Dataset pelatihan (Direkomendasikan)', 'train'),
                ('📊 Validasi - Dataset validasi', 'valid'),
                ('🧪 Uji - Dataset pengujian (Tidak Direkomendasikan)', 'test')
            ],
            value='train', description='Pembagian Dataset:', disabled=False,
            style={'description_width': '110px'}, layout=widgets.Layout(width='auto', max_width='100%')
        ),
        # CHANGED: cleanup_target menggantikan output_prefix
        'cleanup_target': widgets.Dropdown(
            options=[
                ('🧹 Augmented - Hapus file hasil augmentasi', 'augmented'),
                ('🖼️ Sampel - Hapus pratinjau sampel', 'samples'),
                ('🗑️ Keduanya - Hapus hasil dan pratinjau', 'both')
            ],
            value='both', description='Target Pembersihan:', disabled=False,
            style={'description_width': '110px'}, layout=widgets.Layout(width='auto', max_width='100%')
        ),
        'balance_classes': widgets.Checkbox(
            value=True, description='Seimbangkan Kelas (Optimal untuk Layer 1 & 2)',
            indent=False, layout=widgets.Layout(width='auto', margin='6px 0')
        )
    }
    
    # Create info content with simple HTML
    info_content = widgets.HTML("""
    <div style='background: #f8f9fa; padding: 10px; border-radius: 5px; margin: 10px 0;'>
        <strong>Panduan Parameter:</strong><br>
        • Variasi: 2-5 optimal untuk penelitian<br>
        • Jumlah Target: 500-1000 efektif<br>
        • Intensitas: 0.7 optimal, 0.3-0.5 konservatif<br>
        • Pembersihan: Keduanya = pembersihan menyeluruh
    </div>
    """)
    
    # Create container with simple layout
    container = widgets.VBox([
        widgets.HTML("<h6 style='color: #4caf50; margin: 6px 0;'>⚙️ Opsi Dasar Augmentasi</h6>"),
        *widgets_dict.values(),
        info_content
    ], layout=widgets.Layout(width='100%'))
    
    return {
        'container': container,
        'widgets': widgets_dict,
        'validation': {
            'ranges': {
                'num_variations': (1, 10),
                'target_count': (100, 2000),
                'intensity': (0.1, 1.0)
            },
            'required': ['num_variations', 'target_count', 'intensity', 'target_split', 'cleanup_target'],
            'backend_compatible': True
        },
        'backend_mapping': {
            'num_variations': 'augmentation.num_variations',
            'target_count': 'augmentation.target_count',
            'intensity': 'augmentation.intensity',
            'target_split': 'augmentation.target_split',
            'cleanup_target': 'cleanup.default_target',
            'balance_classes': 'augmentation.balance_classes'
        }
    }