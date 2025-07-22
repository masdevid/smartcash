"""
File: smartcash/ui/dataset/augmentation/components/live_preview_widget.py
Deskripsi: Live preview widget dengan responsive image container dan generate button
"""

import ipywidgets as widgets
from typing import Dict, Any

def create_live_preview_widget() -> Dict[str, Any]:
    """Create simplified live preview dengan single image dan generate button"""
    
    # Preview image container - responsive 200x200px
    preview_image = widgets.Image(
        value=b'',  # Empty initially - loads from /data/aug_preview.jpg
        format='jpg',
        layout=widgets.Layout(
            width='200px', height='200px',
            border='2px solid #e0e0e0',
            border_radius='8px',
            object_fit='contain',
            margin='auto'
        )
    )
    
    # Generate preview button
    generate_button = widgets.Button(
        description='🎯 Generate Preview',
        button_style='info',
        tooltip='Generate preview ke /data/aug_preview.jpg',
        icon='image',
        layout=widgets.Layout(
            width='180px', height='32px', 
            margin='8px auto 0 auto'
        )
    )
    
    # Status text untuk preview
    preview_status = widgets.HTML(
        value="<div style='text-align: center; color: #666; font-size: 12px; margin: 4px 0;'>Preview: /data/aug_preview.jpg</div>",
        layout=widgets.Layout(width='100%', margin='4px auto')
    )
    
    # Info content with simple HTML
    info_content = widgets.HTML("""
    <div style='background: #f8f9fa; padding: 8px; border-radius: 4px; margin: 8px 0; font-size: 12px;'>
        <strong>Live Preview:</strong><br>
        • File: /data/aug_preview.jpg<br>
        • Generate: Buat preview dari parameter saat ini<br>
        • Format: 200x200px responsive container
    </div>
    """)
    
    # Main container dengan center alignment  
    container = widgets.VBox([
        widgets.HTML("<h6 style='color: #ff9800; margin: 6px 0;'>🎬 Live Preview Augmentasi</h6>"),
        widgets.VBox([
            preview_image,
            preview_status,
            generate_button
        ], layout=widgets.Layout(
            display='flex',
            flex_flow='column',
            align_items='center',
            margin='8px 0'
        )),
        info_content
    ], layout=widgets.Layout(width='100%'))
    
    # Widget mapping
    widgets_dict = {
        'preview_image': preview_image,
        'generate': generate_button,
        'preview_status': preview_status
    }
    
    return {
        'container': container,
        'widgets': widgets_dict,
        'validation': {
            'required': [],
            'backend_compatible': True
        },
        'backend_mapping': {
            'preview': {
                'output_file': '/data/aug_preview.jpg',
                'container_size': [200, 200]
            }
        },
        'preview_config': {
            'output_path': '/data/aug_preview.jpg',
            'image_size': (200, 200),
            'format': 'jpg',
            'quality': 85,
            'responsive': True
        }
    }