"""
File: smartcash/ui/dataset/augmentation/components/input_options.py
Deskripsi: Compact normalization options dengan orange colors
"""

def _create_normalization_options(preprocessing_config: Dict[str, Any]) -> Dict[str, Any]:
    """Create compact normalization options"""
    norm_config = preprocessing_config.get('normalization', {})
    
    norm_method = widgets.Dropdown(
        options=[
            ('MinMax (0-1): YOLOv5 compatible', 'minmax'),
            ('Standard (z-score): Statistical normalization', 'standard'), 
            ('ImageNet: Transfer learning preset', 'imagenet'),
            ('None: Raw values (0-255)', 'none')
        ],
        value=norm_config.get('method', 'minmax'),
        description='Norm Method:',
        disabled=False,
        layout=widgets.Layout(width='95%'),
        style={'description_width': '100px'}
    )
    
    denormalize = widgets.Checkbox(
        value=norm_config.get('denormalize', False),
        description='Denormalize setelah preprocessing (save as uint8)',
        indent=False,
        layout=widgets.Layout(width='auto', margin='6px 0')
    )
    
    # FIXED: Compact info dengan orange colors
    norm_info = widgets.HTML(
        f"""
        <div style="padding: 6px 8px; background-color: #ff980315; 
                    border-radius: 4px; margin: 6px 0; font-size: 10px;
                    border: 1px solid #ff980340; line-height: 1.3;">
            <strong style="color: #f57c00;">🔧 Normalization Backend:</strong><br>
            • <strong style="color: #f57c00;">MinMax:</strong> OpenCV + NumPy [0.0, 1.0]<br>
            • <strong style="color: #f57c00;">Standard:</strong> Scikit-learn compatible<br>
            • <strong style="color: #f57c00;">ImageNet:</strong> Torchvision preset<br>
            • <strong style="color: #f57c00;">Target:</strong> 640x640 fixed untuk YOLO
        </div>
        """,
        layout=widgets.Layout(width='100%', margin='3px 0')
    )
    
    # Container dengan flexbox
    container = widgets.VBox([
        widgets.HTML("<h6 style='color: #f57c00; margin: 6px 0;'>📊 Normalisasi Augmentasi</h6>"),
        norm_method,
        denormalize,
        norm_info
    ], layout=widgets.Layout(
        padding='10px', 
        width='100%',
        display='flex',
        flex_flow='column',
        align_items='stretch',
        gap='4px'
    ))
    
    return {
        'container': container,
        'widgets': {
            'norm_method': norm_method,
            'denormalize': denormalize
        },
        'validation': {
            'ranges': {},
            'required': ['norm_method'],
            'defaults': {
                'norm_method': 'minmax',
                'denormalize': False
            }
        },
        'backend_mapping': {
            'preprocessing': {
                'normalization_method': 'norm_method',
                'denormalize_output': 'denormalize',
                'target_size': [640, 640]
            }
        }
    }