"""
File: smartcash/ui/dataset/augmentation/components/input_options.py
Deskripsi: Fixed normalization options dengan title yang benar
"""

def _create_normalization_options(preprocessing_config: Dict[str, Any]) -> Dict[str, Any]:
    """Create normalization options dengan title yang diperbaiki"""
    norm_config = preprocessing_config.get('normalization', {})
    
    # Normalization method
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
        layout=widgets.Layout(width='auto'),
        style={'description_width': '100px'}
    )
    
    # Denormalization option
    denormalize = widgets.Checkbox(
        value=norm_config.get('denormalize', False),
        description='Denormalize setelah preprocessing (save as uint8)',
        indent=False,
        layout=widgets.Layout(width='auto', margin='5px 0')
    )
    
    # Target size display
    target_size_display = widgets.HTML(
        f"""
        <div style="padding: 6px 8px; background-color: #f5f5f5; 
                    border-radius: 3px; margin: 5px 0; font-size: 11px;
                    border: 1px solid #ddd;">
            <strong>🎯 Target Size:</strong> 640x640 (Fixed untuk YOLOv5 compatibility)
        </div>
        """,
        layout=widgets.Layout(width='100%')
    )
    
    # Normalization info panel
    norm_info = widgets.HTML(
        f"""
        <div style="padding: 8px; background-color: #fff3e0; 
                    border-radius: 4px; margin: 5px 0; font-size: 11px;
                    border: 1px solid #ff980040;">
            <strong style="color: #f57c00;">🔧 Normalization Backend:</strong><br>
            • <strong style="color: #f57c00;">MinMax:</strong> OpenCV + NumPy array [0.0, 1.0]<br>
            • <strong style="color: #f57c00;">Standard:</strong> Scikit-learn StandardScaler compatible<br>
            • <strong style="color: #f57c00;">ImageNet:</strong> Torchvision transforms preset<br>
            • <strong style="color: #f57c00;">Denormalize:</strong> Convert float32 → uint8
        </div>
        """,
        layout=widgets.Layout(width='100%')
    )
    
    # FIXED: Changed title from "Preprocessing Options" to "Augmentation Normalization"
    container = widgets.VBox([
        widgets.HTML("<h6 style='color: #f57c00; margin: 8px 0;'>🔧 Augmentation Normalization</h6>"),
        norm_method,
        denormalize,
        target_size_display,
        norm_info
    ], layout=widgets.Layout(padding='8px', width='100%'))
    
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