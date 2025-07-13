"""
File: smartcash/ui/dataset/augment/components/advanced_options.py
Description: Advanced options widget with preserved styling and parameter logic

This component creates advanced augmentation parameters (position and lighting)
with all original business logic preserved while following the new UI structure.
"""

import ipywidgets as widgets
from typing import Dict, Any
from smartcash.ui.core.decorators import handle_ui_errors
from ..constants import (
    DEFAULT_POSITION_PARAMS, DEFAULT_LIGHTING_PARAMS,
    HELP_TEXT, AUGMENT_COLORS, SECTION_STYLES
)


def _create_info_panel(content: str, theme: str = 'advanced_options') -> widgets.HTML:
    """Create info panel with preserved styling."""
    style_config = SECTION_STYLES.get(theme, SECTION_STYLES['advanced_options'])
    bg_color = style_config['background']
    border_color = style_config['border_color']
    
    return widgets.HTML(
        f"""
        <div style="padding: 6px 8px; background-color: {bg_color}; 
                    border-radius: 4px; margin: 6px 0; font-size: 10px;
                    border: 1px solid {border_color}; line-height: 1.3;">
            {content}
        </div>
        """,
        layout=widgets.Layout(width='100%', margin='3px 0')
    )


def _create_parameter_info() -> str:
    """Create parameter guidance content."""
    return """
    <strong style="color: #9c27b0; margin-bottom:4px">⚙️ Parameter Guidelines:</strong><br>
    • <strong style='color: #9c27b0;'>Position:</strong> Flip (0.5), Rotation (≤15°), Scale (≤0.1)<br>
    • <strong style='color: #9c27b0;'>Lighting:</strong> Brightness/Contrast (≤0.3), HSV (≤20)<br>
    • <strong style='color: #9c27b0;'>HSV:</strong> Hue affects color tone, Saturation affects intensity<br>
    • <strong style='color: #9c27b0;'>Balance:</strong> Conservative values for currency detection
    """


@handle_ui_errors(error_component_title="Advanced Options Creation Error") 
def create_advanced_options_widget() -> Dict[str, Any]:
    """
    Create advanced options widget with preserved parameter logic.
    
    Features:
    - 🔄 Position parameters (flip, rotation, translate, scale)
    - 💡 Lighting parameters (brightness, contrast, HSV)
    - 🎨 Preserved styling and responsive layout
    - ✅ Business logic validation ranges
    - 📊 Parameter guidance and tooltips
    
    Returns:
        Dictionary containing container, widgets, and metadata
    """
    
    # Position parameter widgets
    position_widgets = {
        'horizontal_flip_slider': widgets.FloatSlider(
            value=DEFAULT_POSITION_PARAMS['horizontal_flip'],
            min=0.0, max=1.0, step=0.1,
            description='H-Flip Prob:',
            continuous_update=False,
            readout=True, readout_format='.1f',
            tooltip=HELP_TEXT['horizontal_flip'],
            style={'description_width': '100px'},
            layout=widgets.Layout(width='100%')
        ),
        
        'rotation_limit_slider': widgets.IntSlider(
            value=DEFAULT_POSITION_PARAMS['rotation_limit'],
            min=0, max=45, step=1,
            description='Rotation (°):',
            continuous_update=False,
            readout=True, readout_format='d',
            tooltip=HELP_TEXT['rotation_limit'],
            style={'description_width': '100px'},
            layout=widgets.Layout(width='100%')
        ),
        
        'translate_limit_slider': widgets.FloatSlider(
            value=DEFAULT_POSITION_PARAMS['translate_limit'],
            min=0.0, max=0.3, step=0.01,
            description='Translate:',
            continuous_update=False,
            readout=True, readout_format='.2f',
            tooltip=HELP_TEXT['translate_limit'],
            style={'description_width': '100px'},
            layout=widgets.Layout(width='100%')
        ),
        
        'scale_limit_slider': widgets.FloatSlider(
            value=DEFAULT_POSITION_PARAMS['scale_limit'],
            min=0.0, max=0.2, step=0.01,
            description='Scale:',
            continuous_update=False,
            readout=True, readout_format='.2f',
            tooltip=HELP_TEXT['scale_limit'],
            style={'description_width': '100px'},
            layout=widgets.Layout(width='100%')
        )
    }
    
    # Lighting parameter widgets
    lighting_widgets = {
        'brightness_limit_slider': widgets.FloatSlider(
            value=DEFAULT_LIGHTING_PARAMS['brightness_limit'],
            min=0.0, max=0.5, step=0.05,
            description='Brightness:',
            continuous_update=False,
            readout=True, readout_format='.2f',
            tooltip=HELP_TEXT['brightness_limit'],
            style={'description_width': '100px'},
            layout=widgets.Layout(width='100%')
        ),
        
        'contrast_limit_slider': widgets.FloatSlider(
            value=DEFAULT_LIGHTING_PARAMS['contrast_limit'],
            min=0.0, max=0.5, step=0.05,
            description='Contrast:',
            continuous_update=False,
            readout=True, readout_format='.2f',
            tooltip=HELP_TEXT['contrast_limit'],
            style={'description_width': '100px'},
            layout=widgets.Layout(width='100%')
        ),
        
        'hsv_hue_slider': widgets.IntSlider(
            value=DEFAULT_LIGHTING_PARAMS['hsv_hue'],
            min=0, max=30, step=1,
            description='HSV Hue:',
            continuous_update=False,
            readout=True, readout_format='d',
            tooltip=HELP_TEXT['hsv_hue'],
            style={'description_width': '100px'},
            layout=widgets.Layout(width='100%')
        ),
        
        'hsv_saturation_slider': widgets.IntSlider(
            value=DEFAULT_LIGHTING_PARAMS['hsv_saturation'],
            min=0, max=50, step=1,
            description='HSV Sat:',
            continuous_update=False,
            readout=True, readout_format='d',
            tooltip=HELP_TEXT['hsv_saturation'],
            style={'description_width': '100px'},
            layout=widgets.Layout(width='100%')
        )
    }
    
    # Combine all widgets
    all_widgets = {**position_widgets, **lighting_widgets}
    
    # Create section headers
    position_header = widgets.HTML(f"""
    <h6 style='color: {AUGMENT_COLORS["primary"]}; margin: 8px 0 4px 0; font-size: 11px; font-weight: 600;'>
        🔄 Position Parameters
    </h6>
    """)
    
    lighting_header = widgets.HTML(f"""
    <h6 style='color: {AUGMENT_COLORS["warning"]}; margin: 12px 0 4px 0; font-size: 11px; font-weight: 600;'>
        💡 Lighting Parameters
    </h6>
    """)
    
    main_header = widgets.HTML(f"""
    <h6 style='color: {AUGMENT_COLORS["primary"]}; margin: 6px 0; font-size: 12px; font-weight: 600;'>
        ⚙️ Advanced Parameters
    </h6>
    """)
    
    # Create info panel
    info_panel = _create_info_panel(_create_parameter_info(), 'advanced_options')
    
    # Create container with organized layout
    container = widgets.VBox([
        main_header,
        
        # Position parameters section
        position_header,
        position_widgets['horizontal_flip_slider'],
        position_widgets['rotation_limit_slider'],
        position_widgets['translate_limit_slider'],
        position_widgets['scale_limit_slider'],
        
        # Lighting parameters section
        lighting_header,
        lighting_widgets['brightness_limit_slider'],
        lighting_widgets['contrast_limit_slider'],
        lighting_widgets['hsv_hue_slider'],
        lighting_widgets['hsv_saturation_slider'],
        
        # Info panel
        info_panel
        
    ], layout=widgets.Layout(
        width='100%',
        padding='10px',
        display='flex',
        flex_flow='column',
        align_items='stretch',
        gap='4px'
    ))
    
    return {
        'container': container,
        'widgets': all_widgets,
        
        # Parameter categories
        'position_widgets': list(position_widgets.keys()),
        'lighting_widgets': list(lighting_widgets.keys()),
        
        # Validation configuration
        'validation': {
            'ranges': {
                'horizontal_flip_slider': (0.0, 1.0),
                'rotation_limit_slider': (0, 45),
                'translate_limit_slider': (0.0, 0.3),
                'scale_limit_slider': (0.0, 0.2),
                'brightness_limit_slider': (0.0, 0.5),
                'contrast_limit_slider': (0.0, 0.5),
                'hsv_hue_slider': (0, 30),
                'hsv_saturation_slider': (0, 50)
            },
            'required': list(all_widgets.keys()),
            'backend_compatible': True
        },
        
        # Backend configuration mapping
        'backend_mapping': {
            # Position parameters
            'horizontal_flip_slider': 'augmentation.position.horizontal_flip',
            'rotation_limit_slider': 'augmentation.position.rotation_limit',
            'translate_limit_slider': 'augmentation.position.translate_limit',
            'scale_limit_slider': 'augmentation.position.scale_limit',
            
            # Lighting parameters
            'brightness_limit_slider': 'augmentation.lighting.brightness_limit',
            'contrast_limit_slider': 'augmentation.lighting.contrast_limit',
            'hsv_hue_slider': 'augmentation.lighting.hsv_hue',
            'hsv_saturation_slider': 'augmentation.lighting.hsv_saturation'
        },
        
        # Form metadata
        'form_type': 'advanced_options',
        'component_version': '2.0.0',
        'preserved_business_logic': True,
        'parameter_categories': ['position', 'lighting']
    }