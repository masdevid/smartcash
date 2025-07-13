"""
File: smartcash/ui/dataset/augment/components/basic_options.py
Description: Basic options widget with preserved styling and business logic

This component creates the basic augmentation options form with all original
business logic and styling preserved while following the new UI structure.
"""

import ipywidgets as widgets
from typing import Dict, Any
from smartcash.ui.core.decorators import handle_ui_errors
from ..constants import (
    DEFAULT_AUGMENTATION_PARAMS, TARGET_SPLIT_OPTIONS, CLEANUP_TARGET_OPTIONS,
    HELP_TEXT, AUGMENT_COLORS, SECTION_STYLES
)


def _create_info_panel(content: str, theme: str = 'basic_options') -> widgets.HTML:
    """Create info panel with preserved styling."""
    style_config = SECTION_STYLES.get(theme, SECTION_STYLES['basic_options'])
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


def _create_info_content() -> str:
    """Create info content with guidance."""
    return """
    <strong style="color: #4caf50; margin-bottom:4px">💡 Parameter Guidance:</strong><br>
    • <strong style='color: #4caf50;'>Variations:</strong> 2-5 optimal for research<br>
    • <strong style='color: #4caf50;'>Target Count:</strong> 500-1000 effective<br>
    • <strong style='color: #4caf50;'>Intensity:</strong> 0.7 optimal, 0.3-0.5 conservative<br>
    • <strong style='color: #4caf50;'>Cleanup:</strong> Both = comprehensive cleanup
    """


@handle_ui_errors(error_component_title="Basic Options Creation Error")
def create_basic_options_widget() -> Dict[str, Any]:
    """
    Create basic options widget with preserved business logic.
    
    Features:
    - 📊 All original form fields with business logic
    - 🎨 Preserved styling and responsive layout
    - ✅ Comprehensive validation rules
    - 🗑️ Cleanup target integration
    - 💡 User guidance and help text
    
    Returns:
        Dictionary containing container, widgets, and metadata
    """
    
    # Create form widgets with preserved business logic
    widgets_dict = {
        'num_variations_slider': widgets.IntSlider(
            value=DEFAULT_AUGMENTATION_PARAMS['num_variations'],
            min=1, max=10, step=1,
            description='Variations:',
            continuous_update=False,
            readout=True, readout_format='d',
            tooltip=HELP_TEXT['num_variations'],
            style={'description_width': '110px'},
            layout=widgets.Layout(width='100%', max_width='100%')
        ),
        
        'target_count_slider': widgets.IntSlider(
            value=DEFAULT_AUGMENTATION_PARAMS['target_count'],
            min=10, max=10000, step=50,
            description='Target Count:',
            continuous_update=False,
            readout=True, readout_format='d',
            tooltip=HELP_TEXT['target_count'],
            style={'description_width': '110px'},
            layout=widgets.Layout(width='100%', max_width='100%')
        ),
        
        'intensity_slider': widgets.FloatSlider(
            value=DEFAULT_AUGMENTATION_PARAMS['intensity'],
            min=0.0, max=1.0, step=0.1,
            description='Intensity:',
            continuous_update=False,
            readout=True, readout_format='.1f',
            tooltip=HELP_TEXT['intensity'],
            style={'description_width': '110px'},
            layout=widgets.Layout(width='100%', max_width='100%')
        ),
        
        'target_split_dropdown': widgets.Dropdown(
            options=TARGET_SPLIT_OPTIONS,
            value=DEFAULT_AUGMENTATION_PARAMS['target_split'],
            description='Target Split:',
            tooltip=HELP_TEXT['target_split'],
            style={'description_width': '110px'},
            layout=widgets.Layout(width='100%', max_width='100%')
        ),
        
        'cleanup_target_dropdown': widgets.Dropdown(
            options=CLEANUP_TARGET_OPTIONS,
            value='both',
            description='Cleanup Target:',
            tooltip=HELP_TEXT['cleanup_target'],
            style={'description_width': '110px'},
            layout=widgets.Layout(width='100%', max_width='100%')
        ),
        
        'balance_classes_checkbox': widgets.Checkbox(
            value=DEFAULT_AUGMENTATION_PARAMS['balance_classes'],
            description='Balance Classes (Layer-based weighting)',
            tooltip=HELP_TEXT['balance_classes'],
            indent=False,
            layout=widgets.Layout(width='auto', margin='6px 0')
        ),
        
        'data_dir_input': widgets.Text(
            value='data',
            description='Data Directory:',
            placeholder='Path to dataset directory',
            style={'description_width': '110px'},
            layout=widgets.Layout(width='100%', max_width='100%')
        )
    }
    
    # Create section header
    header_html = f"""
    <h6 style='color: {AUGMENT_COLORS["success"]}; margin: 6px 0; font-size: 12px; font-weight: 600;'>
        ⚙️ Basic Configuration
    </h6>
    """
    
    # Create info panel with guidance
    info_panel = _create_info_panel(_create_info_content(), 'basic_options')
    
    # Create container with responsive layout
    container = widgets.VBox([
        widgets.HTML(header_html),
        widgets_dict['num_variations_slider'],
        widgets_dict['target_count_slider'],
        widgets_dict['intensity_slider'],
        widgets_dict['target_split_dropdown'],
        widgets_dict['cleanup_target_dropdown'],
        widgets_dict['balance_classes_checkbox'],
        widgets_dict['data_dir_input'],
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
        'widgets': widgets_dict,
        
        # Validation configuration
        'validation': {
            'ranges': {
                'num_variations': (1, 10),
                'target_count': (10, 10000),
                'intensity': (0.0, 1.0)
            },
            'required': [
                'num_variations_slider', 'target_count_slider', 'intensity_slider',
                'target_split_dropdown', 'cleanup_target_dropdown', 'data_dir_input'
            ],
            'backend_compatible': True
        },
        
        # Backend configuration mapping
        'backend_mapping': {
            'num_variations_slider': 'augmentation.num_variations',
            'target_count_slider': 'augmentation.target_count', 
            'intensity_slider': 'augmentation.intensity',
            'target_split_dropdown': 'augmentation.target_split',
            'cleanup_target_dropdown': 'cleanup.default_target',
            'balance_classes_checkbox': 'augmentation.balance_classes',
            'data_dir_input': 'data.dir'
        },
        
        # Form metadata
        'form_type': 'basic_options',
        'component_version': '2.0.0',
        'preserved_business_logic': True
    }