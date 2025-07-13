"""
File: smartcash/ui/dataset/augment/components/augmentation_types.py
Description: Augmentation types widget with preserved business logic

This component creates the augmentation types selection with all original
business logic preserved while following the new UI structure.
"""

import ipywidgets as widgets
from typing import Dict, Any, List
from smartcash.ui.core.decorators import handle_ui_errors
from ..constants import (
    AUGMENTATION_TYPES_OPTIONS, DEFAULT_AUGMENTATION_PARAMS,
    HELP_TEXT, AUGMENT_COLORS, SECTION_STYLES, AUGMENTATION_TIPS
)


def _create_info_panel(content: str, theme: str = 'augmentation_types') -> widgets.HTML:
    """Create info panel with preserved styling."""
    style_config = SECTION_STYLES.get(theme, SECTION_STYLES['augmentation_types'])
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


def _create_augmentation_tips() -> str:
    """Create augmentation tips content."""
    tips_formatted = []
    for tip in AUGMENTATION_TIPS[:4]:  # Show first 4 tips
        tips_formatted.append(f"• {tip}")
    
    return f"""
    <strong style="color: #2196f3; margin-bottom:4px">💡 Augmentation Tips:</strong><br>
    {"<br>".join(tips_formatted)}
    """


def _create_type_descriptions() -> str:
    """Create type descriptions content."""
    return """
    <strong style="color: #2196f3; margin-bottom:4px">🔄 Type Descriptions:</strong><br>
    • <strong style='color: #2196f3;'>Combined:</strong> Position + Lighting (Recommended)<br>
    • <strong style='color: #2196f3;'>Position:</strong> Flip, Rotate, Scale, Translate only<br>
    • <strong style='color: #2196f3;'>Lighting:</strong> Brightness, Contrast, HSV only<br>
    • <strong style='color: #2196f3;'>Custom:</strong> Manual parameter control
    """


@handle_ui_errors(error_component_title="Augmentation Types Creation Error")
def create_augmentation_types_widget() -> Dict[str, Any]:
    """
    Create augmentation types widget with preserved business logic.
    
    Features:
    - 🔄 All original augmentation type options
    - 🎨 Preserved styling and responsive layout
    - ✅ Multi-selection with validation
    - 💡 Type descriptions and guidance
    - 🎯 Default selection logic
    
    Returns:
        Dictionary containing container, widgets, and metadata
    """
    
    # Create augmentation types selection widget
    widgets_dict = {
        'augmentation_types_select': widgets.SelectMultiple(
            options=AUGMENTATION_TYPES_OPTIONS,
            value=DEFAULT_AUGMENTATION_PARAMS['types'],
            description='Types:',
            tooltip=HELP_TEXT['types'],
            style={'description_width': '80px'},
            layout=widgets.Layout(
                width='100%',
                height='120px'  # Show all options without scrolling
            )
        ),
        
        # Preview mode selection
        'preview_mode_checkbox': widgets.Checkbox(
            value=False,
            description='Enable Live Preview Mode',
            tooltip='Preview augmentation results in real-time',
            indent=False,
            layout=widgets.Layout(width='auto', margin='6px 0')
        ),
        
        # Advanced mode toggle
        'custom_mode_checkbox': widgets.Checkbox(
            value=False,
            description='Custom Parameter Mode',
            tooltip='Enable manual control of individual parameters',
            indent=False,
            layout=widgets.Layout(width='auto', margin='6px 0')
        )
    }
    
    # Create section headers
    main_header = widgets.HTML(f"""
    <h6 style='color: {AUGMENT_COLORS["primary"]}; margin: 6px 0; font-size: 12px; font-weight: 600;'>
        🔄 Augmentation Types
    </h6>
    """)
    
    options_header = widgets.HTML(f"""
    <h6 style='color: {AUGMENT_COLORS["info"]}; margin: 8px 0 4px 0; font-size: 11px; font-weight: 600;'>
        📋 Available Options
    </h6>
    """)
    
    modes_header = widgets.HTML(f"""
    <h6 style='color: {AUGMENT_COLORS["success"]}; margin: 8px 0 4px 0; font-size: 11px; font-weight: 600;'>
        ⚙️ Mode Settings
    </h6>
    """)
    
    # Create info panels
    type_info = _create_info_panel(_create_type_descriptions(), 'augmentation_types')
    tips_info = _create_info_panel(_create_augmentation_tips(), 'augmentation_types')
    
    # Create container with organized layout
    container = widgets.VBox([
        main_header,
        
        # Types selection
        options_header,
        widgets_dict['augmentation_types_select'],
        type_info,
        
        # Mode settings
        modes_header,
        widgets_dict['preview_mode_checkbox'],
        widgets_dict['custom_mode_checkbox'],
        
        # Tips
        tips_info
        
    ], layout=widgets.Layout(
        width='100%',
        padding='10px',
        display='flex',
        flex_flow='column',
        align_items='stretch',
        gap='4px'
    ))
    
    # Add interaction logic
    def _on_type_change(change):
        """Handle augmentation type selection changes."""
        selected_types = list(change['new'])
        
        # Auto-enable custom mode if multiple types selected
        if len(selected_types) > 1 and 'custom' in selected_types:
            widgets_dict['custom_mode_checkbox'].value = True
        elif len(selected_types) == 1 and selected_types[0] in ['combined', 'position', 'lighting']:
            widgets_dict['custom_mode_checkbox'].value = False
    
    def _on_custom_mode_change(change):
        """Handle custom mode toggle."""
        if change['new']:
            # If custom mode enabled, add 'custom' to selection if not present
            current_types = list(widgets_dict['augmentation_types_select'].value)
            if 'custom' not in current_types:
                widgets_dict['augmentation_types_select'].value = current_types + ['custom']
    
    # Attach event handlers
    widgets_dict['augmentation_types_select'].observe(_on_type_change, names='value')
    widgets_dict['custom_mode_checkbox'].observe(_on_custom_mode_change, names='value')
    
    return {
        'container': container,
        'widgets': widgets_dict,
        
        # Type options for reference
        'available_types': [option[1] for option in AUGMENTATION_TYPES_OPTIONS],
        'default_types': DEFAULT_AUGMENTATION_PARAMS['types'],
        
        # Validation configuration
        'validation': {
            'min_types': 1,
            'max_types': len(AUGMENTATION_TYPES_OPTIONS),
            'required': ['augmentation_types_select'],
            'backend_compatible': True
        },
        
        # Backend configuration mapping
        'backend_mapping': {
            'augmentation_types_select': 'augmentation.types',
            'preview_mode_checkbox': 'backend.preview_mode',
            'custom_mode_checkbox': 'backend.custom_mode'
        },
        
        # Business logic helpers
        'type_combinations': {
            'combined': ['position', 'lighting'],
            'position': ['horizontal_flip', 'rotation_limit', 'translate_limit', 'scale_limit'],
            'lighting': ['brightness_limit', 'contrast_limit', 'hsv_hue', 'hsv_saturation'],
            'custom': []  # User-defined combination
        },
        
        # Form metadata
        'form_type': 'augmentation_types',
        'component_version': '2.0.0',
        'preserved_business_logic': True,
        'interaction_handlers': ['type_selection', 'mode_toggle']
    }