"""
File: smartcash/ui/dataset/split/components/ui_form.py

Form components for dataset split configuration.
Uses shared components from the UI components directory.
"""

from typing import Dict, Any, Callable
import ipywidgets as widgets
from smartcash.ui.components import create_section_title as create_shared_section_title
from smartcash.ui.components.save_reset_buttons import create_save_reset_buttons

# Constants
STYLES = {
    'section': {
        'border': '1px solid #e0e0e0',
        'border_radius': '4px',
        'padding': '15px',
        'margin': '10px 0',
        'width': '100%'
    },
    'slider': {
        'width': '90%',
        'margin': '10px 0'
    },
    'input': {
        'width': '90%',
        'margin': '5px 0'
    }
}

def create_section_title(title: str, level: int = 4) -> widgets.HTML:
    """Create a styled section title using the shared component.
    
    Args:
        title: Section title text
        level: Heading level (2-6), default is 4
        
    Returns:
        widgets.HTML: Styled title component
    """
    return create_shared_section_title(title=title, level=level)

def create_save_reset_buttons() -> Dict[str, Any]:
    """Create save and reset buttons using shared component.
    
    Returns:
        Dictionary containing buttons and their container with keys:
        - container: VBox container with buttons
        - save_button: Save button widget
        - reset_button: Reset button widget
        - sync_info: Sync info widget (None if not used)
    """
    # Import inside function to avoid circular imports
    from smartcash.ui.components.save_reset_buttons import create_save_reset_buttons as create_shared_buttons
    
    # Call the shared component with the correct parameters
    return create_shared_buttons(
        save_label='Save',
        reset_label='Reset',
        button_width='100px',
        container_width='100%',
        save_tooltip='Save current split configuration',
        reset_tooltip='Reset to default values',
        with_sync_info=False
    )

def create_split_form(config: Dict[str, Any]) -> Dict[str, Any]:
    """Create form components for dataset split configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary of form components
    """
    split_ratios = config.get('data', {}).get('split_ratios', {})
    split_settings = config.get('split_settings', {})
    data_config = config.get('data', {})
    
    # Helper functions for component creation
    def create_ratio_slider(value: float, min_val: float, max_val: float, desc: str) -> widgets.FloatSlider:
        # Create a copy of the slider style to avoid modifying the original
        slider_style = STYLES['slider'].copy()
        
        return widgets.FloatSlider(
            value=value,
            min=min_val,
            max=max_val,
            step=0.05,
            description=desc,
            readout_format='.2f',
            layout=widgets.Layout(**slider_style)
        )
    
    def create_text_input(value: str, desc: str, width: str = '90%') -> widgets.Text:
        # Create a copy of the input style and update width if needed
        input_style = STYLES['input'].copy()
        if 'width' not in input_style:
            input_style['width'] = width
            
        return widgets.Text(
            value=str(value),
            description=desc,
            layout=widgets.Layout(**input_style)
        )
    
    def create_checkbox(value: bool, desc: str) -> widgets.Checkbox:
        return widgets.Checkbox(
            value=bool(value),
            description=desc,
            indent=False,
            layout=widgets.Layout(margin='5px 0')
        )
    
    def create_int_input(value: int, desc: str) -> widgets.IntText:
        # Create a copy of the input style and update width
        input_style = STYLES['input'].copy()
        input_style['width'] = '50%'  # Override width for int inputs
        
        return widgets.IntText(
            value=int(value) if value is not None else 0,
            description=desc,
            layout=widgets.Layout(**input_style)
        )
    
    # Create form components
    form_components = {
        # Ratio sliders
        'train_slider': create_ratio_slider(split_ratios.get('train', 0.7), 0.5, 0.9, 'Train:'),
        'valid_slider': create_ratio_slider(split_ratios.get('valid', 0.15), 0.05, 0.3, 'Valid:'),
        'test_slider': create_ratio_slider(split_ratios.get('test', 0.15), 0.05, 0.3, 'Test:'),
        'total_label': widgets.HTML(
            value="<div style='padding: 10px; color: #28a745; font-weight: bold;'>Total: 1.00</div>",
            layout=widgets.Layout(width='90%')
        ),
        
        # Checkboxes
        'stratified_checkbox': create_checkbox(
            data_config.get('stratified_split', True),
            'Stratified Split'
        ),
        'backup_checkbox': create_checkbox(
            split_settings.get('backup_before_split', True),
            'Backup Before Split'
        ),
        
        # Input fields
        'random_seed': create_int_input(
            data_config.get('random_seed', 42),
            'Random Seed:'
        ),
        'dataset_path': create_text_input(
            split_settings.get('dataset_path', 'data'),
            'Dataset Path:'
        ),
        'preprocessed_path': create_text_input(
            split_settings.get('preprocessed_path', 'data/preprocessed'),
            'Preprocessed:'
        ),
        'backup_dir': create_text_input(
            split_settings.get('backup_dir', 'data/splits_backup'),
            'Backup Directory:'
        )
    }
    
    # Reuse existing shared components
    save_reset_buttons = create_save_reset_buttons()
    
    
    return form_components


def create_ratio_section(components: Dict[str, Any]) -> widgets.VBox:
    """Create the ratio split section.
    
    Args:
        components: Dictionary of form components
        
    Returns:
        widgets.VBox: Container with ratio split controls
    """
    ratio_components = [
        components['train_slider'],
        components['valid_slider'],
        components['test_slider'],
        components['total_label'],
        widgets.HBox([
            components['stratified_checkbox'],
            components['random_seed']
        ])
    ]
    
    return widgets.VBox(
        [create_section_title('Split Ratios')] + ratio_components,
        layout=widgets.Layout(**STYLES['section'])
    )


def create_path_section(components: Dict[str, Any]) -> widgets.VBox:
    """Create the paths and backup section.
    
    Args:
        components: Dictionary of form components
        
    Returns:
        widgets.VBox: Container with path and backup controls
    """
    path_components = [
        components['dataset_path'],
        components['preprocessed_path'],
        components['backup_dir'],
        components['backup_checkbox']
    ]
    
    return widgets.VBox(
        [create_section_title('Paths & Backup')] + path_components,
        layout=widgets.Layout(**STYLES['section'])
    )