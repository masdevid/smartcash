"""
Path Section Component for Dataset Split UI.

This module provides the UI components for configuring output paths and options.
"""

from typing import Dict, Any
import ipywidgets as widgets
from IPython.display import display

def create_path_section(config: Dict[str, Any]) -> Dict[str, Any]:
    """Create the path configuration section using form container.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary containing the section widget and form components
    """
    # Extract config with defaults
    output_config = config.get('output', {})
    
    # Create form container for the section
    section = widgets.VBox([
        widgets.HTML("<h3>Output Paths</h3>")
    ], layout=widgets.Layout(
        width='100%',
        overflow='hidden'  # Prevent content from overflowing
    ))
    
    # Create form widgets
    train_dir = widgets.Text(
        value=output_config.get('train_dir', 'data/train'),
        description='Train Dir:',
        layout=widgets.Layout(width='auto')
    )
    
    val_dir = widgets.Text(
        value=output_config.get('val_dir', 'data/val'),
        description='Val Dir:',
        layout=widgets.Layout(width='auto')
    )
    
    test_dir = widgets.Text(
        value=output_config.get('test_dir', 'data/test'),
        description='Test Dir:',
        layout=widgets.Layout(width='auto')
    )
    
    # Create options group
    options_group = widgets.VBox([
        widgets.HTML("<h4>Options</h4>"),
        widgets.HBox([
            widgets.Checkbox(
                value=output_config.get('create_subdirs', True),
                description='Create subdirectories',
                indent=False,
                layout=widgets.Layout(width='50%')
            ),
            widgets.Checkbox(
                value=output_config.get('overwrite', False),
                description='Overwrite existing',
                indent=False,
                layout=widgets.Layout(width='50%')
            )
        ])
    ])
    
    # Add widgets to section
    section.children += (
        train_dir,
        val_dir,
        test_dir,
        options_group,
        widgets.HTML("<hr>")
    )
    
    # Create references to form controls
    create_subdirs = options_group.children[1].children[0]
    overwrite = options_group.children[1].children[1]
    
    return {
        'path_section': section,
        'train_dir': train_dir,
        'val_dir': val_dir,
        'test_dir': test_dir,
        'create_subdirs': create_subdirs,
        'overwrite': overwrite
    }
