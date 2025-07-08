"""
Path Section Component for Dataset Split UI.

This module provides the UI components for configuring output paths and options.
"""

from typing import Dict, Any
import ipywidgets as widgets
from IPython.display import display

def create_path_section(config: Dict[str, Any]) -> Dict[str, Any]:
    """Create the path configuration section.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary containing the section widget and form components
    """
    # Extract config with defaults
    output_config = config.get('output', {})
    
    # Create form widgets
    train_dir = widgets.Text(
        value=output_config.get('train_dir', 'data/train'),
        description='Train Dir:',
        layout=widgets.Layout(width='400px')
    )
    
    val_dir = widgets.Text(
        value=output_config.get('val_dir', 'data/val'),
        description='Validation Dir:',
        layout=widgets.Layout(width='400px')
    )
    
    test_dir = widgets.Text(
        value=output_config.get('test_dir', 'data/test'),
        description='Test Dir:',
        layout=widgets.Layout(width='400px')
    )
    
    create_subdirs = widgets.Checkbox(
        value=output_config.get('create_subdirs', True),
        description='Create subdirectories',
        indent=False
    )
    
    overwrite = widgets.Checkbox(
        value=output_config.get('overwrite', False),
        description='Overwrite existing files',
        indent=False
    )
    
    # Create section
    section = widgets.VBox([
        widgets.HTML("<h3>Output Paths</h3>"),
        train_dir,
        val_dir,
        test_dir,
        widgets.HBox([create_subdirs, overwrite]),
        widgets.HTML("<hr>")
    ])
    
    return {
        'path_section': section,
        'train_dir': train_dir,
        'val_dir': val_dir,
        'test_dir': test_dir,
        'create_subdirs': create_subdirs,
        'overwrite': overwrite
    }
