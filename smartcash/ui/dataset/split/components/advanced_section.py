"""
Advanced Section Component for Dataset Split UI.

This module provides the UI components for advanced configuration options.
"""

from typing import Dict, Any
import ipywidgets as widgets
from IPython.display import display

def create_advanced_section(config: Dict[str, Any]) -> Dict[str, Any]:
    """Create the advanced configuration section.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary containing the section widget and form components
    """
    # Extract config with defaults
    data_config = config.get('data', {})
    adv_config = config.get('advanced', {})
    
    # Create form widgets
    seed = widgets.IntText(
        value=data_config.get('seed', 42),
        description='Random Seed:',
        layout=widgets.Layout(width='200px')
    )
    
    shuffle = widgets.Checkbox(
        value=data_config.get('shuffle', True),
        description='Shuffle data before splitting',
        indent=False
    )
    
    stratify = widgets.Checkbox(
        value=data_config.get('stratify', False),
        description='Stratified split (maintain class distribution)',
        indent=False
    )
    
    use_relative_paths = widgets.Checkbox(
        value=adv_config.get('use_relative_paths', True),
        description='Use relative paths',
        indent=False
    )
    
    preserve_structure = widgets.Checkbox(
        value=adv_config.get('preserve_structure', True),
        description='Preserve directory structure',
        indent=False
    )
    
    symlink = widgets.Checkbox(
        value=adv_config.get('symlink', False),
        description='Create symlinks instead of copying files',
        indent=False
    )
    
    backup = widgets.Checkbox(
        value=adv_config.get('backup', True),
        description='Create backup before overwriting',
        indent=False
    )
    
    backup_dir = widgets.Text(
        value=adv_config.get('backup_dir', 'backups'),
        description='Backup directory:',
        layout=widgets.Layout(width='400px')
    )
    
    # Create section
    section = widgets.VBox([
        widgets.HTML("<h3>Advanced Settings</h3>"),
        widgets.HTML("<h4>Data Splitting</h4>"),
        widgets.HBox([seed, shuffle, stratify]),
        widgets.HTML("<h4>File Operations</h4>"),
        use_relative_paths,
        preserve_structure,
        symlink,
        widgets.HBox([backup, backup_dir] if backup.value else [backup]),
    ])
    
    # Show/hide backup dir based on backup checkbox
    def on_backup_change(change):
        if change['name'] == 'value':
            if change['new']:
                children = list(section.children)
                children[-1] = widgets.HBox([backup, backup_dir])
                section.children = children
            else:
                children = list(section.children)
                children[-1] = widgets.HBox([backup])
                section.children = children
    
    backup.observe(on_backup_change)
    
    return {
        'advanced_section': section,
        'seed': seed,
        'shuffle': shuffle,
        'stratify': stratify,
        'use_relative_paths': use_relative_paths,
        'preserve_structure': preserve_structure,
        'symlink': symlink,
        'backup': backup,
        'backup_dir': backup_dir
    }
