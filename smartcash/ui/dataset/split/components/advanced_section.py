"""
Advanced Section Component for Dataset Split UI.

This module provides the UI components for advanced configuration options.
"""

from typing import Dict, Any
import ipywidgets as widgets
from IPython.display import display

def create_advanced_section(config: Dict[str, Any]) -> Dict[str, Any]:
    """Create the advanced configuration section using form container.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary containing the section widget and form components
    """
    # Extract config with defaults
    data_config = config.get('data', {})
    adv_config = config.get('advanced', {})
    
    # Create form container for the section
    section = widgets.VBox([
        widgets.HTML("<h3>Advanced Settings</h3>")
    ], layout=widgets.Layout(
        width='100%',
        overflow='hidden'  # Prevent content from overflowing
    ))
    
    # Data splitting section
    splitting_group = widgets.VBox([
        widgets.HTML("<h4>Data Splitting</h4>"),
        widgets.HBox([
            widgets.IntText(
                value=data_config.get('seed', 42),
                description='Random Seed:',
                layout=widgets.Layout(width='150px')
            ),
            widgets.Checkbox(
                value=data_config.get('shuffle', True),
                description='Shuffle data',
                indent=False,
                layout=widgets.Layout(width='150px')
            ),
            widgets.Checkbox(
                value=data_config.get('stratify', False),
                description='Stratified split',
                indent=False,
                layout=widgets.Layout(width='150px')
            )
        ])
    ])
    
    # File operations section
    file_ops_group = widgets.VBox([
        widgets.HTML("<h4>File Operations</h4>"),
        widgets.VBox([
            widgets.HBox([
                widgets.Checkbox(
                    value=adv_config.get('use_relative_paths', True),
                    description='Use relative paths',
                    indent=False,
                    layout=widgets.Layout(width='50%')
                ),
                widgets.Checkbox(
                    value=adv_config.get('preserve_structure', True),
                    description='Preserve directory structure',
                    indent=False,
                    layout=widgets.Layout(width='50%')
                )
            ]),
            widgets.HBox([
                widgets.Checkbox(
                    value=adv_config.get('symlink', False),
                    description='Create symlinks',
                    indent=False,
                    layout=widgets.Layout(width='50%')
                ),
                widgets.Checkbox(
                    value=adv_config.get('backup', True),
                    description='Create backup',
                    indent=False,
                    layout=widgets.Layout(width='50%')
                )
            ]),
            widgets.HBox([
                widgets.Text(
                    value=adv_config.get('backup_dir', 'backups'),
                    description='Backup dir:',
                    layout=widgets.Layout(width='100%', margin='0 0 0 25px'),
                    disabled=not adv_config.get('backup', True)
                )
            ])
        ])
    ])
    
    # Add widgets to section
    section.children += (
        splitting_group,
        file_ops_group,
        widgets.HTML("<hr>")
    )
    
    # Create references to form controls
    seed = splitting_group.children[1].children[0]
    shuffle = splitting_group.children[1].children[1]
    stratify = splitting_group.children[1].children[2]
    
    use_relative_paths = file_ops_group.children[1].children[0].children[0]
    preserve_structure = file_ops_group.children[1].children[0].children[1]
    symlink = file_ops_group.children[1].children[1].children[0]
    backup = file_ops_group.children[1].children[1].children[1]
    backup_dir = file_ops_group.children[1].children[2].children[0]
    
    # Add backup toggle handler
    def on_backup_change(change):
        backup_dir.disabled = not change['new']
    
    backup.observe(on_backup_change, 'value')
    
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
