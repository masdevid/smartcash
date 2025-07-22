"""
Ratio Section Component for Dataset Split UI.

This module provides the UI components for configuring dataset split ratios.
"""

from typing import Dict, Any
import ipywidgets as widgets
from IPython.display import display

def create_ratio_section(config: Dict[str, Any]) -> Dict[str, Any]:
    """Create the ratio configuration section using form container.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary containing the section widget and form components
    """
    # Extract config with defaults
    data_config = config.get('data', {})
    ratios = data_config.get('split_ratios', {
        'train': 0.7,
        'val': 0.15,
        'test': 0.15
    })
    
    # Create form container for the section
    section = widgets.VBox([
        widgets.HTML("<h3>Split Ratios</h3>")
    ], layout=widgets.Layout(
        width='100%',
        overflow='hidden'  # Prevent content from overflowing
    ))
    
    # Create form widgets
    train_ratio = widgets.FloatSlider(
        value=ratios.get('train', 0.7),
        min=0.0,
        max=1.0,
        step=0.05,
        description='Train:',
        continuous_update=False,
        layout=widgets.Layout(width='auto')
    )
    
    val_ratio = widgets.FloatSlider(
        value=ratios.get('val', 0.15),
        min=0.0,
        max=1.0,
        step=0.05,
        description='Validation:',
        continuous_update=False,
        layout=widgets.Layout(width='auto')
    )
    
    test_ratio = widgets.FloatSlider(
        value=ratios.get('test', 0.15),
        min=0.0,
        max=1.0,
        step=0.05,
        description='Test:',
        continuous_update=False,
        layout=widgets.Layout(width='auto')
    )
    
    # Create info text
    info_text = widgets.HTML(
        value="<i>Note: Ratios will be automatically normalized to sum to 1.0</i>"
    )
    
    # Add widgets to section
    section.children += (
        train_ratio,
        val_ratio,
        test_ratio,
        info_text,
        widgets.HTML("<hr>")
    )
    
    # Add ratio change handler
    def _on_ratio_change(change):
        total = train_ratio.value + val_ratio.value + test_ratio.value
        if abs(total - 1.0) > 0.001:  # Allow for floating point errors
            factor = 1.0 / total if total > 0 else 1.0
            with train_ratio.hold_trait_notifications():
                train_ratio.value = round(train_ratio.value * factor, 2)
                val_ratio.value = round(val_ratio.value * factor, 2)
                test_ratio.value = round(test_ratio.value * factor, 2)
    
    train_ratio.observe(_on_ratio_change, 'value')
    val_ratio.observe(_on_ratio_change, 'value')
    test_ratio.observe(_on_ratio_change, 'value')
    
    return {
        'ratio_section': section,
        'train_ratio': train_ratio,
        'val_ratio': val_ratio,
        'test_ratio': test_ratio
    }
