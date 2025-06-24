"""
Tips Panel Component

A reusable tips panel component with modern styling and flexible content.
"""

import ipywidgets as widgets
from typing import List, Dict, Optional, Union


def create_tips_panel(
    title: str = "💡 Tips & Requirements",
    tips: Optional[List[Union[str, List[str]]]] = None,
    gradient_start: str = "#e3f2fd",
    gradient_end: str = "#f3e5f5",
    border_color: str = "#2196f3",
    title_color: str = "#1976d2",
    text_color: str = "#424242",
    columns: int = 2,
    margin: str = "20px 0"
) -> widgets.HTML:
    """
    Create a modern tips panel with gradient background and flexible content.
    
    Args:
        title: Panel title with emoji
        tips: List of tips. Can be strings or lists of strings for multi-column layout
        gradient_start: Start color of the gradient background
        gradient_end: End color of the gradient background
        border_color: Left border color
        title_color: Title text color
        text_color: Tips text color
        columns: Number of columns for tips layout (1-4)
        margin: CSS margin for the panel
        
    Returns:
        IPython HTML widget containing the tips panel
    """
    # Default tips if none provided
    if tips is None:
        tips = [
            ["Pastikan Google Drive memiliki ruang minimal 12GB", "Proses setup memerlukan waktu 1-2 menit"],
            ["Setup akan membuat folder struktur data lengkap", "Koneksi internet stabil diperlukan"]
        ]
    
    # Ensure tips is a list of lists
    if all(isinstance(tip, str) for tip in tips):
        # If single list of strings, split into columns
        tips = [tips[i::columns] for i in range(columns)]
    
    # Generate columns HTML
    columns_html = ""
    for col_tips in tips:
        if not col_tips:  # Skip empty columns
            continue
            
        items_html = "\n".join(
            f'<li style="margin: 4px 0; line-height: 1.2;">{tip}</li>'
            for tip in col_tips
            if tip  # Skip empty tips
        )
        
        columns_html += f"""
        <div style="flex: 1; min-width: 180px;">
            <ul style="margin: 0; padding-left: 16px; color: {text_color}; font-size: 12px;">
                {items_html}
            </ul>
        </div>
        """
    
    # Generate the complete HTML
    html = f"""
    <div style="
        background: linear-gradient(135deg, {gradient_start} 0%, {gradient_end} 100%);
        padding: 12px 16px;
        border-radius: 8px;
        border-left: 3px solid {border_color};
        margin: {margin};
    ">
        <h4 style="
            margin: 0 0 8px 0;
            color: {title_color};
            display: flex;
            align-items: center;
            font-size: 14px;
            font-weight: 600;
        ">
            {title}
        </h4>
        <div style="
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 6px;
        ">
            {columns_html}
        </div>
    </div>
    """
    
    return widgets.HTML(value=html)


def create_multi_tips_panels(
    panels: List[Dict[str, any]],
    direction: str = "vertical"
) -> Union[widgets.VBox, widgets.HBox]:
    """
    Create multiple tips panels in a vertical or horizontal layout.
    
    Args:
        panels: List of panel configurations (kwargs for create_tips_panel)
        direction: 'vertical' or 'horizontal' layout
        
    Returns:
        VBox or HBox containing the tips panels
    """
    panel_widgets = [create_tips_panel(**panel) for panel in panels]
    
    if direction.lower() == "horizontal":
        return widgets.HBox(panel_widgets, layout=widgets.Layout(align_items='stretch'))
    return widgets.VBox(panel_widgets)
