"""
File: smartcash/ui/utils/layout_utils.py
Deskripsi: Responsive layout utilities untuk prevent horizontal scroll dan improve UX
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional

def create_divider(margin: str = "10px 0", color: str = "#dee2e6", height: str = "1px") -> widgets.HTML:
    """
    Create responsive divider dengan consistent styling.
    
    Args:
        margin: CSS margin untuk divider
        color: Warna divider
        height: Tinggi divider
        
    Returns:
        HTML widget divider yang responsive
    """
    return widgets.HTML(
        value=f"""
        <div style="width: 100%; height: {height}; background-color: {color}; 
                   margin: {margin}; border: none; overflow: hidden;"></div>
        """,
        layout=widgets.Layout(width='100%', margin='0', padding='0')
    )

def create_responsive_container(
    children: list,
    container_type: str = "vbox",
    width: str = "100%",
    max_width: str = "100%",
    padding: str = "8px",
    margin: str = "0px",
    justify_content: str = "flex-start",
    align_items: str = "stretch"
) -> widgets.Widget:
    """
    Create responsive container yang prevent horizontal scroll.
    
    Args:
        children: List widget children
        container_type: 'vbox' atau 'hbox'
        width: Lebar container
        max_width: Maksimum lebar
        padding: Padding container
        margin: Margin container
        justify_content: CSS justify-content
        align_items: CSS align-items
        
    Returns:
        Responsive container widget
    """
    layout = widgets.Layout(
        width=width,
        max_width=max_width,
        padding=padding,
        margin=margin,
        justify_content=justify_content,
        align_items=align_items,
        overflow='hidden'  # Prevent horizontal scroll
    )
    
    if container_type.lower() == "hbox":
        return widgets.HBox(children, layout=layout)
    else:
        return widgets.VBox(children, layout=layout)

def create_responsive_two_column(
    left_content: widgets.Widget,
    right_content: widgets.Widget,
    left_width: str = "48%",
    right_width: str = "48%",
    gap: str = "2%",
    vertical_align: str = "flex-start"
) -> widgets.HBox:
    """
    Create responsive two-column layout yang tidak overflow.
    
    Args:
        left_content: Widget untuk kolom kiri
        right_content: Widget untuk kolom kanan
        left_width: Lebar kolom kiri
        right_width: Lebar kolom kanan
        gap: Gap antara kolom
        vertical_align: Vertical alignment
        
    Returns:
        HBox container dengan two-column layout
    """
    # Wrap content dengan responsive container
    left_wrapper = widgets.VBox(
        [left_content],
        layout=widgets.Layout(
            width=left_width,
            margin='0',
            padding='4px',
            overflow='hidden'
        )
    )
    
    right_wrapper = widgets.VBox(
        [right_content],
        layout=widgets.Layout(
            width=right_width,
            margin='0',
            padding='4px', 
            overflow='hidden'
        )
    )
    
    return widgets.HBox(
        [left_wrapper, right_wrapper],
        layout=widgets.Layout(
            width='100%',
            max_width='100%',
            justify_content='space-between',
            align_items=vertical_align,
            margin='0',
            padding='0',
            overflow='hidden'
        )
    )

def create_responsive_grid(
    items: list,
    columns: int = 2,
    gap: str = "8px",
    item_min_width: str = "200px"
) -> widgets.Widget:
    """
    Create responsive grid layout.
    
    Args:
        items: List widget items
        columns: Jumlah kolom
        gap: Gap antara items
        item_min_width: Minimum width per item
        
    Returns:
        Grid container yang responsive
    """
    rows = []
    
    # Split items ke dalam rows
    for i in range(0, len(items), columns):
        row_items = items[i:i+columns]
        
        # Wrap setiap item dengan responsive container
        wrapped_items = []
        for item in row_items:
            wrapper = widgets.VBox(
                [item],
                layout=widgets.Layout(
                    width=f"{100/columns:.1f}%",
                    min_width=item_min_width,
                    margin='2px',
                    padding='2px',
                    overflow='hidden'
                )
            )
            wrapped_items.append(wrapper)
        
        # Create row
        row = widgets.HBox(
            wrapped_items,
            layout=widgets.Layout(
                width='100%',
                justify_content='space-between',
                align_items='flex-start',
                margin='0',
                padding='0',
                overflow='hidden'
            )
        )
        rows.append(row)
    
    return widgets.VBox(
        rows,
        layout=widgets.Layout(
            width='100%',
            max_width='100%',
            margin='0',
            padding='0',
            overflow='hidden'
        )
    )

def apply_responsive_fixes(widget: widgets.Widget) -> widgets.Widget:
    """
    Apply responsive fixes ke widget yang sudah ada.
    
    Args:
        widget: Widget yang akan di-fix
        
    Returns:
        Widget dengan responsive fixes
    """
    if hasattr(widget, 'layout'):
        # Apply responsive layout fixes
        if not widget.layout.max_width:
            widget.layout.max_width = '100%'
        
        if not widget.layout.overflow:
            widget.layout.overflow = 'hidden'
        
        # Fix margin dan padding jika berlebihan
        if hasattr(widget.layout, 'margin') and widget.layout.margin and 'px' in widget.layout.margin:
            try:
                margin_val = int(widget.layout.margin.replace('px', ''))
                if margin_val > 20:  # Reduce excessive margins
                    widget.layout.margin = '10px'
            except:
                pass
    
    # Recursively fix children jika ada
    if hasattr(widget, 'children'):
        for child in widget.children:
            apply_responsive_fixes(child)
    
    return widget

def get_responsive_button_layout(width: str = "auto", max_width: str = "150px") -> widgets.Layout:
    """
    Get standardized responsive button layout.
    
    Args:
        width: Button width
        max_width: Maximum button width
        
    Returns:
        Layout object untuk button
    """
    return widgets.Layout(
        width=width,
        max_width=max_width,
        height='32px',
        margin='2px',
        overflow='hidden'
    )

def get_responsive_dropdown_layout(width: str = "100%", description_width: str = "80px") -> Dict[str, Any]:
    """
    Get standardized responsive dropdown layout dan style.
    
    Args:
        width: Dropdown width
        description_width: Description label width
        
    Returns:
        Dictionary dengan layout dan style
    """
    return {
        'layout': widgets.Layout(
            width=width,
            max_width='100%',
            margin='3px 0',
            overflow='hidden'
        ),
        'style': {'description_width': description_width}
    }

def create_section_header(
    title: str,
    icon: str = "📋",
    color: str = "#333",
    font_size: str = "16px",
    margin: str = "15px 0 8px 0"
) -> widgets.HTML:
    """
    Create responsive section header.
    
    Args:
        title: Header title
        icon: Header icon
        color: Text color
        font_size: Font size
        margin: Header margin
        
    Returns:
        HTML widget untuk section header
    """
    return widgets.HTML(
        value=f"""
        <h4 style='color: {color}; margin: {margin}; font-size: {font_size}; 
                   padding: 0; overflow: hidden; text-overflow: ellipsis;
                   white-space: nowrap;'>
            {icon} {title}
        </h4>
        """,
        layout=widgets.Layout(width='100%', margin='0', padding='0')
    )