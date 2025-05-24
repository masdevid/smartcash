"""
File: smartcash/ui/utils/layout_utils.py
Deskripsi: Layout utilities dengan backward compatibility dan fitur responsive baru
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional

# =============================================================================
# BACKWARD COMPATIBILITY - Existing Layout Constants
# =============================================================================

# Standard layouts (existing - unchanged)
STANDARD_LAYOUTS = {
    'header': widgets.Layout(margin='0 0 15px 0'),
    'section': widgets.Layout(margin='15px 0 10px 0'),
    'container': widgets.Layout(width='100%', padding='10px'),
    'output': widgets.Layout(
        width='100%',
        border='1px solid #ddd',
        min_height='100px',
        max_height='300px',
        margin='10px 0',
        overflow='auto'
    ),
    'button': widgets.Layout(margin='10px 0'),
    'button_small': widgets.Layout(margin='5px'),
    'hbox': widgets.Layout(
        display='flex',
        flex_flow='row wrap',
        align_items='center',
        width='100%'
    ),
    'vbox': widgets.Layout(
        display='flex',
        flex_flow='column',
        align_items='stretch',
        width='100%'
    )
}

# Layout untuk container utama (existing - unchanged)
MAIN_CONTAINER = widgets.Layout(
    width='100%',
    padding='10px'
)

# Layout untuk output widget (existing - unchanged)
OUTPUT_WIDGET = widgets.Layout(
    width='100%',
    border='1px solid #ddd',
    min_height='100px',
    margin='10px 0',
    padding='8px 4px'
)

# Layout untuk tombol (existing - unchanged)
BUTTON = widgets.Layout(
    margin='10px 0'
)

# Layout untuk tombol yang disembunyikan (existing - unchanged)
HIDDEN_BUTTON = widgets.Layout(
    margin='10px 0',
    display='none'
)

# Layout untuk input text (existing - unchanged)
TEXT_INPUT = widgets.Layout(
    width='60%',
    margin='10px 0'
)

# Layout untuk textarea (existing - unchanged)
TEXT_AREA = widgets.Layout(
    width='60%',
    height='150px',
    margin='10px 0'
)

# Layout untuk radio dan checkbox (existing - unchanged)
SELECTION = widgets.Layout(
    margin='10px 0'
)

# Layout untuk grup widget horizontal (existing - unchanged)
HORIZONTAL_GROUP = widgets.Layout(
    display='flex',
    flex_flow='row wrap',
    align_items='center',
    width='100%'
)

# Layout untuk grup widget vertikal (existing - unchanged)
VERTICAL_GROUP = widgets.Layout(
    display='flex',
    flex_flow='column',
    align_items='stretch',
    width='100%'
)

# Layout untuk divider (existing - unchanged)
DIVIDER = widgets.Layout(
    height='1px',
    border='0',
    border_top='1px solid #eee',
    margin='15px 0'
)

# Layout untuk kartu/card (existing - unchanged)
CARD = widgets.Layout(
    border='1px solid #ddd',
    border_radius='4px',
    padding='15px',
    margin='10px 0',
    width='100%'
)

# Layout untuk tab container (existing - unchanged)
TABS = widgets.Layout(
    width='100%',
    margin='10px 0'
)

# Layout untuk accordion (existing - unchanged)
ACCORDION = widgets.Layout(
    width='100%',
    margin='10px 0'
)

# =============================================================================
# BACKWARD COMPATIBILITY - Existing Functions
# =============================================================================

def create_divider(margin: str = "15px 0", color: str = "#eee", height: str = "1px") -> widgets.HTML:
    """
    Create divider horizontal (backward compatible + enhanced).
    
    Args:
        margin: CSS margin untuk divider (default sesuai existing)
        color: Warna divider (default sesuai existing)
        height: Tinggi divider
        
    Returns:
        HTML widget divider
    """
    return widgets.HTML(
        value=f"<hr style='margin: {margin}; border: 0; border-top: {height} solid {color};'>",
        layout=widgets.Layout(width='100%', margin='0', padding='0')
    )

# =============================================================================
# NEW RESPONSIVE UTILITIES
# =============================================================================

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

# =============================================================================
# ENHANCED RESPONSIVE LAYOUTS (New constants dengan responsive features)
# =============================================================================

# Responsive versions of existing layouts
RESPONSIVE_LAYOUTS = {
    'container': widgets.Layout(width='100%', max_width='100%', padding='10px', overflow='hidden'),
    'two_column_left': widgets.Layout(width='47%', margin='0', padding='4px', overflow='hidden'),
    'two_column_right': widgets.Layout(width='47%', margin='0', padding='4px', overflow='hidden'),
    'button_responsive': widgets.Layout(width='auto', max_width='150px', height='32px', margin='2px'),
    'dropdown_responsive': widgets.Layout(width='100%', max_width='100%', margin='3px 0', overflow='hidden'),
    'no_scroll': widgets.Layout(overflow='hidden', max_width='100%')
}