"""
File: smartcash/ui/components/form_container.py
Deskripsi: Reusable form container with responsive layout options (row, column, grid)
"""

from enum import Enum, auto
from typing import Dict, Any, Optional, List, Literal, Union, Tuple
import ipywidgets as widgets
from IPython.display import display

class LayoutType(Enum):
    """Available layout types for the form container."""
    COLUMN = auto()  # Vertical stacking (default)
    ROW = auto()     # Horizontal arrangement
    GRID = auto()    # Grid-based layout

class FormItem:
    """Wrapper for form items with optional layout configuration."""
    
    @staticmethod
    def _validate_align_items(value: Optional[str]) -> str:
        """Validate and normalize align_items value.
        
        Args:
            value: The align_items value to validate
            
        Returns:
            str: A valid flexbox align-items value
        """
        if not value:
            return 'stretch'
            
        valid_values = {
            'flex-start', 'flex-end', 'center', 
            'baseline', 'stretch', 'inherit', 'initial', 'unset'
        }
        
        # Map common aliases to valid values
        alias_map = {
            'left': 'flex-start',
            'right': 'flex-end',
            'middle': 'center'
        }
        
        # Convert to lowercase and check if it's a known alias
        normalized = str(value).lower()
        if normalized in alias_map:
            return alias_map[normalized]
            
        # Return the value if valid, otherwise default to 'stretch'
        return value if value in valid_values else 'stretch'
    
    def __init__(
        self,
        widget: widgets.Widget,
        width: Optional[str] = None,
        height: Optional[str] = None,
        flex: Optional[Union[int, str]] = None,
        grid_area: Optional[str] = None,
        justify_content: Optional[str] = None,
        align_items: Optional[str] = None,
    ):
        self.widget = widget
        self.width = width
        self.height = height
        self.flex = flex
        self.grid_area = grid_area
        self.justify_content = justify_content
        self.align_items = self._validate_align_items(align_items)

def create_form_container(
    layout_type: Union[LayoutType, str] = LayoutType.COLUMN,
    container_margin: str = "8px 0",
    container_padding: str = "16px",
    gap: str = "8px",
    grid_columns: Optional[Union[int, str]] = None,
    grid_template_areas: Optional[List[str]] = None,
    grid_auto_flow: str = "row",
    **layout_kwargs
) -> Dict[str, Any]:
    """
    Create a form container with flexible layout options.
    
    Args:
        layout_type: Type of layout to use (COLUMN, ROW, or GRID)
        container_margin: Margin around the container (e.g., "8px 0")
        container_padding: Padding inside the container (e.g., "16px")
        gap: Spacing between items (e.g., "8px")
        grid_columns: For GRID layout, number of columns or template (e.g., 3 or "1fr 2fr")
        grid_template_areas: For GRID layout, defines named grid areas
        grid_auto_flow: For GRID layout, controls auto-placement of items ("row", "column", "row dense", "column dense")
        **layout_kwargs: Additional layout properties (e.g., width, height, etc.)
        
    Returns:
        Dictionary containing:
            - 'container': The main container widget
            - 'form_container': The container for form elements
            - 'add_item': Method to add items to the form
            - 'set_layout': Method to change layout dynamically
    """
    # Convert string layout type to enum if needed
    if isinstance(layout_type, str):
        layout_type = LayoutType[layout_type.upper()]
    
    # Store items to maintain order and configuration
    form_items: List[FormItem] = []
    
    # Create layout based on type
    def create_layout():
        if layout_type == LayoutType.COLUMN:
            return widgets.VBox(
                layout=widgets.Layout(
                    width='100%',
                    padding=container_padding,
                    gap=gap,
                    **layout_kwargs
                )
            )
        elif layout_type == LayoutType.ROW:
            return widgets.HBox(
                layout=widgets.Layout(
                    width='100%',
                    padding=container_padding,
                    gap=gap,
                    flex_flow='row wrap',
                    **layout_kwargs
                )
            )
        else:  # GRID
            grid_template_columns = (
                f"repeat({grid_columns}, 1fr)" 
                if isinstance(grid_columns, int) 
                else grid_columns or "1fr"
            )
            
            return widgets.GridBox(
                layout=widgets.Layout(
                    width='100%',
                    padding=container_padding,
                    gap=gap,
                    grid_template_columns=grid_template_columns,
                    grid_template_areas=' '.join(f'"{area}"' for area in grid_template_areas) if grid_template_areas else None,
                    grid_auto_flow=grid_auto_flow,
                    **layout_kwargs
                )
            )
    
    # Create form container
    form_container = create_layout()
    
    # Create main container
    container = widgets.VBox(
        [form_container],
        layout=widgets.Layout(
            width='100%',
            margin=container_margin,
            padding='0',
            overflow='visible'
        )
    )
    
    def add_item(
        widget: Union[widgets.Widget, FormItem],
        width: Optional[str] = None,
        height: Optional[str] = None,
        flex: Optional[Union[int, str]] = None,
        grid_area: Optional[str] = None,
        justify_content: Optional[str] = None,
        align_items: Optional[str] = None,
        index: Optional[int] = None
    ) -> None:
        """Add an item to the form container.
        
        Args:
            widget: The widget to add, or a FormItem instance
            width: Width of the item (e.g., '100%', '200px')
            height: Height of the item
            flex: Flex grow/shrink value
            grid_area: For GRID layout, the grid area name
            justify_content: Justify content for the item's container
            align_items: Align items for the item's container
            index: Position to insert the item (None for append)
        """
        if isinstance(widget, FormItem):
            form_item = widget
        else:
            form_item = FormItem(
                widget=widget,
                width=width,
                height=height,
                flex=flex,
                grid_area=grid_area,
                justify_content=justify_content,
                align_items=align_items
            )
        
        # Configure widget layout
        widget_layout = widget.layout if widget.layout else {}
        
        if layout_type == LayoutType.GRID and form_item.grid_area:
            widget_layout.grid_area = form_item.grid_area
        
        if form_item.width:
            widget_layout.width = form_item.width
        if form_item.height:
            widget_layout.height = form_item.height
        
        # Add to items list and update container
        if index is not None:
            form_items.insert(index, form_item)
        else:
            form_items.append(form_item)
        
        update_container()
    
    def update_container():
        """Update the container with current items and layout."""
        children = []
        
        for item in form_items:
            # Create a container for each item to support individual layout
            item_container = widgets.Box(
                [item.widget],
                layout=widgets.Layout(
                    width=item.width if layout_type != LayoutType.GRID else '100%',
                    height=item.height,
                    flex=item.flex if layout_type != LayoutType.GRID else None,
                    justify_content=item.justify_content,
                    align_items=item.align_items,
                    overflow='visible'
                )
            )
            
            if layout_type == LayoutType.GRID and item.grid_area:
                item_container.layout.grid_area = item.grid_area
            
            children.append(item_container)
        
        form_container.children = children
    
    def set_layout(
        new_layout_type: Union[LayoutType, str],
        **layout_kwargs
    ) -> None:
        """Change the layout type dynamically.
        
        Args:
            new_layout_type: New layout type (COLUMN, ROW, or GRID)
            **layout_kwargs: Additional layout properties
        """
        nonlocal form_container, layout_type
        
        # Update layout type
        if isinstance(new_layout_type, str):
            new_layout_type = LayoutType[new_layout_type.upper()]
        layout_type = new_layout_type
        
        # Create new container with the same items
        old_children = form_container.children
        form_container = create_layout()
        container.children = [form_container]
        
        # Re-add all items
        for item in form_items:
            add_item(item)
    
    return {
        'container': container,
        'form_container': form_container,
        'add_item': add_item,
        'set_layout': set_layout,
        'items': form_items  # Expose items for advanced manipulation
    }
