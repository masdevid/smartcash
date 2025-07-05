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
        
        # Ensure the widget has a layout
        if not hasattr(self.widget, 'layout') or self.widget.layout is None:
            self.widget.layout = widgets.Layout()
    
    @property
    def layout(self):
        """Return the widget's layout."""
        return self.widget.layout

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
    state = {
        'layout_type': LayoutType[layout_type.upper()] if isinstance(layout_type, str) else layout_type,
        'form_items': [],
        'form_container': None,
        'grid_columns': grid_columns,
        'grid_template_areas': grid_template_areas,
        'grid_auto_flow': grid_auto_flow,
        'container_padding': container_padding,
        'gap': gap
    }
    state.update(layout_kwargs)

    def create_layout():
        l_type = state['layout_type']
        layout_props = {
            'width': '100%',
            'padding': state['container_padding'],
            'gap': state['gap'],
            **{k: v for k, v in state.items() if k not in ['layout_type', 'form_items', 'form_container', 'grid_columns', 'grid_template_areas', 'grid_auto_flow', 'container_padding', 'gap']}
        }

        if l_type == LayoutType.COLUMN:
            return widgets.VBox([], layout=widgets.Layout(display='flex', flex_flow='column', align_items='stretch', **layout_props))
        elif l_type == LayoutType.ROW:
            layout_props['flex_flow'] = 'row wrap'
            return widgets.HBox([], layout=widgets.Layout(display='flex', **layout_props))
        elif l_type == LayoutType.GRID:
            if state['grid_columns']:
                layout_props['grid_template_columns'] = f"repeat({state['grid_columns']}, 1fr)" if isinstance(state['grid_columns'], int) else state['grid_columns']
            if state['grid_template_areas']:
                layout_props['grid_template_areas'] = ' '.join([f'"{area}"' for area in state['grid_template_areas']])
            if state['grid_auto_flow']:
                layout_props['grid_auto_flow'] = state['grid_auto_flow']
            return widgets.GridBox([], layout=widgets.Layout(display='grid', **layout_props))

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
        if not isinstance(widget, FormItem):
            widget = FormItem(
                widget=widget,
                width=width, height=height, flex=flex, grid_area=grid_area,
                justify_content=justify_content, align_items=align_items
            )
        
        if index is not None:
            state['form_items'].insert(index, widget)
        else:
            state['form_items'].append(widget)
        
        update_container()
    
    def update_container():
        """Update the container with current items and layout."""
        children = []
        for item in state['form_items']:
            item.widget.layout.width = item.width
            item.widget.layout.height = item.height
            if state['layout_type'] == LayoutType.GRID:
                item.widget.layout.grid_area = item.grid_area
            else:
                item.widget.layout.flex = item.flex
            children.append(item.widget)
        state['form_container'].children = tuple(children)
    
    def set_layout(
        new_layout_type: Union[LayoutType, str],
        **kwargs
    ) -> None:
        """Change the layout type dynamically."""
        state['layout_type'] = LayoutType[new_layout_type.upper()] if isinstance(new_layout_type, str) else new_layout_type
        state.update(kwargs)
        state['form_container'] = create_layout()
        container.children = (state['form_container'],)
        update_container()
    
    container = widgets.Box(
        layout=widgets.Layout(width='100%', margin=container_margin, padding='0', overflow='visible')
    )
    set_layout(state['layout_type'])

    return {
        'container': container,
        'add_item': add_item,
        'set_layout': set_layout,
        'get_form_container': lambda: state['form_container'],
        'get_items': lambda: state['form_items']
    }
