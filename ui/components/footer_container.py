"""
Footer container component for consistent UI footers across the application.

This module provides a flexible footer container that supports multiple
info panels (InfoBox or InfoAccordion) with configurable flex layout.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Any, Optional, List, Union, Callable, Literal, TypedDict, ClassVar
import ipywidgets as widgets
from IPython.display import HTML

# Import shared components
from smartcash.ui.components.info.info_component import InfoAccordion, InfoBox

class PanelType(Enum):
    """Types of panels that can be added to the footer."""
    INFO_BOX = auto()
    INFO_ACCORDION = auto()

@dataclass
class PanelConfig:
    """Configuration for a single panel in the footer."""
    panel_type: PanelType
    title: str = ""
    content: str = ""
    style: str = "info"
    flex: str = "1"
    min_width: str = "200px"
    open_by_default: bool = True
    panel_id: Optional[str] = None

class FooterContainer:
    """Flexible footer container supporting multiple panels with configurable layout.
    
    This container can hold multiple panels of different types (InfoBox or InfoAccordion)
    with flexible layout configuration using flexbox.
    """
    
    def __init__(self, 
                 panels: Optional[List[PanelConfig]] = None,
                 style: Optional[Dict[str, str]] = None,
                 layout: Optional[Dict[str, str]] = None):
        """Initialize the footer container.
        
        Args:
            panels: List of panel configurations to add initially
            style: CSS styles for the container
            layout: Layout configuration for the container
        """
        self._panels: Dict[str, Dict[str, Any]] = {}
        self.container = None
        self.style = style or {}
        self.layout_config = layout or {
            'display': 'flex',
            'flex_flow': 'row wrap',
            'align_items': 'stretch',
            'justify_content': 'space-between',
            'width': '100%',
            'border': '1px solid #e0e0e0',
            'margin': '10px 0 0 0',
            'padding': '10px',
            'background': '#f9f9f9'
        }
        
        # Add initial panels if provided
        if panels:
            for panel in panels:
                self.add_panel(panel)
        
        # Initialize tips if enabled
        if show_tips:
            self._init_tips(tips_title, tips_content)
    
    def add_panel(self, config: PanelConfig) -> str:
        """Add a new panel to the footer.
        
        Args:
            config: Configuration for the new panel
            
        Returns:
            str: The ID of the created panel
        """
        # Generate ID if not provided
        panel_id = config.panel_id or f"panel_{len(self._panels) + 1}"
        
        # Create the panel based on type
        if config.panel_type == PanelType.INFO_BOX:
            panel = InfoBox(
                title=config.title,
                content=config.content,
                style=config.style
            )
        else:  # PanelType.INFO_ACCORDION
            panel = InfoAccordion(
                title=config.title,
                content=config.content,
                style=config.style,
                open_by_default=config.open_by_default
            )
        
        # Configure panel layout
        panel.layout = widgets.Layout(
            flex=config.flex,
            min_width=config.min_width,
            margin='0 5px 10px 5px',
            overflow='hidden'
        )
        
        # Store panel reference
        self._panels[panel_id] = {
            'widget': panel,
            'config': config
        }
        
        # Update container
        self._update_container()
        
        return panel_id
        
    def remove_panel(self, panel_id: str) -> None:
        """Remove a panel from the footer.
        
        Args:
            panel_id: ID of the panel to remove
        """
        if panel_id in self._panels:
            del self._panels[panel_id]
            self._update_container()
    
    def update_panel(self, panel_id: str, **kwargs) -> None:
        """Update a panel's configuration.
        
        Args:
            panel_id: ID of the panel to update
            **kwargs: Attributes to update (title, content, style, etc.)
        """
        if panel_id in self._panels:
            panel_info = self._panels[panel_id]
            config = panel_info['config']
            
            # Update config
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)
            
            # Update widget if needed
            if 'title' in kwargs:
                panel_info['widget'].title = kwargs['title']
            if 'content' in kwargs:
                panel_info['widget'].content = kwargs['content']
            if 'style' in kwargs:
                panel_info['widget'].style = kwargs['style']
            
            # Update layout if needed
            if 'flex' in kwargs or 'min_width' in kwargs:
                layout = panel_info['widget'].layout or {}
                if 'flex' in kwargs:
                    layout.flex = kwargs['flex']
                if 'min_width' in kwargs:
                    layout.min_width = kwargs['min_width']
    
    def get_panel(self, panel_id: str) -> Optional[Any]:
        """Get a panel widget by ID.
        
        Args:
            panel_id: ID of the panel to get
            
        Returns:
            Optional[Any]: The panel widget or None if not found
        """
        return self._panels.get(panel_id, {}).get('widget')
    
    def _update_container(self) -> None:
        """Update the container with current panels."""
        if not self._panels:
            self.container = widgets.VBox(layout=widgets.Layout(display='none'))
            return
            
        # Create a horizontal box for the panels
        panel_widgets = [
            panel['widget'].container 
            for panel in self._panels.values()
            if panel['widget'] is not None
        ]
        
        if not panel_widgets:
            return
            
        # Create flex container for panels
        flex_container = widgets.HBox(
            panel_widgets,
            layout=widgets.Layout(
                width='100%',
                flex_flow='row wrap',
                align_items='stretch',
                justify_content='flex-start',
                margin='0 -5px'  # Negative margin to counteract panel margins
            )
        )
        
        # Create the main container
        self.container = widgets.VBox(
            [flex_container],
            layout=widgets.Layout(
                **self.layout_config,
                **self.style
            )
        )
    
    def show_panel(self, panel_id: str, visible: bool = True) -> None:
        """Show or hide a panel.
        
        Args:
            panel_id: ID of the panel to show/hide
            visible: Whether to show (True) or hide (False) the panel
        """
        if panel_id in self._panels and self._panels[panel_id]['widget'] is not None:
            self._panels[panel_id]['widget'].layout.display = 'flex' if visible else 'none'
    
    def toggle_panel(self, panel_id: str) -> None:
        """Toggle the visibility of a panel.
        
        Args:
            panel_id: ID of the panel to toggle
        """
        if panel_id in self._panels and self._panels[panel_id]['widget'] is not None:
            current = self._panels[panel_id]['widget'].layout.display
            self.show_panel(panel_id, current == 'none')
    
    def update_info(self, title: str, content: Union[str, widgets.HTML], style: str = 'info') -> None:
        """Update the info panel with new content.
        
        Args:
            title: Title for the info panel
            content: Content to display (string or HTML widget)
            style: Style for the info panel ('info', 'success', 'warning', 'error')
        """
        # Ensure info_panel exists
        if self.info_panel is None:
            from smartcash.ui.components.info.info_component import InfoBox
            self.info_panel = InfoBox(
                title=title,
                content="",
                style=style
            )
        
        # Store the content for testing
        if isinstance(content, widgets.HTML):
            self._info_content = content.value if hasattr(content, 'value') else str(content)
            content_widget = content
        else:
            self._info_content = str(content)
            content_widget = widgets.HTML(value=str(content))
        
        # Update the info panel
        if hasattr(self.info_panel, 'update_content'):
            # For InfoBox component
            self.info_panel.update_content(self._info_content)
            if hasattr(self.info_panel, 'update_style'):
                self.info_panel.update_style(style)
        elif hasattr(self.info_panel, 'content'):
            # For widgets with content attribute
            self.info_panel.content = content_widget
            if hasattr(self.info_panel, 'style'):
                self.info_panel.style = style
        
        # Update title if supported
        if hasattr(self.info_panel, 'title'):
            self.info_panel.title = title
            
        # Ensure the panel is visible and has a layout
        if not hasattr(self.info_panel, 'layout') or self.info_panel.layout is None:
            self.info_panel.layout = widgets.Layout()
            
        # Ensure the panel is visible
        self.show_component('info', True)
        
        # For testing purposes, make sure we can access the raw content
        if not hasattr(self, '_info_content'):
            if isinstance(content, widgets.HTML):
                self._info_content = content.value if hasattr(content, 'value') else str(content)
            else:
                self._info_content = str(content)
                
    def show_component(self, component: str, show: bool = True) -> None:
        """Show or hide a specific component.
        
        Args:
            component: Component to show/hide ('info', 'tips')
            show: Whether to show (True) or hide (False) the component
        """
        if component == 'info':
            self.show_info = show
            if self.info_panel is not None:
                self.info_panel.container.layout.display = 'block' if show else 'none'
        elif component == 'tips':
            self.show_tips = show
            if self.tips_panel is not None:
                self.tips_panel.container.layout.display = 'block' if show else 'none'
    
    def toggle_tips(self) -> None:
        """Toggle the visibility of the tips panel."""
        self._tips_visible = not self._tips_visible
        self.show_component('tips', self._tips_visible)
        
        # Toggle the tips panel if it exists
        if self.tips_panel is not None and self.show_tips:
            self.tips_panel.toggle()
    
    def refresh(self) -> None:
        """Refresh the footer container."""
        self._create_container()
        
        # Refresh info panel if it exists
        if self.info_panel is not None:
            self.info_panel.refresh()
            
        # Refresh tips panel if it exists
        if self.tips_panel is not None:
            self.tips_panel.refresh()
    
    @property
    def info_content(self) -> str:
        """Get the current info content as a string.
        
        Returns:
            str: The raw string content of the info panel
        """
        # Return the stored raw content if available
        if hasattr(self, '_info_content') and self._info_content is not None:
            return self._info_content
            
        # Fallback to extracting from the info panel
        if self.info_panel is not None:
            # For InfoBox with content attribute
            if hasattr(self.info_panel, 'content'):
                content = self.info_panel.content
                if hasattr(content, 'value'):
                    return content.value
                return str(content) if content is not None else ""
            # For widgets with value attribute
            if hasattr(self.info_panel, 'value'):
                value = self.info_panel.value
                return value if isinstance(value, str) else str(value)
                
        return ""

def create_footer_container(
    panels: Optional[List[PanelConfig]] = None,
    style: Optional[Dict] = None,
    **layout_kwargs
) -> FooterContainer:
    """Create a footer container with the specified panels.
    
    Args:
        panels: List of panel configurations
        style: CSS styles for the container
        **layout_kwargs: Additional layout configuration
            (e.g., flex_flow, align_items, justify_content, etc.)
            
    Returns:
        FooterContainer: The created footer container
        
    Example:
        # Create a footer with two panels side by side
        footer = create_footer_container(
            panels=[
                PanelConfig(
                    panel_type=PanelType.INFO_BOX,
                    title="Info",
                    content="This is an info box",
                    flex="1",
                    min_width="300px"
                ),
                PanelConfig(
                    panel_type=PanelType.INFO_ACCORDION,
                    title="Details",
                    content="More details here...",
                    flex="2",
                    min_width="400px"
                )
            ],
            style={"border_top": "2px solid #007bff"},
            flex_flow="row wrap",
            justify_content="space-between"
        )
    """
    # Create layout config from kwargs
    layout = {
        'display': 'flex',
        'flex_flow': 'row wrap',
        'align_items': 'stretch',
        'justify_content': 'space-between',
        'width': '100%',
        'border': '1px solid #e0e0e0',
        'margin': '10px 0 0 0',
        'padding': '10px',
        'background': '#f9f9f9'
    }
    
    # Update with any layout kwargs
    layout.update(layout_kwargs)
    
    # Create the footer
    return FooterContainer(panels=panels, style=style, layout=layout)
