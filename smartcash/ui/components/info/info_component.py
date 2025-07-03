"""
Info components for displaying information and notifications.

This module provides UI components for displaying information in various formats
like accordions, styled boxes, and tabs, all built on top of the BaseUIComponent.
"""

from typing import Dict, Optional, Union, List
import ipywidgets as widgets
from ipywidgets import Layout

from smartcash.ui.components.base_component import BaseUIComponent
from smartcash.ui.utils.constants import ALERT_STYLES


class InfoBox(BaseUIComponent):
    """A styled box for displaying information with different alert levels."""
    
    def __init__(
        self,
        content: str,
        style: str = "info",
        title: Optional[str] = None,
        padding: int = 10,
        border_radius: int = 5,
        component_name: str = "info_box"
    ):
        """Initialize the InfoBox component.
        
        Args:
            content: The content to display (HTML string)
            style: Style of the info box ('info', 'success', 'warning', 'error')
            title: Optional title for the info box
            padding: Padding in pixels
            border_radius: Border radius in pixels
            component_name: Unique name for this component instance
        """
        super().__init__(component_name)
        self.content = content
        self.style = style
        self.title = title
        self.padding = padding
        self.border_radius = border_radius
        self._initialized = False
        
    def _create_ui_components(self) -> None:
        """Create the UI components for the info box."""
        style_config = ALERT_STYLES.get(self.style, ALERT_STYLES['info'])
        
        # Create the styled content
        styled_content = self._get_styled_content()
        
        # Create the main container
        if self.title:
            title_style = f"""
                font-weight: bold;
                margin-bottom: 5px;
                color: {style_config['text_color']};
            """
            title_html = f'<div style="{title_style}">{self.title}</div>'
            content_html = f"{title_html}{styled_content}"
        else:
            content_html = styled_content
            
        self._ui_components['content'] = widgets.HTML(value=content_html)
        self._ui_components['container'] = self._ui_components['content']
        
    def _get_styled_content(self) -> str:
        """Get the styled content HTML."""
        style_config = ALERT_STYLES.get(self.style, ALERT_STYLES['info'])
        
        return f"""
        <div style="
            padding: {self.padding}px;
            background-color: {style_config['bg_color']};
            color: {style_config['text_color']};
            border-radius: {self.border_radius}px;
            border: 1px solid {style_config.get('border_color', style_config['bg_color'])};
        ">
            {self.content}
        </div>
        """
        
    def update_content(self, content: str) -> None:
        """Update the content of the info box.
        
        Args:
            content: New content to display
        """
        self.content = content
        if self._initialized:
            self._ui_components['content'].value = self._get_styled_content()
            
    def update_style(self, style: str) -> None:
        """Update the style of the info box.
        
        Args:
            style: New style to apply ('info', 'success', 'warning', 'error')
        """
        self.style = style
        if self._initialized:
            self._ui_components['content'].value = self._get_styled_content()


class InfoAccordion(InfoBox):
    """A collapsible accordion for displaying information."""
    
    def __init__(
        self,
        title: str,
        content: Union[str, widgets.Widget],
        style: str = "info",
        icon: Optional[str] = None,
        open_by_default: bool = False,
        component_name: str = "info_accordion"
    ):
        """Initialize the InfoAccordion component.
        
        Args:
            title: Title of the accordion
            content: Content to display (HTML string or widget)
            style: Style of the info box ('info', 'success', 'warning', 'error')
            icon: Optional icon to display next to the title
            open_by_default: Whether the accordion should be open by default
            component_name: Unique name for this component instance
        """
        super().__init__(
            content=content if isinstance(content, str) else "",
            style=style,
            title=title,
            component_name=component_name
        )
        self.accordion_title = title
        self.icon = icon
        self.open_by_default = open_by_default
        self.custom_widget = None if isinstance(content, str) else content
        
    def _create_ui_components(self) -> None:
        """Create the UI components for the accordion."""
        style_config = ALERT_STYLES.get(self.style, ALERT_STYLES['info'])
        
        # Set icon
        icon = self.icon if self.icon is not None else style_config['icon']
        title_with_icon = f"{icon} {self.accordion_title}"
        
        # Create content widget
        if self.custom_widget is not None:
            content_widget = self.custom_widget
        else:
            content_widget = widgets.HTML(value=self._get_styled_content())
        
        # Create accordion
        accordion = widgets.Accordion(children=[content_widget])
        accordion.set_title(0, title_with_icon)
        
        # Set initial state
        accordion.selected_index = 0 if self.open_by_default else None
        
        # Store components
        self._ui_components['content'] = content_widget
        self._ui_components['accordion'] = accordion
        self._ui_components['container'] = accordion
        
    def set_open(self, is_open: bool = True) -> None:
        """Set whether the accordion is open.
        
        Args:
            is_open: Whether the accordion should be open
        """
        if self._initialized:
            self._ui_components['accordion'].selected_index = 0 if is_open else None


class TabbedInfo(BaseUIComponent):
    """A component that displays information in tabs."""
    
    def __init__(
        self,
        tabs_content: Dict[str, Union[str, widgets.Widget]],
        style: str = "info",
        component_name: str = "tabbed_info"
    ):
        """Initialize the TabbedInfo component.
        
        Args:
            tabs_content: Dictionary mapping tab titles to content
            style: Style of the info boxes ('info', 'success', 'warning', 'error')
            component_name: Unique name for this component instance
        """
        super().__init__(component_name)
        self.tabs_content = tabs_content
        self.style = style
        
    def _create_ui_components(self) -> None:
        """Create the UI components for the tabbed info."""
        style_config = ALERT_STYLES.get(self.style, ALERT_STYLES['info'])
        
        # Create a widget for each tab
        tab_widgets = []
        tab_titles = []
        
        for title, content in self.tabs_content.items():
            if isinstance(content, str):
                styled_content = f"""
                <div style="
                    padding: 10px;
                    background-color: {style_config['bg_color']};
                    color: {style_config['text_color']};
                    border-radius: 5px;
                ">
                    {content}
                </div>
                """
                tab_widgets.append(widgets.HTML(value=styled_content))
            else:
                tab_widgets.append(content)
            tab_titles.append(title)
        
        # Create tabs
        tabs = widgets.Tab(children=tab_widgets)
        
        # Set tab titles
        for i, title in enumerate(tab_titles):
            tabs.set_title(i, title)
        
        # Store components
        self._ui_components['tabs'] = tabs
        self._ui_components['container'] = tabs
        
    def add_tab(self, title: str, content: Union[str, widgets.Widget]) -> None:
        """Add a new tab to the tabbed info.
        
        Args:
            title: Title of the new tab
            content: Content of the new tab (HTML string or widget)
        """
        self.tabs_content[title] = content
        if self._initialized:
            # Rebuild the tabs
            self._create_ui_components()


# Backward compatibility functions
def create_info_accordion(
    title: str,
    content: Union[str, widgets.Widget],
    style: str = "info",
    icon: Optional[str] = None,
    open_by_default: bool = False
) -> widgets.Accordion:
    """Legacy function for backward compatibility."""
    accordion = InfoAccordion(
        title=title,
        content=content,
        style=style,
        icon=icon,
        open_by_default=open_by_default
    )
    return accordion.show()


def style_info_content(
    content: str,
    style: str = "info",
    padding: int = 10,
    border_radius: int = 5
) -> str:
    """Legacy function for backward compatibility."""
    style_config = ALERT_STYLES.get(style, ALERT_STYLES['info'])
    return f"""
    <div style="
        padding: {padding}px;
        background-color: {style_config['bg_color']};
        color: {style_config['text_color']};
        border-radius: {border_radius}px;
    ">
        {content}
    </div>
    """


def create_tabbed_info(
    tabs_content: Dict[str, str],
    style: str = "info"
) -> widgets.Tab:
    """Legacy function for backward compatibility."""
    tabbed_info = TabbedInfo(tabs_content=tabs_content, style=style)
    return tabbed_info.show()
