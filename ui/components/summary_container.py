"""
File: smartcash/ui/components/summary_container.py
Description: Reusable summary container component with dynamic styling.

This module provides a blank, reusable summary container component with
dynamic background and border styles featuring a modern gradient look.
It can be used for displaying various types of content with consistent
styling across the application.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
import ipywidgets as widgets


class SummaryContainer:
    """A blank, reusable container with dynamic styling and modern gradient look."""
    
    # Predefined style themes with gradient backgrounds
    THEMES = {
        "default": {
            "gradient": "linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%)",
            "border": "1px solid #dee2e6",
            "text_color": "#212529"
        },
        "primary": {
            "gradient": "linear-gradient(135deg, #cfe2ff 0%, #9ec5fe 100%)",
            "border": "1px solid #9ec5fe",
            "text_color": "#084298"
        },
        "success": {
            "gradient": "linear-gradient(135deg, #d1e7dd 0%, #a3cfbb 100%)",
            "border": "1px solid #a3cfbb",
            "text_color": "#0f5132"
        },
        "warning": {
            "gradient": "linear-gradient(135deg, #fff3cd 0%, #ffe69c 100%)",
            "border": "1px solid #ffe69c",
            "text_color": "#664d03"
        },
        "danger": {
            "gradient": "linear-gradient(135deg, #f8d7da 0%, #f1aeb5 100%)",
            "border": "1px solid #f1aeb5",
            "text_color": "#842029"
        },
        "info": {
            "gradient": "linear-gradient(135deg, #cff4fc 0%, #9eeaf9 100%)",
            "border": "1px solid #9eeaf9",
            "text_color": "#055160"
        }
    }
    
    def __init__(self, theme: str = "default", title: str = "", icon: str = ""):
        """Initialize the summary container.
        
        Args:
            theme: Theme name from predefined themes (default, primary, success, warning, danger, info)
            title: Optional title for the summary container
            icon: Optional icon to display next to the title
        """
        # Apply theme styling
        theme_style = self.THEMES.get(theme, self.THEMES["default"])
        
        # Create content widgets
        widgets_list = []
        
        # Add title if provided
        if title:
            self.title = widgets.HTML(
                f"<h4 style='margin: 0 0 10px 0; font-size: 1.1rem; color: {theme_style['text_color']}'>" 
                f"{icon + ' ' if icon else ''}{title}</h4>"
            )
            widgets_list.append(self.title)
        
        # Create content area
        self.content = widgets.HTML(value="")
        widgets_list.append(self.content)
        
        # Create the container with dynamic styling
        self.container = widgets.Box(
            widgets_list,
            layout=widgets.Layout(
                width="100%",
                padding="15px",
                margin="0 0 15px 0",
                border=theme_style["border"],
                border_radius="8px",
                background=theme_style["gradient"],
                box_shadow="0 2px 5px rgba(0,0,0,0.05)"
            )
        )
        
        # Store theme for later use
        self.theme = theme
        self.theme_style = theme_style
        
        # Add CSS class for styling
        self.container.add_class("summary-container")
    
    def set_content(self, content: str) -> None:
        """Set the HTML content of the container.
        
        Args:
            content: HTML content to display in the container
        """
        self.content.value = content
    
    def set_theme(self, theme: str) -> None:
        """Change the container theme.
        
        Args:
            theme: Theme name from predefined themes (default, primary, success, warning, danger, info)
        """
        if theme not in self.THEMES:
            theme = "default"
            
        theme_style = self.THEMES[theme]
        self.theme = theme
        self.theme_style = theme_style
        
        # Update container styling
        self.container.layout.border = theme_style["border"]
        self.container.layout.background = theme_style["gradient"]
        
        # Update title if it exists
        if hasattr(self, 'title'):
            title_text = self.title.value
            if title_text:
                # Extract the title text from HTML
                import re
                title_match = re.search(r'>([^<]+)</h4>', title_text)
                if title_match:
                    title_content = title_match.group(1)
                    # Check if there's an icon
                    icon_match = re.search(r'>([^<]{1,2}\s+)', title_content)
                    icon = icon_match.group(1) if icon_match else ""
                    title_text = title_content[len(icon):] if icon else title_content
                    
                    # Update title with new styling
                    self.title.value = (
                        f"<h4 style='margin: 0 0 10px 0; font-size: 1.1rem; color: {theme_style['text_color']}'>" 
                        f"{icon}{title_text}</h4>"
                    )
    
    def set_html(self, html: str, theme: Optional[str] = None) -> None:
        """Set HTML content with optional theme change.
        
        Args:
            html: HTML content to display
            theme: Optional theme to apply
        """
        if theme:
            self.set_theme(theme)
        self.set_content(html)
    
    def show_message(self, title: str, message: str, message_type: str = "info", icon: Optional[str] = None) -> None:
        """Show a message with title and content.
        
        Args:
            title: Message title
            message: Message content
            message_type: Message type (info, success, warning, danger)
            icon: Optional icon to display
        """
        # Set theme based on message type
        self.set_theme(message_type)
        
        # Default icons based on message type
        if icon is None:
            icons = {
                "info": "ℹ️",
                "success": "✅",
                "warning": "⚠️",
                "danger": "❌",
                "primary": "🔷"
            }
            icon = icons.get(message_type, "")
        
        # Create HTML content
        html_content = (
            f"<div style='padding: 10px;'>"
            f"<h5 style='margin: 0; color: {self.theme_style['text_color']};'>{icon} {title}</h5>"
            f"<div style='margin-top: 8px; color: {self.theme_style['text_color']};'>{message}</div>"
            f"</div>"
        )
        
        self.set_content(html_content)
    
    def show_status(self, items: Dict[str, Any], title: str = "", icon: str = "") -> None:
        """Show a status summary with multiple items.
        
        Args:
            items: Dictionary of status items and their values
            title: Optional title for the status
            icon: Optional icon for the title
        """
        html_parts = []
        
        # Add title if provided
        if title:
            html_parts.append(f"<h5 style='margin: 0 0 10px 0; color: {self.theme_style['text_color']};'>{icon} {title}</h5>")
        
        # Add status items
        html_parts.append("<div style='display: flex; flex-direction: column; gap: 5px;'>")
        
        for label, value in items.items():
            html_parts.append(
                f"<div style='display: flex; justify-content: space-between;'>"
                f"<span style='color: {self.theme_style['text_color']};'>{label}</span>"
                f"<span style='font-weight: bold; color: {self.theme_style['text_color']};'>{value}</span>"
                f"</div>"
            )
        
        html_parts.append("</div>")
        
        self.set_content("".join(html_parts))
    
    def clear(self) -> None:
        """Clear the container content."""
        self.content.value = ""


def create_summary_container(theme: str = "default", title: str = "", icon: str = "") -> SummaryContainer:
    """Create a summary container with dynamic styling and modern gradient look.
    
    Args:
        theme: Theme name from predefined themes (default, primary, success, warning, danger, info)
        title: Optional title for the summary container
        icon: Optional icon to display next to the title
        
    Returns:
        SummaryContainer instance
    """
    return SummaryContainer(theme=theme, title=title, icon=icon)
