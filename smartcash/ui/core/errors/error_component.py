"""
File: smartcash/ui/core/errors/error_component.py

Error component for displaying error messages with optional traceback in Jupyter notebooks.

This module provides a clean, interactive error display component that can show/hide
detailed traceback information and supports different error types (error, warning, info, success).
"""

from typing import Optional, Dict, Any, List, Union

# Conditional imports to handle environments without IPython/Jupyter
try:
    import ipywidgets as widgets
    from IPython.display import display, HTML
    JUPYTER_AVAILABLE = True
except ImportError:
    # Fallback for environments without IPython/Jupyter
    JUPYTER_AVAILABLE = False
    
    # Mock widgets for non-Jupyter environments
    class MockWidget:
        def __init__(self, *args, **kwargs):
            self.value = kwargs.get('value', '')
            self.description = kwargs.get('description', '')
        def __getattr__(self, name):
            return MockWidget()
    
    class widgets:
        VBox = MockWidget
        HBox = MockWidget
        HTML = MockWidget
        Button = MockWidget
        Output = MockWidget
        Layout = MockWidget
        Widget = MockWidget
        
    def display(obj):
        print(obj)
    
    class HTML:
        def __init__(self, html_str):
            self.data = html_str

class ErrorComponent:
    """🚨 Clean error component with reliable traceback toggle"""
    
    # Common layout styles
    CONTENT_LAYOUT = {
        'width': '100%',
        'max_width': '100%',
        'margin': '10px 0',
        'padding': '0',
        '_css': {
            'max-width': '100%',
            'width': '100% !important',
            'box-sizing': 'border-box',
            'overflow': 'hidden',
            'padding': '0 8px',
            'position': 'relative'
        }
    }
    
    CONTAINER_LAYOUT = {
        'width': '100%',
        'max_width': '100%',
        'margin': '0',
        'padding': '0',
        '_css': {
            'display': 'flex',
            'justify-content': 'center',
            'box-sizing': 'border-box',
            'max-width': '100% !important',
            'width': '100% !important',
            'overflow': 'hidden',
            'padding': '0',
            'position': 'relative'
        }
    }
    
    def __init__(self, title: str = "🚨 Error", width: str = '100%'):
        self.title = title
        self.width = width
        self._components = {}
    
    def _create_layout(self, layout_type: str = 'container') -> widgets.Layout:
        """Create a layout based on type (container or content)"""
        layout_data = self.CONTAINER_LAYOUT if layout_type == 'container' else self.CONTENT_LAYOUT
        return widgets.Layout(**layout_data)
    
    def _create_container(self, children: List[widgets.Widget], **kwargs) -> widgets.VBox:
        """Create a container with consistent styling"""
        layout = self._create_layout('container')
        container = widgets.VBox(children=children, layout=layout, **kwargs)
        return container
    
    def _create_content(self, children: List[widgets.Widget], **kwargs) -> widgets.VBox:
        """Create content area with consistent styling"""
        layout = self._create_layout('content')
        return widgets.VBox(children=children, layout=layout, **kwargs)
        
    def create(
        self,
        error_message: str,
        traceback: Optional[str] = None,
        error_type: str = "error",
        show_traceback: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create error widget with simple traceback toggle
        
        Args:
            error_message: Main error message
            traceback: Optional traceback text
            error_type: Type of error (error, warning, info, success)
            show_traceback: Whether to show traceback toggle
            
        Returns:
            Dictionary containing the widget and its components
        """
        style = self._get_styles().get(error_type.lower(), self._get_styles()["error"])
        has_traceback = bool(traceback and show_traceback)
        
        # Create main error display
        error_display = self._create_main_error_display(
            error_message, 
            style, 
            has_traceback
        )
        
        # Create container with appropriate children
        children = [error_display]
        if has_traceback:
            traceback_widget = self._create_traceback_section(traceback)
            children.append(traceback_widget)
        
        # Create content and container
        content = self._create_content(children)
        container = self._create_container([content])
        
        # Set up components dictionary
        self._components = {
            'widget': container,
            'container': container,
            'error_widget': error_display,
            'message_widget': error_display,
            'traceback_widget': traceback_widget if has_traceback else None
        }
        
        return self._components
        
    @staticmethod
    def _get_styles() -> Dict[str, Dict[str, str]]:
        """Get style definitions for different error types"""
        return {
            "error": {
                "bg": "linear-gradient(135deg, rgba(244, 67, 54, 0.12) 0%, rgba(244, 67, 54, 0.08) 100%)",
                "border": "rgba(244, 67, 54, 0.5)",
                "color": "#b71c1c",
                "icon": "❌",
                "hover_bg": "rgba(244, 67, 54, 0.15)",
                "dark_mode": {
                    "bg": "rgba(244, 67, 54, 0.2)",
                    "color": "#ff8a80"
                },
                "aria_label": "Error"
            },
            "warning": {
                "bg": "linear-gradient(135deg, rgba(255, 167, 38, 0.15) 0%, rgba(255, 167, 38, 0.08) 100%)",
                "border": "rgba(255, 167, 38, 0.5)",
                "color": "#e65100",
                "icon": "⚠️",
                "hover_bg": "rgba(255, 167, 38, 0.18)",
                "dark_mode": {
                    "bg": "rgba(255, 167, 38, 0.25)",
                    "color": "#ffd180"
                },
                "aria_label": "Warning"
            },
            "info": {
                "bg": "linear-gradient(135deg, rgba(33, 150, 243, 0.12) 0%, rgba(33, 150, 243, 0.06) 100%)",
                "border": "rgba(33, 150, 243, 0.5)",
                "color": "#0d47a1",
                "icon": "ℹ️",
                "hover_bg": "rgba(33, 150, 243, 0.15)",
                "dark_mode": {
                    "bg": "rgba(33, 150, 243, 0.2)",
                    "color": "#82b1ff"
                },
                "aria_label": "Information"
            },
            "success": {
                "bg": "linear-gradient(135deg, rgba(76, 175, 80, 0.12) 0%, rgba(76, 175, 80, 0.06) 100%)",
                "border": "rgba(76, 175, 80, 0.5)",
                "color": "#1b5e20",
                "icon": "✅",
                "hover_bg": "rgba(76, 175, 80, 0.15)",
                "dark_mode": {
                    "bg": "rgba(76, 175, 80, 0.2)",
                    "color": "#b9f6ca"
                },
                "aria_label": "Success"
            }
        }
    
    def _create_main_error_display(self, message: str, style: Dict[str, Any], has_traceback: bool) -> widgets.HTML:
        """Create main error display with toggle button and enhanced styling"""
        error_id = f"error-{id(self)}"
        
        # Create toggle button if traceback is available
        toggle_button = self._create_toggle_button(error_id, style) if has_traceback else ""
        
        # Add dark mode media query
        dark_mode_style = ""
        if 'dark_mode' in style:
            dark_mode_style = f"""
            @media (prefers-color-scheme: dark) {{
                #{error_id} {{
                    background: {style['dark_mode']['bg']} !important;
                    border-color: {style['border']} !important;
                }}
                #{error_id} .error-message {{
                    color: {style['dark_mode']['color']} !important;
                }}
                #{error_id} .error-title {{
                    color: {style['dark_mode']['color']} !important;
                }}
            }}
            """
        
        # Create HTML content for the error display with enhanced styling and accessibility
        html_content = f"""
        <style>
            {dark_mode_style}
            
            #{error_id} {{
                width: 100% !important;
                max-width: 100% !important;
                background: {style['bg']};
                border-left: 4px solid {style['border']};
                padding: 14px 18px;
                border-radius: 6px;
                margin: 10px 0;
                box-sizing: border-box;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
                box-shadow: 0 2px 12px rgba(0,0,0,0.08);
                position: relative;
                overflow: hidden;
                transition: all 0.2s ease-in-out;
            }}
            
            #{error_id}:hover {{
                box-shadow: 0 4px 16px rgba(0,0,0,0.12);
                transform: translateY(-1px);
            }}
            
            #{error_id} .error-header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin: 0 0 10px 0;
                padding: 0;
            }}
            
            #{error_id} .error-title-container {{
                display: flex;
                align-items: center;
                gap: 10px;
            }}
            
            #{error_id} .error-icon {{
                font-size: 18px;
                line-height: 1;
            }}
            
            #{error_id} .error-title {{
                font-weight: 600;
                color: {style['color']};
                font-size: 15px;
                margin: 0;
                padding: 0;
                line-height: 1.4;
            }}
            
            #{error_id} .error-message {{
                margin: 10px 0 0 0;
                padding: 0;
                color: #2c3e50;
                font-size: 14px;
                line-height: 1.6;
                word-wrap: break-word;
                white-space: pre-wrap;
            }}
            
            @media (max-width: 600px) {{
                #{error_id} {{
                    padding: 12px 14px;
                    border-radius: 4px;
                }}
                
                #{error_id} .error-title {{
                    font-size: 14px;
                }}
                
                #{error_id} .error-message {{
                    font-size: 13px;
                }}
            }}
        </style>
        
        <div id="{error_id}" role="alert" aria-label="{style.get('aria_label', 'Notification')}">
            <div class="error-header">
                <div class="error-title-container">
                    <span class="error-icon" aria-hidden="true">{style['icon']}</span>
                    <h3 class="error-title">{self.title}</h3>
                </div>
                {toggle_button}
            </div>
            <div class="error-message">
                {message}
            </div>
        </div>
        """
        
        return widgets.HTML(html_content)
    
    def _create_message_content(self, message: str) -> str:
        """Create the HTML content for the error message."""
        return f"""
        <div style="
            margin: 8px 0 0 0;
            padding: 0;
            color: #333;
            font-size: 14px;
            line-height: 1.5;
            word-wrap: break-word;
            white-space: pre-wrap;
        ">
            {message}
        </div>
        """
    
    def _create_header(self, style: Dict[str, str], toggle_button: str) -> str:
        """Create the header section of the error message."""
        return f"""
        <div style="
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin: 0 0 8px 0;
            padding: 0;
        ">
            <div style="
                font-weight: 600;
                color: {style['color']};
                font-size: 15px;
                display: flex;
                align-items: center;
                gap: 8px;
            ">
                <span>{style['icon']}</span>
                <span>{self.title}</span>
            </div>
            {toggle_button}
        </div>
        """
    
    def _create_toggle_button(self, error_id: str, style: Dict[str, Any]) -> str:
        """Create an accessible toggle button for showing/hiding traceback."""
        return f"""
        <button 
            id="toggle-{error_id}" 
            class="error-toggle-button"
            aria-expanded="false"
            aria-controls="traceback-{error_id}"
            style="
                background: transparent;
                border: 1px solid {style['color']};
                color: {style['color']};
                border-radius: 4px;
                padding: 4px 12px;
                font-size: 12px;
                font-weight: 500;
                cursor: pointer;
                transition: all 0.2s ease;
                outline: none;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
                line-height: 1.5;
                margin: 0;
                display: inline-flex;
                align-items: center;
                gap: 6px;
            "
            onmouseover="
                this.style.background = '{style.get('hover_bg', 'rgba(0,0,0,0.05)')}';
            "
            onmouseout="
                this.style.background = 'transparent';
            "
            onfocus="
                this.style.outline = '2px solid {style['color']}40';
                this.style.outlineOffset = '2px';
            "
            onblur="
                this.style.outline = 'none';
            "
            onclick="
                var traceback = document.getElementById('traceback-{error_id}');
                var button = document.getElementById('toggle-{error_id}');
                var isExpanded = button.getAttribute('aria-expanded') === 'true';
                
                if (isExpanded) {{
                    traceback.style.maxHeight = '0';
                    traceback.style.opacity = '0';
                    button.setAttribute('aria-expanded', 'false');
                    button.innerHTML = 'Show Details <span style=\'font-size:12px\'>▼</span>';
                }} else {{
                    traceback.style.maxHeight = traceback.scrollHeight + 'px';
                    traceback.style.opacity = '1';
                    button.setAttribute('aria-expanded', 'true');
                    button.innerHTML = 'Hide Details <span style=\'font-size:12px\'>▲</span>';
                }}
                
                // Dispatch event for any listeners
                var event = new CustomEvent('tracebackToggle', {{ 
                    detail: {{ 
                        id: '{error_id}',
                        visible: !isExpanded
                    }} 
                }});
                document.dispatchEvent(event);
            "
        >
            Show Details <span style="font-size:12px">▼</span>
        </button>
        
        <style>
            @media (prefers-color-scheme: dark) {{
                #{error_id} .error-toggle-button {{
                    border-color: {style['dark_mode']['color']} !important;
                    color: {style['dark_mode']['color']} !important;
                }}
                
                #{error_id} .error-toggle-button:hover {{
                    background: {style.get('dark_mode', {{}}).get('bg', 'rgba(255,255,255,0.1)')} !important;
                }}
            }}
            
            @media (max-width: 600px) {{
                #{error_id} .error-toggle-button {{
                    padding: 3px 10px;
                    font-size: 11px;
                }}
            }}
        </style>
        """
    
    def _create_traceback_section(self, traceback: str) -> widgets.HTML:
        """Create an accessible traceback section with smooth animations."""
        error_id = f"error-{id(self)}"
        
        return widgets.HTML(f"""
        <div 
            id="traceback-{error_id}" 
            class="error-traceback"
            role="region"
            aria-live="polite"
            style="
                margin-top: 10px;
                padding: 0;
                background: rgba(0, 0, 0, 0.02);
                border-radius: 4px;
                overflow: hidden;
                max-height: 0;
                opacity: 0;
                transition: max-height 0.3s ease-out, opacity 0.2s ease-in-out;
            "
        >
            <div style="
                padding: 12px;
                font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
                font-size: 12px;
                line-height: 1.5;
                color: #d32f2f;
                white-space: pre-wrap;
                word-break: break-word;
                overflow-x: auto;
                max-height: 300px;
                overflow-y: auto;
            ">
                {traceback}
            </div>
        </div>
        
        <style>
            @media (prefers-color-scheme: dark) {{
                #{error_id} .error-traceback {{
                    background: rgba(255, 255, 255, 0.05) !important;
                }}
                
                #{error_id} .error-traceback div {{
                    color: #ff8a80 !important;
                }}
            }}
            
            @media (max-width: 600px) {{
                #{error_id} .error-traceback div {{
                    font-size: 11px;
                    padding: 10px;
                }}
            }}
        </style>
        """)


def create_error_component(
    error_message: str,
    traceback: Optional[str] = None,
    title: str = "🚨 Error",
    error_type: str = "error",
    show_traceback: bool = True,
    width: str = '100%',
    **kwargs
) -> Dict[str, Any]:
    """
    Factory function to create an error component
    
    Args:
        error_message: Main error message
        traceback: Optional stack trace  
        title: Error title
        error_type: Type of error (error, warning, info, success)
        show_traceback: Whether to show traceback toggle
        width: Component width
        
    Returns:
        Dictionary containing the widget and its components
    """
    error_component = ErrorComponent(title=title, width=width)
    return error_component.create(
        error_message=error_message,
        traceback=traceback,
        error_type=error_type,
        show_traceback=show_traceback,
        **kwargs
    )
