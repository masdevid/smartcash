"""
Error component with collapsible traceback support
"""
import ipywidgets as widgets
from typing import Optional, Dict, Any
from IPython.display import display, HTML

class ErrorComponent:
    """Reusable error component with collapsible traceback"""
    
    def __init__(self, title: str = "Error", width: str = '100%'):
        """Initialize error component"""
        self.title = title
        self.width = width
        self._components = {}
        
    def create(
        self,
        error_message: str,
        traceback: Optional[str] = None,
        error_type: str = "error",
        show_traceback: bool = True
    ) -> Dict[str, Any]:
        """
        Create error widget with optional collapsible traceback
        
        Args:
            error_message: Main error message to display
            traceback: Optional traceback text
            error_type: Type of error (error, warning, info)
            show_traceback: Whether to show traceback toggle
            
        Returns:
            Dictionary containing the widget and other components
        """
        # Define styles based on error type
        styles = {
            "error": {
                "bg": "#ffebee",
                "border": "#f44336",
                "color": "#d32f2f",
                "icon": "❌"
            },
            "warning": {
                "bg": "#fff8e1",
                "border": "#ffc107",
                "color": "#ff8f00",
                "icon": "⚠️"
            },
            "info": {
                "bg": "#e3f2fd",
                "border": "#2196f3",
                "color": "#1565c0",
                "icon": "ℹ️"
            }
        }
        
        style = styles.get(error_type.lower(), styles["error"])
        
        # Create main error widget
        error_widget = widgets.HTML(
            value=self._create_error_html(
                error_message, 
                style,
                show_traceback and traceback is not None
            ),
            layout=widgets.Layout(width=self.width)
        )
        
        # Create traceback widget if traceback is provided
        traceback_widget = None
        if traceback and show_traceback:
            traceback_widget = widgets.Textarea(
                value=traceback,
                layout=widgets.Layout(
                    width='100%',
                    height='200px',
                    display='none',  # Initially hidden
                    font_family='monospace',
                    margin_top='10px'
                ),
                disabled=True
            )
            
            # Toggle button for traceback
            toggle_button = widgets.Button(
                description='Show Traceback',
                button_style='',
                tooltip='Click to show/hide traceback',
                layout=widgets.Layout(width='auto')
            )
            
            def on_toggle_click(button):
                if traceback_widget.layout.display == 'none':
                    traceback_widget.layout.display = 'block'
                    button.description = 'Hide Traceback'
                else:
                    traceback_widget.layout.display = 'none'
                    button.description = 'Show Traceback'
            
            toggle_button.on_click(on_toggle_click)
            
            # Create container for traceback components
            traceback_container = widgets.VBox(
                [toggle_button, traceback_widget],
                layout=widgets.Layout(width='100%', margin_top='10px')
            )
            
            # Create final container
            container = widgets.VBox(
                [error_widget, traceback_container],
                layout=widgets.Layout(width='100%')
            )
        else:
            container = error_widget
        
        # Store components
        self._components = {
            'container': container,
            'error_widget': error_widget,
            'traceback_widget': traceback_widget,
            'toggle_button': toggle_button if traceback and show_traceback else None
        }
        
        return self._components
    
    def _create_error_html(self, message: str, style: Dict[str, str], has_traceback: bool) -> str:
        """Create HTML for error message"""
        return f"""
        <div style="
            background: {style['bg']};
            border: 1px solid {style['border']};
            border-radius: 4px; 
            padding: 15px; 
            margin: 10px 0;
        ">
            <h3 style="color: {style['color']}; margin-top: 0; margin-bottom: 10px;">
                {style['icon']} {self.title}
            </h3>
            <div style="margin-bottom: 10px;">{message}</div>
            <div style="color: #666; font-size: 0.9em; margin-top: 10px;">
                <em>Silakan coba refresh cell atau periksa dependencies.</em>
                {'<br>Klik "Show Traceback" di bawah untuk detail lebih lanjut.' if has_traceback else ''}
            </div>
        </div>
        """

def create_error_component(
    error_message: str,
    traceback: Optional[str] = None,
    title: str = "Error",
    error_type: str = "error",
    show_traceback: bool = True,
    width: str = '100%'
) -> Dict[str, Any]:
    """
    Helper function to create an error component
    
    Args:
        error_message: Main error message
        traceback: Optional traceback text
        title: Title of the error component
        error_type: Type of error (error, warning, info)
        show_traceback: Whether to show traceback toggle
        width: Width of the component
        
    Returns:
        Dictionary containing the widget and other components
    """
    error_component = ErrorComponent(title=title, width=width)
    return error_component.create(
        error_message=error_message,
        traceback=traceback,
        error_type=error_type,
        show_traceback=show_traceback
    )
