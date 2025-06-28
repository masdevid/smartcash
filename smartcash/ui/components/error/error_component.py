"""
File: smartcash/ui/components/error/error_component.py
Deskripsi: Simplified error component dengan traceback toggle yang reliable
"""

import ipywidgets as widgets
from typing import Optional, Dict, Any
from IPython.display import display, HTML

class ErrorComponent:
    """🚨 Clean error component dengan reliable traceback toggle"""
    
    def __init__(self, title: str = "🚨 Error", width: str = '100%'):
        self.title = title
        self.width = width
        self._components = {}
        
    def create(
        self,
        error_message: str,
        traceback: Optional[str] = None,
        error_type: str = "error",
        show_traceback: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create error widget dengan simple traceback toggle
        
        Args:
            error_message: Pesan error utama
            traceback: Traceback optional
            error_type: Tipe error (error, warning, info, success)
            show_traceback: Tampilkan toggle traceback
            
        Returns:
            Dictionary berisi widget dan komponen lainnya
        """
        # Color schemes
        styles = self._get_styles()
        style = styles.get(error_type.lower(), styles["error"])
        
        # Create main error display
        error_display = self._create_main_error_display(error_message, style, traceback is not None and show_traceback)
        
        # Create a container with proper width constraints
        container_style = """
            width: 100% !important;
            max-width: 100% !important;
            margin: 0 !important;
            padding: 0 !important;
            box-sizing: border-box !important;
            overflow: hidden !important;
        """
        
        content_style = """
            width: 100% !important;
            max-width: 100% !important;
            margin: 0 auto !important;
            padding: 0 8px !important;
            box-sizing: border-box !important;
            overflow: hidden !important;
        """
        
        # Create traceback section if needed
        if traceback and show_traceback:
            traceback_widget = self._create_traceback_section(traceback)
            
            # Create a VBox for the error and traceback
            content = widgets.VBox(
                [error_display, traceback_widget],
                layout=widgets.Layout(
                    width='100%',
                    margin='0',
                    padding='0',
                    _css={
                        'max-width': '1200px',
                        'width': '100%',
                        'margin': '10px 0'
                    }
                )
            )
            
            # Create the outer container
            container = widgets.VBox(
                layout=widgets.Layout(
                    width='100%',
                    margin='0',
                    padding='0',
                    _css={
                        'display': 'flex',
                        'justify-content': 'center',
                        'box-sizing': 'border-box',
                        'max-width': '100%',
                        'overflow': 'hidden',
                        'padding': '0 8px'
                    }
                )
            )
            
            # Add the content to the container
            container.children = [content]
            
        else:
            # For single error display without traceback
            error_display.layout = widgets.Layout(
                width='100%',
                margin='10px 0',
                _css={
                    'max-width': '100%',
                    'width': '100%',
                    'box-sizing': 'border-box',
                    'overflow': 'hidden',
                    'padding': '0 8px'
                }
            )
            
            container = widgets.VBox(
                layout=widgets.Layout(
                    width='100%',
                    margin='0',
                    padding='0',
                    _css={
                        'display': 'flex',
                        'justify-content': 'center',
                        'box-sizing': 'border-box',
                        'max-width': '100%',
                        'overflow': 'hidden',
                        'padding': '0 8px'
                    }
                )
            )
            
            # Add the error display to the container
            container.children = [error_display]
        
        # Store components
        self._components = {
            'widget': container,
            'container': container,
            'error_widget': error_display,
            'message_widget': error_display,
            'traceback_widget': getattr(self, '_traceback_widget', None)
        }
        
        return self._components
    
    def _get_styles(self) -> Dict[str, Dict[str, str]]:
        """Glassmorphism color schemes"""
        return {
            "error": {
                "bg": "linear-gradient(135deg, rgba(244, 67, 54, 0.1) 0%, rgba(244, 67, 54, 0.05) 100%)",
                "border": "rgba(244, 67, 54, 0.3)",
                "color": "#d32f2f",
                "icon": "🚨"
            },
            "warning": {
                "bg": "linear-gradient(135deg, rgba(255, 193, 7, 0.1) 0%, rgba(255, 193, 7, 0.05) 100%)",
                "border": "rgba(255, 193, 7, 0.3)",
                "color": "#ff8f00",
                "icon": "⚠️"
            },
            "info": {
                "bg": "linear-gradient(135deg, rgba(33, 150, 243, 0.1) 0%, rgba(33, 150, 243, 0.05) 100%)",
                "border": "rgba(33, 150, 243, 0.3)",
                "color": "#1565c0",
                "icon": "ℹ️"
            },
            "success": {
                "bg": "linear-gradient(135deg, rgba(76, 175, 80, 0.1) 0%, rgba(76, 175, 80, 0.05) 100%)",
                "border": "rgba(76, 175, 80, 0.3)",
                "color": "#2e7d32",
                "icon": "✅"
            }
        }
    
    def _create_main_error_display(self, message: str, style: Dict[str, str], has_traceback: bool) -> widgets.HTML:
        """Create main error display dengan toggle button"""
        error_id = f"error-{id(self)}"
        
        toggle_button = f"""
        <button onclick="
            const traceback = document.getElementById('traceback-{error_id}');
            const button = this;
            if (traceback) {{
                const isHidden = traceback.style.display === 'none';
                traceback.style.display = isHidden ? 'block' : 'none';
                button.innerHTML = isHidden ? '▲ Sembunyikan' : '📋 Detail';
            }}
            "
            style="
                background: {style['color']};
                color: white;
                border: none;
                border-radius: 4px;
                padding: 4px 8px;
                font-size: 11px;
                cursor: pointer;
                margin-left: 8px;
                white-space: nowrap;
            ">
            📋 Detail
        </button>
        """ if has_traceback else ""
        
        html_content = f"""
        <div style="
            width: 100% !important;
            max-width: 100% !important;
            box-sizing: border-box !important;
            padding: 0 !important;
            margin: 8px 0 !important;
            overflow: hidden !important;
            word-wrap: break-word !important;
            word-break: break-word !important;
        ">
            <div style="
                background: {style['bg']};
                backdrop-filter: blur(10px);
                -webkit-backdrop-filter: blur(10px);
                border: 1px solid {style['border']};
                border-radius: 12px;
                padding: 16px;
                margin: 0;
                box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
                width: 100% !important;
                max-width: 100% !important;
                box-sizing: border-box !important;
                overflow: hidden !important;
            ">
                <div style="
                    display: flex; 
                    align-items: flex-start;
                    margin-bottom: 12px;
                    flex-wrap: wrap;
                    gap: 8px;
                    width: 100% !important;
                    max-width: 100% !important;
                    box-sizing: border-box !important;
                    overflow: hidden !important;
                ">
                    <span style="font-size: 18px; flex-shrink: 0; margin-top: 2px;">{style['icon']}</span>
                    <h3 style="
                        color: {style['color']}; 
                        margin: 0; 
                        font-size: 16px;
                        font-weight: 600;
                        flex: 1;
                        min-width: 0;
                        word-break: break-word;
                    ">{self.title}</h3>
                    {toggle_button}
                </div>
                
                <div style="
                    color: {style['color']}; 
                    font-size: 14px; 
                    line-height: 1.5;
                    margin-bottom: 12px;
                    word-wrap: break-word !important;
                    overflow-wrap: break-word !important;
                    width: 100% !important;
                    max-width: 100% !important;
                    box-sizing: border-box !important;
                    overflow: hidden !important;
                    white-space: normal !important;
                ">{message}</div>
                
                <div style="
                    color: #666; 
                    font-size: 12px; 
                    font-style: italic;
                    border-top: 1px solid {style['border']};
                    padding-top: 8px;
                    margin-top: 8px;
                    word-wrap: break-word !important;
                    overflow-wrap: break-word !important;
                    width: 100% !important;
                    max-width: 100% !important;
                    box-sizing: border-box !important;
                    overflow: hidden !important;
                    white-space: normal !important;
                ">
                    💡 Cobalah refresh cell atau periksa dependencies
                </div>
            </div>
        </div>
        """
        
        return widgets.HTML(
            value=html_content,
            layout=widgets.Layout(
                width='100%',
                overflow='hidden'
            )
        )
    
    def _create_traceback_section(self, traceback: str) -> widgets.HTML:
        """Create simple traceback section"""
        error_id = f"error-{id(self)}"
        
        traceback_widget = widgets.HTML(
            value=f"""
            <div style="width: 100%; max-width: 100%; box-sizing: border-box; padding: 0 8px;">
                <div id="traceback-{error_id}" style="
                    display: none;
                    background: linear-gradient(135deg, rgba(248, 249, 250, 0.9) 0%, rgba(248, 249, 250, 0.8) 100%);
                    backdrop-filter: blur(10px);
                    -webkit-backdrop-filter: blur(10px);
                    border: 1px solid rgba(0, 0, 0, 0.1);
                    border-radius: 8px;
                    margin: 4px 0 16px 0;
                    overflow: hidden;
                    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
                    width: 100%;
                    max-width: 100%;
                    box-sizing: border-box;
                ">
                    <div style="
                        padding: 8px 12px;
                        background: #e9ecef;
                        border-bottom: 1px solid #dee2e6;
                        font-weight: 600;
                        font-size: 13px;
                        color: #495057;
                    ">
                        🔍 Stack Trace Details
                    </div>
                    <pre style="
                        margin: 0;
                        padding: 12px;
                        color: #212529;
                        background: transparent;
                        font-family: 'Courier New', monospace;
                        font-size: 11px;
                        line-height: 1.4;
                        white-space: pre-wrap;
                        word-wrap: break-word;
                        overflow-wrap: break-word;
                        max-height: 300px;
                        overflow-y: auto;
                        box-sizing: border-box;
                        width: 100%;
                        max-width: 100%;
                    ">{traceback}</pre>
                </div>
            </div>
            """,
            layout=widgets.Layout(
                width='100%',
                overflow='hidden'
            )
        )
        
        self._traceback_widget = traceback_widget
        return traceback_widget

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
    Factory function untuk membuat error component
    
    Args:
        error_message: Pesan error utama
        traceback: Stack trace optional  
        title: Judul error component
        error_type: Tipe error (error, warning, info, success)
        show_traceback: Tampilkan toggle traceback
        width: Lebar component
        
    Returns:
        Dictionary berisi widget dan komponen lainnya
    """
    error_component = ErrorComponent(title=title, width=width)
    return error_component.create(
        error_message=error_message,
        traceback=traceback,
        error_type=error_type,
        show_traceback=show_traceback,
        **kwargs
    )