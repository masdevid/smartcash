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
        
        # Create traceback section if needed
        if traceback and show_traceback:
            traceback_widget = self._create_traceback_section(traceback)
            container = widgets.VBox([error_display, traceback_widget], 
                layout=widgets.Layout(width=self.width, margin='10px 0'))
        else:
            container = error_display
        
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
        toggle_button = f"""
        <button onclick="toggleTraceback()" style="
            background: {style['color']};
            color: white;
            border: none;
            border-radius: 4px;
            padding: 4px 8px;
            font-size: 11px;
            cursor: pointer;
            margin-left: 8px;
        ">
            📋 Detail
        </button>
        """ if has_traceback else ""
        
        html_content = f"""
        <div style="
            background: {style['bg']};
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border: 1px solid {style['border']};
            border-radius: 12px;
            padding: 16px;
            margin: 10px 0;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
            max-width: 100%;
            box-sizing: border-box;
        ">
            <div style="
                display: flex; 
                align-items: center; 
                margin-bottom: 12px;
                flex-wrap: wrap;
                gap: 8px;
            ">
                <span style="font-size: 18px; flex-shrink: 0;">{style['icon']}</span>
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
                word-wrap: break-word;
                overflow-wrap: break-word;
            ">{message}</div>
            
            <div style="
                color: #666; 
                font-size: 12px; 
                font-style: italic;
                border-top: 1px solid {style['border']};
                padding-top: 8px;
                margin-top: 8px;
                word-wrap: break-word;
            ">
                💡 Cobalah refresh cell atau periksa dependencies
            </div>
        </div>
        
        <script>
        function toggleTraceback() {{
            const traceback = document.getElementById('traceback-section');
            if (traceback) {{
                traceback.style.display = traceback.style.display === 'none' ? 'block' : 'none';
            }}
        }}
        </script>
        """
        
        return widgets.HTML(value=html_content, layout=widgets.Layout(width=self.width))
    
    def _create_traceback_section(self, traceback: str) -> widgets.HTML:
        """Create simple traceback section"""
        traceback_widget = widgets.HTML(
            value=f"""
            <div id="traceback-section" style="
                display: none;
                background: linear-gradient(135deg, rgba(248, 249, 250, 0.9) 0%, rgba(248, 249, 250, 0.8) 100%);
                backdrop-filter: blur(10px);
                -webkit-backdrop-filter: blur(10px);
                border: 1px solid rgba(0, 0, 0, 0.1);
                border-radius: 8px;
                margin-top: 8px;
                overflow: hidden;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
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
                ">{traceback}</pre>
            </div>
            """,
            layout=widgets.Layout(width='100%')
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