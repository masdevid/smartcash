"""
File: smartcash/ui/components/error/error_component.py
Deskripsi: Enhanced error component dengan modern glassmorphism design dan smooth animations
"""

import ipywidgets as widgets
from typing import Optional, Dict, Any
from IPython.display import display, HTML

class ErrorComponent:
    """✨ Modern error component dengan glassmorphism design dan smooth animations"""
    
    def __init__(self, title: str = "🚨 Error", width: str = '100%'):
        """Initialize error component dengan modern styling"""
        self.title = title
        self.width = width
        self._components = {}
        self._is_expanded = False
        
    def create(
        self,
        error_message: str,
        traceback: Optional[str] = None,
        error_type: str = "error",
        show_traceback: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        🎨 Create modern error widget dengan expandable design
        
        Args:
            error_message: Pesan error utama
            traceback: Traceback optional
            error_type: Tipe error (error, warning, info, success)
            show_traceback: Tampilkan toggle traceback
            
        Returns:
            Dictionary berisi widget dan komponen lainnya
        """
        # 🎨 Modern style definitions dengan glassmorphism
        styles = self._get_modern_styles()
        style = styles.get(error_type.lower(), styles["error"])
        
        # 📱 Create main error display
        error_display = self._create_main_error_display(error_message, style, traceback is not None and show_traceback)
        
        # 🔍 Create expandable traceback section
        if traceback and show_traceback:
            traceback_section = self._create_traceback_section(traceback, style)
            container = widgets.VBox([error_display, traceback_section], 
                layout=widgets.Layout(width=self.width, margin='10px 0'))
        else:
            container = error_display
        
        # 💾 Store components untuk reuse
        self._components = {
            'widget': container,
            'container': container,
            'error_widget': error_display,
            'message_widget': error_display,
            'traceback_widget': getattr(self, '_traceback_widget', None),
            'toggle_button': getattr(self, '_toggle_button', None)
        }
        
        return self._components
    
    def _get_modern_styles(self) -> Dict[str, Dict[str, str]]:
        """🎨 Modern glassmorphism color schemes"""
        return {
            "error": {
                "bg": "linear-gradient(135deg, rgba(244, 67, 54, 0.1) 0%, rgba(244, 67, 54, 0.05) 100%)",
                "border": "rgba(244, 67, 54, 0.3)",
                "color": "#d32f2f",
                "icon": "🚨",
                "shadow": "0 8px 32px rgba(244, 67, 54, 0.1)",
                "accent": "#f44336"
            },
            "warning": {
                "bg": "linear-gradient(135deg, rgba(255, 193, 7, 0.1) 0%, rgba(255, 193, 7, 0.05) 100%)",
                "border": "rgba(255, 193, 7, 0.3)",
                "color": "#ff8f00",
                "icon": "⚠️",
                "shadow": "0 8px 32px rgba(255, 193, 7, 0.1)",
                "accent": "#ffc107"
            },
            "info": {
                "bg": "linear-gradient(135deg, rgba(33, 150, 243, 0.1) 0%, rgba(33, 150, 243, 0.05) 100%)",
                "border": "rgba(33, 150, 243, 0.3)",
                "color": "#1565c0",
                "icon": "ℹ️",
                "shadow": "0 8px 32px rgba(33, 150, 243, 0.1)",
                "accent": "#2196f3"
            },
            "success": {
                "bg": "linear-gradient(135deg, rgba(76, 175, 80, 0.1) 0%, rgba(76, 175, 80, 0.05) 100%)",
                "border": "rgba(76, 175, 80, 0.3)",
                "color": "#2e7d32",
                "icon": "✅",
                "shadow": "0 8px 32px rgba(76, 175, 80, 0.1)",
                "accent": "#4caf50"
            }
        }
    
    def _create_main_error_display(self, message: str, style: Dict[str, str], has_traceback: bool) -> widgets.HTML:
        """📱 Create main error display dengan modern design"""
        expand_hint = "🔽 Klik untuk detail" if has_traceback else ""
        
        html_content = f"""
        <div id="error-container" style="
            background: {style['bg']};
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border: 1px solid {style['border']};
            border-radius: 16px;
            padding: 20px;
            margin: 10px 0;
            box-shadow: {style['shadow']};
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            cursor: {'pointer' if has_traceback else 'default'};
            position: relative;
            overflow: hidden;
        " {'onclick="toggleTraceback()"' if has_traceback else ''}>
            
            {self._create_background_pattern(style['accent'])}
            
            <div style="position: relative; z-index: 2;">
                <div style="display: flex; align-items: center; margin-bottom: 12px;">
                    <span style="font-size: 24px; margin-right: 12px;">{style['icon']}</span>
                    <h3 style="
                        color: {style['color']}; 
                        margin: 0; 
                        font-size: 18px;
                        font-weight: 600;
                        flex: 1;
                    ">{self.title}</h3>
                    {f'<span style="color: {style["color"]}; font-size: 12px; opacity: 0.7;">{expand_hint}</span>' if has_traceback else ''}
                </div>
                
                <div style="
                    color: {style['color']}; 
                    font-size: 14px; 
                    line-height: 1.6;
                    margin-bottom: 12px;
                    font-weight: 500;
                ">{message}</div>
                
                <div style="
                    color: rgba(102, 102, 102, 0.8); 
                    font-size: 12px; 
                    font-style: italic;
                    border-top: 1px solid rgba(255, 255, 255, 0.1);
                    padding-top: 12px;
                    margin-top: 12px;
                ">
                    💡 Cobalah refresh cell atau periksa dependencies
                    {('<br>🔍 ' + expand_hint) if has_traceback else ''}
                </div>
            </div>
        </div>
        
        {self._get_toggle_script() if has_traceback else ''}
        """
        
        return widgets.HTML(value=html_content, layout=widgets.Layout(width=self.width))
    
    def _create_traceback_section(self, traceback: str, style: Dict[str, str]) -> widgets.VBox:
        """🔍 Create expandable traceback section dengan modern styling"""
        # Modern traceback container
        traceback_container = widgets.HTML(
            value=f"""
            <div id="traceback-section" style="
                display: none;
                background: linear-gradient(135deg, rgba(20, 20, 20, 0.95) 0%, rgba(40, 40, 40, 0.9) 100%);
                backdrop-filter: blur(20px);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 12px;
                margin-top: 10px;
                padding: 0;
                overflow: hidden;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
                transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
                max-height: 400px;
            ">
                <div style="
                    padding: 16px 20px;
                    background: rgba(255, 255, 255, 0.05);
                    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                ">
                    <span style="color: #ffffff; font-weight: 600; font-size: 14px;">
                        🔍 Stack Trace Details
                    </span>
                    <button onclick="copyTraceback()" style="
                        background: rgba(255, 255, 255, 0.1);
                        border: 1px solid rgba(255, 255, 255, 0.2);
                        border-radius: 6px;
                        color: #ffffff;
                        padding: 4px 12px;
                        font-size: 12px;
                        cursor: pointer;
                        transition: all 0.2s ease;
                    " onmouseover="this.style.background='rgba(255, 255, 255, 0.2)'" 
                       onmouseout="this.style.background='rgba(255, 255, 255, 0.1)'">
                        📋 Copy
                    </button>
                </div>
                <pre id="traceback-content" style="
                    margin: 0;
                    padding: 20px;
                    color: #e0e0e0;
                    background: transparent;
                    font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
                    font-size: 12px;
                    line-height: 1.5;
                    overflow-x: auto;
                    white-space: pre-wrap;
                    word-wrap: break-word;
                    max-height: 300px;
                    overflow-y: auto;
                ">{traceback}</pre>
            </div>
            
            <script>
            function copyTraceback() {{
                const content = document.getElementById('traceback-content').textContent;
                navigator.clipboard.writeText(content).then(() => {{
                    const button = event.target;
                    const originalText = button.textContent;
                    button.textContent = '✅ Copied!';
                    button.style.background = 'rgba(76, 175, 80, 0.3)';
                    setTimeout(() => {{
                        button.textContent = originalText;
                        button.style.background = 'rgba(255, 255, 255, 0.1)';
                    }}, 2000);
                }});
            }}
            </script>
            """,
            layout=widgets.Layout(width='100%')
        )
        
        # Store untuk reference
        self._traceback_widget = traceback_container
        
        return widgets.VBox([traceback_container], layout=widgets.Layout(width='100%'))
    
    def _create_background_pattern(self, accent_color: str) -> str:
        """🎨 Create subtle background pattern"""
        return f"""
        <div style="
            position: absolute;
            top: -50%;
            right: -50%;
            width: 100%;
            height: 100%;
            background: radial-gradient(circle, {accent_color}10 0%, transparent 70%);
            opacity: 0.3;
            z-index: 1;
        "></div>
        """
    
    def _get_toggle_script(self) -> str:
        """⚡ JavaScript untuk smooth toggle animation"""
        return """
        <script>
        let tracebackVisible = false;
        
        function toggleTraceback() {
            const container = document.getElementById('error-container');
            const traceback = document.getElementById('traceback-section');
            
            if (!traceback) return;
            
            tracebackVisible = !tracebackVisible;
            
            if (tracebackVisible) {
                // Show dengan smooth animation
                traceback.style.display = 'block';
                traceback.style.opacity = '0';
                traceback.style.transform = 'translateY(-20px)';
                
                // Trigger animation
                setTimeout(() => {
                    traceback.style.transition = 'all 0.4s cubic-bezier(0.4, 0, 0.2, 1)';
                    traceback.style.opacity = '1';
                    traceback.style.transform = 'translateY(0)';
                }, 10);
                
                // Update container hover effect
                container.style.transform = 'translateY(-2px)';
                container.style.boxShadow = '0 12px 48px rgba(0, 0, 0, 0.15)';
                
            } else {
                // Hide dengan smooth animation
                traceback.style.transition = 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)';
                traceback.style.opacity = '0';
                traceback.style.transform = 'translateY(-20px)';
                
                setTimeout(() => {
                    traceback.style.display = 'none';
                }, 300);
                
                // Reset container
                container.style.transform = 'translateY(0)';
                container.style.boxShadow = '0 8px 32px rgba(0, 0, 0, 0.1)';
            }
        }
        
        // Add hover effects
        document.addEventListener('DOMContentLoaded', function() {
            const container = document.getElementById('error-container');
            if (container && container.style.cursor === 'pointer') {
                container.addEventListener('mouseenter', () => {
                    if (!tracebackVisible) {
                        container.style.transform = 'translateY(-2px)';
                        container.style.boxShadow = '0 12px 48px rgba(0, 0, 0, 0.15)';
                    }
                });
                
                container.addEventListener('mouseleave', () => {
                    if (!tracebackVisible) {
                        container.style.transform = 'translateY(0)';
                        container.style.boxShadow = '0 8px 32px rgba(0, 0, 0, 0.1)';
                    }
                });
            }
        });
        </script>
        """

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
    🚀 Factory function untuk membuat modern error component
    
    Args:
        error_message: Pesan error utama
        traceback: Stack trace optional  
        title: Judul error component
        error_type: Tipe error (error, warning, info, success)
        show_traceback: Tampilkan toggle traceback
        width: Lebar component
        **kwargs: Additional arguments
        
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