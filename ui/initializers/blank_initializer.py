"""
File: smartcash/ui/initializers/blank_initializer.py
Deskripsi: BlankInitializer untuk cell sederhana tanpa config management complex
"""

from typing import Dict, Any, Optional, Callable
import ipywidgets as widgets
from IPython.display import display
from smartcash.ui.utils.fallback_utils import create_error_ui, show_status_safe, try_operation_safe

class BlankInitializer:
    """Simple initializer untuk cell sederhana tanpa overhead config/logger/handler"""
    
    def __init__(self, module_name: str, title: Optional[str] = None, description: Optional[str] = None):
        self.module_name = module_name
        self.title = title or module_name.replace('_', ' ').title()
        self.description = description
        self.callbacks = []
    
    def initialize(self, content_fn: Callable[[], Any] = None, **kwargs) -> Any:
        """Initialize sederhana dengan content function"""
        try:
            # Create header jika ada title/description
            header_widget = self._create_header() if self.title else None
            
            # Create main content
            if content_fn and callable(content_fn):
                main_content = try_operation_safe(
                    lambda: content_fn(**kwargs),
                    fallback_value=widgets.HTML("<p>No content available</p>")
                )
            else:
                main_content = widgets.HTML(f"<p>Simple {self.module_name} cell ready</p>")
            
            # Combine components
            components = [widget for widget in [header_widget, main_content] if widget is not None]
            container = widgets.VBox(components) if len(components) > 1 else main_content
            
            # Basic metadata
            result = {
                'ui': container,
                'module_name': self.module_name,
                'title': self.title,
                'description': self.description,
                'initialized': True
            }
            
            # Trigger callbacks
            [try_operation_safe(lambda cb=cb: cb(result)) for cb in self.callbacks]
            
            return container
            
        except Exception as e:
            return create_error_ui(f"BlankInitializer error: {str(e)}", self.module_name)
    
    def _create_header(self) -> widgets.HTML:
        """Create simple header dengan title dan description"""
        header_html = f"<h3 style='color: #2c3e50; margin: 10px 0;'>{self.title}</h3>"
        
        if self.description:
            header_html += f"<p style='color: #7f8c8d; margin: 5px 0 15px 0;'>{self.description}</p>"
        
        return widgets.HTML(header_html)
    
    # Simple callback management
    add_callback = lambda self, cb: self.callbacks.append(cb) if cb not in self.callbacks else None
    remove_callback = lambda self, cb: self.callbacks.remove(cb) if cb in self.callbacks else None
    
    def set_title(self, title: str) -> None:
        """Update title dengan one-liner"""
        self.title = title
    
    def set_description(self, description: str) -> None:
        """Update description dengan one-liner"""
        self.description = description


# Factory functions
def create_blank_cell(module_name: str, content_fn: Callable = None, 
                     title: str = None, description: str = None, **kwargs) -> Any:
    """Factory untuk create blank cell dengan content function"""
    initializer = BlankInitializer(module_name, title, description)
    return initializer.initialize(content_fn, **kwargs)

def create_simple_cell(module_name: str, html_content: str = "", 
                      title: str = None, description: str = None) -> Any:
    """Factory untuk create simple HTML cell"""
    content_fn = lambda: widgets.HTML(html_content) if html_content else widgets.HTML(f"<p>{module_name} ready</p>")
    return create_blank_cell(module_name, content_fn, title, description)

def create_widget_cell(module_name: str, widgets_list: list = None,
                      layout_type: str = 'vbox', title: str = None, description: str = None) -> Any:
    """Factory untuk create cell dengan widget list"""
    def content_fn():
        widgets_list_safe = widgets_list or [widgets.HTML(f"<p>{module_name} widget cell</p>")]
        container_class = widgets.VBox if layout_type.lower() == 'vbox' else widgets.HBox
        return container_class(widgets_list_safe)
    
    return create_blank_cell(module_name, content_fn, title, description)

# One-liner utilities
create_text_cell = lambda module, text, **kw: create_simple_cell(module, f"<div style='padding:10px;'>{text}</div>", **kw)
create_markdown_cell = lambda module, md, **kw: create_simple_cell(module, f"<div class='markdown'>{md}</div>", **kw)
create_info_cell = lambda module, info, **kw: create_simple_cell(module, f"<div style='background:#e3f2fd;padding:10px;border-radius:5px;'>{info}</div>", **kw)