"""
File: smartcash/ui/utils/ui_factory_utils.py
Deskripsi: Factory utilities untuk consolidated UI initialization patterns
"""

from typing import Dict, Any, Optional, Type, Callable
from IPython.display import display

from smartcash.ui.utils.logging_utils import suppress_all_outputs
from smartcash.ui.utils.fallback_utils import create_fallback_ui, safe_operation
from smartcash.ui.utils.config_cell_initializer import ConfigCellInitializer
from smartcash.ui.utils.common_initializer import CommonInitializer


def create_ui_with_suppression(initializer_class: Type, 
                              module_name: str, 
                              config_key: str = None,
                              env=None, 
                              config=None, 
                              **kwargs) -> Any:
    """
    Universal factory untuk create UI dengan automatic suppression.
    
    Args:
        initializer_class: Class initializer (ConfigCellInitializer atau CommonInitializer)
        module_name: Nama modul
        config_key: Key untuk config (untuk ConfigCellInitializer)
        env: Environment manager
        config: Config override
        **kwargs: Additional arguments
        
    Returns:
        UI components atau fallback UI
    """
    try:
        suppress_all_outputs()  # Consolidated suppression
        
        # Determine initialization type
        if issubclass(initializer_class, ConfigCellInitializer):
            config_key = config_key or f"{module_name}_config"
            initializer = initializer_class(module_name, config_key)
        elif issubclass(initializer_class, CommonInitializer):
            logger_namespace = f"smartcash.ui.{module_name}"
            initializer = initializer_class(module_name, logger_namespace)
        else:
            # Generic initializer
            initializer = initializer_class(module_name)
        
        # Initialize dengan safe operation
        result = initializer.initialize(env, config, **kwargs)
        
        # Auto-display jika result adalah widget
        if hasattr(result, 'children') or hasattr(result, 'layout'):
            display(result)
        
        return result
        
    except Exception as e:
        fallback_ui = create_fallback_ui(f"Factory error: {str(e)}", module_name)
        display(fallback_ui.get('ui', fallback_ui))
        return fallback_ui

def create_config_ui(initializer_class: Type[ConfigCellInitializer], 
                    module_name: str, 
                    config_filename: str = None,
                    env=None, 
                    config=None, 
                    **kwargs) -> Any:
    """Factory khusus untuk config UI dengan auto-display"""
    config_filename = config_filename or f"{module_name}_config"
    return create_ui_with_suppression(initializer_class, module_name, config_filename, env, config, **kwargs)

def create_processing_ui(initializer_class: Type[CommonInitializer], 
                        module_name: str, 
                        logger_namespace: str = None,
                        env=None, 
                        config=None, 
                        **kwargs) -> Any:
    """Factory khusus untuk processing UI dengan auto-display"""
    logger_namespace = logger_namespace or f"smartcash.{module_name}"
    
    # Override initializer creation untuk processing UI
    class ProcessingInitializer(initializer_class):
        def __init__(self):
            super().__init__(module_name, logger_namespace)
    
    return create_ui_with_suppression(ProcessingInitializer, module_name, None, env, config, **kwargs)

def create_notebook_cell(cell_factory: Callable, 
                        cell_name: str,
                        auto_display: bool = True,
                        **factory_kwargs) -> Any:
    """
    Create notebook cell dengan consistent pattern.
    
    Args:
        cell_factory: Factory function untuk create UI
        cell_name: Nama cell untuk error reporting
        auto_display: Auto display hasil
        **factory_kwargs: Arguments untuk factory function
        
    Returns:
        UI result atau fallback
    """
    try:
        suppress_all_outputs()
        
        with safe_operation({}, f"initializing {cell_name}"):
            result = cell_factory(**factory_kwargs)
            
            if auto_display and result:
                if hasattr(result, 'children') or hasattr(result, 'layout'):
                    display(result)
                elif isinstance(result, dict) and 'ui' in result:
                    display(result['ui'])
            
            return result
            
    except Exception as e:
        fallback = create_fallback_ui(f"Cell error: {str(e)}", cell_name)
        if auto_display:
            display(fallback.get('ui', fallback))
        return fallback

def register_cell_factory(cell_name: str, factory_function: Callable) -> Callable:
    """
    Register cell factory dengan consistent naming pattern.
    
    Args:
        cell_name: Nama cell
        factory_function: Factory function
        
    Returns:
        Wrapped factory function
    """
    def wrapped_factory(**kwargs):
        return create_notebook_cell(factory_function, cell_name, **kwargs)
    
    # Set function name untuk debugging
    wrapped_factory.__name__ = f"create_{cell_name}_cell"
    wrapped_factory.__doc__ = f"Factory untuk {cell_name} cell dengan auto-suppression"
    
    return wrapped_factory

# Pre-configured factory templates
config_cell_factory = lambda init_class, module, config_file=None: register_cell_factory(
    module, lambda **kw: create_config_ui(init_class, module, config_file, **kw)
)

processing_cell_factory = lambda init_class, module, namespace=None: register_cell_factory(
    module, lambda **kw: create_processing_ui(init_class, module, namespace, **kw)
)

# One-liner utilities untuk common patterns
quick_config_cell = lambda init_class, name: config_cell_factory(init_class, name)()
quick_processing_cell = lambda init_class, name: processing_cell_factory(init_class, name)()

def create_split_dataset_cell():
    """Factory untuk split dataset config cell"""
    from smartcash.ui.dataset.split.split_initializer import initialize_split_ui
    return create_notebook_cell(initialize_split_ui, "split_dataset")

def create_minimal_template_cell(title: str, description: str = None) -> Dict[str, Any]:
    """Create minimal template cell untuk rapid prototyping"""
    import ipywidgets as widgets
    
    description = description or f"Template untuk {title}"
    
    header = widgets.HTML(f"<h3>🚀 {title}</h3><p>ℹ️ {description}</p>")
    status = widgets.Output(layout=widgets.Layout(width='100%', min_height='100px'))
    
    ui = widgets.VBox([header, status], layout=widgets.Layout(width='100%', padding='15px'))
    
    result = {
        'ui': ui,
        'status': status,
        'title': title,
        'template_mode': True
    }
    
    display(ui)
    return result

# Decorator untuk auto-factory creation
def ui_cell(cell_name: str, auto_display: bool = True):
    """Decorator untuk convert function jadi UI cell factory"""
    def decorator(func: Callable):
        def wrapper(**kwargs):
            return create_notebook_cell(func, cell_name, auto_display, **kwargs)
        
        wrapper.__name__ = f"create_{cell_name}_cell"
        wrapper.__doc__ = func.__doc__ or f"UI cell untuk {cell_name}"
        return wrapper
    
    return decorator

# Usage example dengan decorator:
# @ui_cell("my_config")
# def my_config_factory(env=None, config=None):
#     return create_config_ui(MyConfigInitializer, "my_module", "my_config", env, config)