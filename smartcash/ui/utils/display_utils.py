"""
File: smartcash/ui/utils/display_utils.py
Description: Safe display utilities to handle ZMQDisplayPublisher issues
"""

def safe_display(widget, fallback_print=True):
    """
    Safely display a widget with fallback for ZMQDisplayPublisher issues.
    
    This function handles the common issue where ZMQDisplayPublisher.publish()
    gets an unexpected keyword argument 'display' in certain Jupyter/Colab environments.
    
    Args:
        widget: Widget or component to display
        fallback_print: Whether to fall back to print() if all display methods fail
    """
    # Check if we're in an interactive environment (notebook/Colab)
    try:
        from IPython import get_ipython
        if get_ipython() is None:
            # Not in IPython environment, suppress display
            return
    except ImportError:
        # IPython not available, suppress display
        return
    try:
        # Try standard IPython display
        from IPython.display import display
        display(widget)
    except TypeError as e:
        if "unexpected keyword argument 'display'" in str(e):
            # Fallback for ZMQDisplayPublisher issue
            try:
                from IPython.core.display import publish_display_data
                publish_display_data(data={
                    'text/plain': str(widget),
                    'text/html': widget._repr_html_() if hasattr(widget, '_repr_html_') else str(widget)
                })
            except Exception:
                if fallback_print:
                    print(widget)
                else:
                    raise
        else:
            raise
    except Exception as e:
        if fallback_print:
            print(widget)
        else:
            raise