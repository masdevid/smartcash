"""
File: smartcash/ui/components/form_container.py
Deskripsi: Reusable form container with save/reset buttons at the bottom
"""

from typing import Dict, Any, Literal, Optional, Callable
import ipywidgets as widgets

from smartcash.ui.components.save_reset_buttons import create_save_reset_buttons

def create_form_container(
    save_label: str = "Simpan",
    reset_label: str = "Reset",
    save_tooltip: str = "Simpan konfigurasi saat ini",
    reset_tooltip: str = "Reset ke nilai default",
    button_width: str = '100px',
    with_sync_info: bool = False,
    sync_message: str = "Konfigurasi akan otomatis disinkronkan.",
    show_icons: bool = False,
    alignment: Literal['left', 'center', 'right'] = 'right',
    container_margin: str = "8px 0",
    container_padding: str = "0",
    form_spacing: str = "8px",
    on_save: Optional[Callable] = None,
    on_reset: Optional[Callable] = None,
    show_buttons: bool = True
) -> Dict[str, Any]:
    """
    Create a form container with optional save/reset buttons at the bottom and an empty container for custom forms.
    
    Args:
        save_label: Label text for save button
        reset_label: Label text for reset button
        save_tooltip: Tooltip text for save button
        reset_tooltip: Tooltip text for reset button
        button_width: Width of each button
        with_sync_info: Whether to show sync info message
        sync_message: Text for sync info message
        show_icons: Whether to show icons on buttons (False for text only)
        alignment: Button alignment ('left', 'center', or 'right')
        container_margin: Margin around the container
        container_padding: Padding inside the container
        form_spacing: Spacing between form container and buttons
        on_save: Callback function for save button
        on_reset: Callback function for reset button
        show_buttons: Whether to show save/reset buttons (default: True)
        
    Returns:
        Dictionary containing:
            - container: The main container widget
            - form_container: Empty container for custom form widgets
            - save_button: Save button widget
            - reset_button: Reset button widget
    """
    # Create an empty container for custom form widgets
    form_container = widgets.VBox(
        [],
        layout=widgets.Layout(
            width='100%',
            display='flex',
            flex_flow='column',
            align_items='stretch',
            margin='0' if not show_buttons else '0 0 ' + form_spacing + ' 0'  # Add spacing at bottom only if buttons are shown
        )
    )
    
    # Create container components list
    container_components = [form_container]
    
    # Result dictionary
    result = {
        'form_container': form_container
    }
    
    # Create save/reset buttons if enabled
    if show_buttons:
        buttons = create_save_reset_buttons(
            save_label=save_label,
            reset_label=reset_label,
            save_tooltip=save_tooltip,
            reset_tooltip=reset_tooltip,
            button_width=button_width,
            with_sync_info=with_sync_info,
            sync_message=sync_message,
            show_icons=show_icons,
            alignment=alignment
        )
        
        # Connect callbacks if provided
        if on_save:
            buttons['save_button'].on_click(on_save)
        
        if on_reset:
            buttons['reset_button'].on_click(on_reset)
        
        # Add buttons to container components
        container_components.append(buttons['container'])
        
        # Add button references to result
        result.update({
            'save_button': buttons['save_button'],
            'reset_button': buttons['reset_button'],
            'sync_info': buttons.get('sync_info')
        })
    
    # Create main container with flex column layout
    container = widgets.VBox(
        container_components,
        layout=widgets.Layout(
            width='100%',
            display='flex',
            flex_flow='column',
            align_items='stretch',
            margin=container_margin,
            padding=container_padding
        )
    )
    
    # Add container to result
    result['container'] = container
    
    return result
