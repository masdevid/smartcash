"""
File: smartcash/ui/components/save_reset_buttons.py
Deskripsi: Save reset buttons dengan flex layout dan compact styling
"""

import ipywidgets as widgets
from typing import Dict, Any, Literal, Optional

def create_save_reset_buttons(
    save_label: str = "Simpan", 
    reset_label: str = "Reset", 
    button_width: str = '100px',
    container_width: str = '100%', 
    save_tooltip: str = "Simpan konfigurasi saat ini",
    reset_tooltip: str = "Reset ke nilai default", 
    with_sync_info: bool = False,
    sync_message: str = "Konfigurasi akan otomatis disinkronkan.",
    show_icons: bool = False,
    alignment: Literal['left', 'center', 'right'] = 'right'
) -> Dict[str, Any]:
    """Create save dan reset buttons dengan flex layout.
    
    Args:
        save_label: Label text for save button
        reset_label: Label text for reset button
        button_width: Width of each button
        container_width: Width of the container
        save_tooltip: Tooltip text for save button
        reset_tooltip: Tooltip text for reset button
        with_sync_info: Whether to show sync info message
        sync_message: Text for sync info message
        show_icons: Whether to show icons on buttons (False for text only)
        alignment: Button alignment ('left', 'center', or 'right')
        
    Returns:
        Dictionary containing container and button widgets
    """
    
    # Check if save label already has emoji (simple unicode range check)
    def has_emoji(text):
        # Basic emoji range in unicode
        emoji_ranges = [
            (0x1F600, 0x1F64F),  # Emoticons
            (0x1F300, 0x1F5FF),  # Misc Symbols and Pictographs
            (0x1F680, 0x1F6FF),  # Transport and Map
            (0x1F1E0, 0x1F1FF),  # Flags (iOS)
            (0x2600, 0x26FF),    # Misc symbols
            (0x2700, 0x27BF),    # Dingbats
            (0xFE00, 0xFE0F)     # Variation Selectors
        ]
        return any(ord(char) in range(r[0], r[1] + 1) for char in text for r in emoji_ranges)
    
    save_has_emoji = has_emoji(save_label)
    
    # Only add emoji if label doesn't already have one
    save_emoji = '💾 ' if (show_icons and not save_has_emoji) else ''
    save_button = widgets.Button(
        description=f"{save_emoji}{save_label}".strip(),
        button_style='', 
        tooltip=save_tooltip,
        layout=widgets.Layout(
            width=button_width, 
            height='32px', 
            margin='0 4px 0 0',
            border='1px solid #4CAF50',  # Green border for save
            font_weight='500'
        ),
        style={
            'button_color': '#E8F5E9',  # Light green background
            'text_color': '#2E7D32'     # Dark green text
        }
    )
    
    # Check if reset label already has emoji
    reset_has_emoji = has_emoji(reset_label)
    
    # Only add emoji if label doesn't already have one
    reset_emoji = '↩️ ' if (show_icons and not reset_has_emoji) else ''
    reset_button = widgets.Button(
        description=f"{reset_emoji}{reset_label}".strip(),
        button_style='',
        tooltip=reset_tooltip,
        layout=widgets.Layout(
            width=button_width, 
            height='32px', 
            margin='0',
            border='1px solid #757575',  # Gray border for reset
            font_weight='500'
        ),
        style={
            'button_color': '#FAFAFA',  # Light gray background
            'text_color': '#424242'     # Dark gray text
        }
    )
    
    # Map alignment parameter to justify-content CSS value
    justify_content_map = {
        'left': 'flex-start',
        'center': 'center',
        'right': 'flex-end'
    }
    justify_content = justify_content_map.get(alignment, 'flex-end')  # Default to right alignment
    
    # Flex container untuk buttons
    button_container = widgets.HBox([save_button, reset_button], 
        layout=widgets.Layout(
            width='auto', 
            display='flex',
            flex_flow='row nowrap',
            justify_content=justify_content,  # Use the mapped alignment value
            align_items='center', 
            gap='6px',
            margin='0', 
            padding='0'
        ))
    
    components = [button_container]
    sync_info_widget = None
    
    if with_sync_info and sync_message:
        sync_info_widget = widgets.HTML(f"""
        <div style='margin-top: 3px; font-style: italic; color: #666; text-align: right; 
                    font-size: 10px; line-height: 1.2; max-width: 100%; overflow: hidden; 
                    text-overflow: ellipsis;'>
            ℹ️ {sync_message}
        </div>""",
        layout=widgets.Layout(width='100%', margin='0'))
        components.append(sync_info_widget)
    
    container = widgets.VBox(components, 
        layout=widgets.Layout(
            width=container_width, 
            max_width='100%', 
            margin='4px 0', 
            padding='0', 
            overflow='hidden',
            display='flex',
            flex_flow='column',
            align_items='stretch'
        ))
    
    return {
        'container': container,
        'save_button': save_button, 
        'reset_button': reset_button,
        'sync_info': sync_info_widget
    }