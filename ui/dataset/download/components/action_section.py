"""
File: smartcash/ui/dataset/download/components/action_section.py
Deskripsi: Fixed action section yang kompatibel dengan struktur UI yang sudah ada
"""

import ipywidgets as widgets
from smartcash.ui.utils.constants import COLORS

def create_action_section():
    """Create action section dengan button naming yang konsisten dengan handler expectations."""
    
    # 💾 Save/Reset buttons section
    save_reset_buttons = create_save_reset_buttons(
        save_label='Simpan',
        reset_label='Reset', 
        button_width='100px',
        container_width='100%'
    )
    
    # ℹ️ Sync info message
    sync_info = create_sync_info_message(
        message="Konfigurasi akan otomatis disinkronkan dengan Google Drive saat disimpan atau direset.",
        icon="info",
        color="#666",
        font_style="italic",
        margin_top="5px",
        width="100%"
    )
    
    # 🔘 Main action buttons dengan naming yang tepat
    action_buttons = create_main_action_buttons(
        primary_label="Download Dataset",
        primary_icon="download",
        secondary_buttons=[
            ("Check Dataset", "search", "info")
        ],
        cleanup_enabled=True
    )
    
    # 📦 Container
    container = widgets.VBox([
        save_reset_buttons['container'],
        sync_info['container'], 
        action_buttons['container']
    ], layout=widgets.Layout(align_items='flex-end', width='100%'))
    
    # 🔗 Return dengan key mapping yang tepat
    return {
        'container': container,
        
        # Save/Reset buttons (consistent keys)
        'save_button': save_reset_buttons['save_button'],
        'reset_button': save_reset_buttons['reset_button'],
        'save_reset_buttons': save_reset_buttons,
        
        # Action buttons (consistent keys) 
        'download_button': action_buttons['download_button'],    # Handler expects this key
        'check_button': action_buttons['check_button'],          # Handler expects this key
        'cleanup_button': action_buttons.get('cleanup_button'), # Handler expects this key
        'action_buttons': action_buttons,
        
        # Info components
        'sync_info': sync_info
    }

def create_save_reset_buttons(save_label='Simpan', reset_label='Reset', 
                             button_width='100px', container_width='100%'):
    """Create save dan reset buttons."""
    
    # 💾 Save button
    save_button = widgets.Button(
        description=save_label,
        button_style='primary',
        tooltip='Simpan konfigurasi saat ini',
        icon='save',
        layout=widgets.Layout(width=button_width, height='32px')
    )
    
    # 🔄 Reset button  
    reset_button = widgets.Button(
        description=reset_label,
        button_style='',
        tooltip='Reset ke nilai default',
        icon='refresh',
        layout=widgets.Layout(width=button_width, height='32px')
    )
    
    # 📦 Container
    container = widgets.HBox(
        [save_button, reset_button],
        layout=widgets.Layout(
            width=container_width,
            justify_content='flex-end',
            align_items='center',
            margin='5px 0'
        )
    )
    
    return {
        'container': container,
        'save_button': save_button,      # Key yang diharapkan handler
        'reset_button': reset_button     # Key yang diharapkan handler
    }

def create_sync_info_message(message, icon="info", color="#666", 
                           font_style="italic", margin_top="5px", width="100%"):
    """Create sync info message."""
    
    icon_map = {
        "info": "ℹ️",
        "warning": "⚠️", 
        "success": "✅"
    }
    
    icon_emoji = icon_map.get(icon, "ℹ️")
    
    info_html = f"""
    <div style="margin-top: {margin_top}; width: {width}; text-align: right;">
        <span style="color: {color}; font-style: {font_style}; font-size: 0.9em;">
            {icon_emoji} {message}
        </span>
    </div>
    """
    
    container = widgets.HTML(info_html)
    
    return {
        'container': container,
        'message': message
    }

def create_main_action_buttons(primary_label="Download Dataset", primary_icon="download",
                              secondary_buttons=None, cleanup_enabled=True):
    """Create main action buttons dengan key mapping yang tepat untuk handlers."""
    
    if secondary_buttons is None:
        secondary_buttons = [("Check Dataset", "search", "info")]
    
    # 🔘 Download button (primary)
    download_button = widgets.Button(
        description=primary_label,
        button_style='success',
        tooltip=f'Klik untuk {primary_label.lower()}',
        icon='download',
        layout=widgets.Layout(width='140px', height='35px')
    )
    
    # 🔍 Check button
    check_button = widgets.Button(
        description="Check Dataset", 
        button_style='info',
        tooltip='Periksa status dataset yang sudah ada',
        icon='search',
        layout=widgets.Layout(width='140px', height='35px')
    )
    
    # 🧹 Cleanup button (conditional)
    cleanup_button = None
    button_list = [download_button, check_button]
    
    if cleanup_enabled:
        cleanup_button = widgets.Button(
            description="Cleanup Dataset",
            button_style='warning', 
            tooltip='Hapus dataset yang sudah ada',
            icon='trash',
            layout=widgets.Layout(width='140px', height='35px')
        )
        button_list.append(cleanup_button)
    
    # 📦 Container
    container = widgets.HBox(
        button_list,
        layout=widgets.Layout(
            width='100%',
            justify_content='flex-start',
            align_items='center',
            margin='10px 0'
        )
    )
    
    # 🔗 Return dengan key yang diharapkan handler
    result = {
        'container': container,
        'download_button': download_button,  # Handler expects this exact key
        'check_button': check_button,        # Handler expects this exact key 
        'buttons': button_list
    }
    
    # Add cleanup button jika ada
    if cleanup_button:
        result['cleanup_button'] = cleanup_button  # Handler expects this exact key
    
    return result