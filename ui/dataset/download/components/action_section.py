from smartcash.ui.components.action_buttons import create_action_buttons
from smartcash.ui.components.save_reset_buttons import create_save_reset_buttons
from smartcash.ui.components.sync_info_message import create_sync_info_message
import ipywidgets as widgets

def create_action_section():
    save_reset_buttons = create_save_reset_buttons(
        save_label='Simpan',
        reset_label='Reset',
        button_width='100px',
        container_width='100%'
    )
    sync_info = create_sync_info_message(
        message="Konfigurasi akan otomatis disinkronkan dengan Google Drive saat disimpan atau direset.",
        icon="info",
        color="#666",
        font_style="italic",
        margin_top="5px",
        width="100%"
    )
    action_buttons = create_action_buttons(
        primary_label="Download Dataset",
        primary_icon="download",
        secondary_buttons=[
            ("Check Dataset", "search", "info")
        ],
        cleanup_enabled=True
    )
    container = widgets.VBox([
        save_reset_buttons['container'],
        sync_info['container'],
        action_buttons['container']
    ], layout=widgets.Layout(align_items='flex-end', width='100%'))
    return {
        'container': container,
        'save_button': save_reset_buttons['save_button'],
        'reset_button': save_reset_buttons['reset_button'],
        'save_reset_buttons': save_reset_buttons,
        'sync_info': sync_info,
        'action_buttons': action_buttons
    } 