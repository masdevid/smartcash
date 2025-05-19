import ipywidgets as widgets
from smartcash.ui.utils.header_utils import create_header
from smartcash.ui.utils.constants import COLORS, ICONS
from smartcash.ui.utils.layout_utils import create_divider
from smartcash.ui.components.status_panel import create_status_panel
from .options_panel import create_options_panel
from .action_section import create_action_section
from .progress_section import create_progress_section
from .log_section import create_log_section

def create_download_ui(env=None, config=None):
    config = config or {}
    roboflow_config = config.get('data', {}).get('roboflow', {})
    # Header
    header = create_header(f"{ICONS.get('download', '📥')} Dataset Download", "Download dataset untuk training model SmartCash")
    # Status panel
    status_panel = create_status_panel("Konfigurasi download dataset", "info")
    # Options panel (form fields)
    options = create_options_panel(roboflow_config)
    # Action section
    actions = create_action_section()
    # Progress section
    progress = create_progress_section()
    # Log section
    logs = create_log_section()
    # UI assembly
    ui = widgets.VBox([
        header,
        status_panel,
        widgets.HTML(f"<h4 style='color: {COLORS.get('dark', '#333')}; margin-top: 15px; margin-bottom: 10px;'>{ICONS.get('settings', '⚙️')} Pengaturan Download</h4>"),
        options['panel'],
        widgets.VBox([
            actions['save_reset_buttons']['container'],
            actions['sync_info']['container']
        ], layout=widgets.Layout(align_items='flex-end', width='100%')),
        create_divider(),
        actions['action_buttons']['container'],
        logs['confirmation_area'],
        progress['progress_container'],
        logs['log_accordion'],
        logs['summary_container']
    ], layout=widgets.Layout(width='100%'))
    # Compose ui_components dict
    ui_components = {
        'ui': ui,
        'header': header,
        'status_panel': status_panel,
        'workspace': options['workspace'],
        'project': options['project'],
        'version': options['version'],
        'api_key': options['api_key'],
        'output_dir': options['output_dir'],
        'validate_dataset': options['validate_dataset'],
        'backup_checkbox': options['backup_checkbox'],
        'backup_dir': options['backup_dir'],
        'input_options': options['panel'],
        'download_button': actions['action_buttons']['primary_button'],
        'check_button': actions['action_buttons']['secondary_buttons'][0] if 'secondary_buttons' in actions['action_buttons'] else None,
        'save_button': actions['save_button'],
        'reset_config_button': actions['reset_button'],
        'save_reset_buttons': actions['save_reset_buttons'],
        'sync_info': actions['sync_info'],
        'cleanup_button': actions['action_buttons'].get('cleanup_button'),
        'button_container': actions['action_buttons']['container'],
        'summary_container': logs['summary_container'],
        'confirmation_area': logs['confirmation_area'],
        'module_name': 'download',
        'progress_bar': progress['progress_bar'],
        'progress_container': progress['progress_container'],
        'current_progress': progress.get('current_progress'),
        'overall_label': progress.get('overall_label'),
        'step_label': progress.get('step_label'),
        'status': logs['log_output'],
        'log_output': logs['log_output'],
        'log_accordion': logs['log_accordion']
    }
    return ui_components 