from typing import Dict, Any
import yaml
from pathlib import Path
from IPython.display import display, HTML, clear_output
from smartcash.common.constants import DRIVE_DATASET_PATH, DRIVE_PREPROCESSED_PATH

class DatasetConfigManager:
    """Manages dataset configuration operations."""
    
    DEFAULT_CONFIG = {
        'data': {
            'split_ratios': {'train': 0.7, 'valid': 0.15, 'test': 0.15},
            'stratified_split': True,
            'random_seed': 42,
            'backup_before_split': True,
            'backup_dir': 'data/splits_backup',
            'dataset_path': DRIVE_DATASET_PATH,
            'preprocessed_path': DRIVE_PREPROCESSED_PATH
        }
    }
    
    @staticmethod
    def load_config(config_path: str = 'configs/dataset_config.yaml') -> Dict[str, Any]:
        """Load dataset configuration with fallback to defaults."""
        config = DatasetConfigManager.DEFAULT_CONFIG.copy()
        try:
            if Path(config_path).exists():
                with open(config_path, 'r') as f:
                    loaded = yaml.safe_load(f) or {}
                config['data'].update(loaded.get('data', {}))
        except Exception as e:
            print(f"⚠️ Failed to load config from {config_path}: {str(e)}")
        return config

    @staticmethod
    def save_config(config: Dict[str, Any], config_path: str = 'configs/dataset_config.yaml') -> bool:
        """Save dataset configuration to file."""
        try:
            Path(config_path).parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, 'w') as f:
                yaml.dump({'data': config['data']}, f, default_flow_style=False)
            return True
        except Exception as e:
            print(f"❌ Error saving config: {str(e)}")
            return False

def update_config_from_ui(config: Dict[str, Any], ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Update configuration from UI components with validation."""
    config.setdefault('data', {})
    data = config['data']

    # Update split ratios
    if 'split_sliders' in ui_components and len(ui_components['split_sliders']) >= 3:
        train, val, test = [s.value for s in ui_components['split_sliders'][:3]]
        total = train + val + test
        factor = 100.0 / total if total else 1.0
        data['split_ratios'] = {
            'train': train * factor / 100.0,
            'valid': val * factor / 100.0,
            'test': test * factor / 100.0
        }

    # Update other settings
    data.update({
        'stratified_split': ui_components.get('stratified', {}).get('value', True),
        'random_seed': ui_components.get('advanced_options', {}).get('children', [{}])[0].get('value', 42),
        'backup_before_split': ui_components.get('advanced_options', {}).get('children', [{}])[1].get('value', True),
        'backup_dir': ui_components.get('advanced_options', {}).get('children', [{}])[2].get('value', 'data/splits_backup'),
        'dataset_path': ui_components.get('data_paths', {}).get('children', [{}])[1].get('value', DRIVE_DATASET_PATH),
        'preprocessed_path': ui_components.get('data_paths', {}).get('children', [{}])[2].get('value', DRIVE_PREPROCESSED_PATH)
    })
    return config

def initialize_from_config(ui_components: Dict[str, Any], config: Dict[str, Any], env=None, logger=None) -> Dict[str, Any]:
    """Initialize UI components from configuration."""
    try:
        config = DatasetConfigManager.load_config()
        data = config['data']

        # Update split sliders
        sliders = ui_components.get('split_sliders', [])
        if len(sliders) >= 3:
            for i, key in enumerate(['train', 'valid', 'test']):
                sliders[i].value = data['split_ratios'].get(key, 0.7 - 0.55 * i) * 100

        # Update other UI components
        if 'stratified' in ui_components:
            ui_components['stratified'].value = data.get('stratified_split', True)
        
        advanced = ui_components.get('advanced_options', {}).get('children', [])
        if len(advanced) >= 3:
            advanced[0].value = data.get('random_seed', 42)
            advanced[1].value = data.get('backup_before_split', True)
            advanced[2].value = data.get('backup_dir', 'data/splits_backup')

        paths = ui_components.get('data_paths', {}).get('children', [])
        if len(paths) >= 3:
            paths[1].value = data.get('dataset_path', DRIVE_DATASET_PATH)
            paths[2].value = data.get('preprocessed_path', DRIVE_PREPROCESSED_PATH)

        # Update UI status
        _update_status_panels(ui_components, config, env)
        if logger: logger.info("✅ UI split config initialized successfully")
    except Exception as e:
        if logger: logger.error(f"❌ Error initializing UI: {str(e)}")
    return ui_components

def _update_status_panels(ui_components: Dict[str, Any], config: Dict[str, Any], env=None):
    """Update status panels and stats HTML."""
    from smartcash.ui.utils.constants import COLORS, ICONS
    drive_path = getattr(env, 'drive_path', '/content/drive/MyDrive') if env and getattr(env, 'is_drive_mounted', False) else None
    
    if 'status_panel' in ui_components and drive_path:
        ui_components['status_panel'].value = (
            f'<div style="padding:10px; background-color:{COLORS["alert_info_bg"]}; '
            f'color:{COLORS["alert_info_text"]}; margin:10px 0; border-radius:4px; '
            f'border-left:4px solid {COLORS["alert_info_text"]};">'
            f'<p style="margin:5px 0">{ICONS["info"]} Connected to Google Drive at <strong><code>{drive_path}</code></strong> 🟢</p>'
            '</div>'
        )

    if 'current_stats_html' in ui_components:
        dataset_path = config['data'].get('dataset_path', DRIVE_DATASET_PATH)
        ui_components['current_stats_html'].value = (
            f'<div style="text-align:center; padding:15px;">'
            f'<h3 style="color:{COLORS["dark"]}; margin-bottom:10px;">{ICONS["dataset"]} Dataset Info</h3>'
            f'<p style="color:{COLORS["dark"]};">Click <strong>Visualize Class Distribution</strong> to see dataset stats.</p>'
            f'<p style="color:{COLORS["muted"]};">Dataset path: {dataset_path}</p>'
            '</div>'
        )

def _handle_button_action(action: str, ui_components: Dict[str, Any], config: Dict[str, Any], env=None, logger=None):
    """Generic handler for button actions."""
    from smartcash.ui.utils.alert_utils import create_status_indicator
    from smartcash.ui.utils.constants import COLORS, ICONS
    
    output = ui_components.get('output_box')
    if not output:
        return

    with output:
        clear_output(wait=True)
        messages = {
            'save': (f"{ICONS['save']} Saving split configuration...", "success", "saved"),
            'reset': (f"{ICONS['refresh']} Resetting split configuration...", "success", "reset")
        }
        action_type, status, success_msg = messages[action]
        
        display(create_status_indicator("info", action_type))
        try:
            if action == 'save':
                config = update_config_from_ui(config, ui_components)
                success = DatasetConfigManager.save_config(config)
            else:  # reset
                config.update(DatasetConfigManager.load_config())
                initialize_from_config(ui_components, config, env, logger)
                success = True

            if success:
                display(create_status_indicator("success", f"{ICONS['success']} Split configuration {success_msg} successfully"))
                if 'status_panel' in ui_components:
                    ui_components['status_panel'].value = (
                        f'<div style="padding:10px; background-color:{COLORS["alert_success_bg"]}; '
                        f'color:{COLORS["alert_success_text"]}; margin:10px 0; border-radius:4px; '
                        f'border-left:4px solid {COLORS["alert_success_text"]};">'
                        f'<p style="margin:5px 0">{ICONS["success"]} Split configuration {success_msg}</p>'
                        '</div>'
                    )
            else:
                display(create_status_indicator("error", f"{ICONS['error']} Failed to {action} configuration"))
            if logger: logger.info(f"✅ Split configuration {success_msg} successfully")
        except Exception as e:
            if logger: logger.error(f"❌ Error during {action}: {str(e)}")
            display(create_status_indicator("error", f"{ICONS['error']} Error: {str(e)}"))

def handle_slider_change(change, ui_components: Dict[str, Any]):
    """Handle slider percentage changes."""
    if change['name'] != 'value' or 'split_sliders' not in ui_components or len(ui_components['split_sliders']) < 3:
        return
    
    sliders = ui_components['split_sliders']
    total = sum(s.value for s in sliders)
    if abs(total - 100.0) <= 0.5:
        return

    changed_idx = next((i for i, s in enumerate(sliders) if s is change['owner']), None)
    if changed_idx is not None:
        remaining = 100.0 - change['new']
        other_total = sum(s.value for i, s in enumerate(sliders) if i != changed_idx)
        if other_total > 0:
            ratio = remaining / other_total
            for i, slider in enumerate(sliders):
                if i != changed_idx:
                    slider.value *= ratio

def handle_visualize_button(b, ui_components: Dict[str, Any], config: Dict[str, Any], env=None, logger=None):
    """Handle dataset visualization button."""
    output = ui_components.get('output_box')
    if not output:
        return

    with output:
        clear_output(wait=True)
        from smartcash.ui.utils.alert_utils import create_status_indicator
        from smartcash.ui.utils.constants import ICONS, COLORS
        display(create_status_indicator("info", f"{ICONS['chart']} Preparing dataset visualization..."))

        try:
            from smartcash.ui.dataset.split_config_visualization import (
                get_dataset_stats, update_stats_cards, show_distribution_visualization
            )
            stats = get_dataset_stats(config, env, logger)
            if 'current_stats_html' in ui_components:
                update_stats_cards(ui_components['current_stats_html'], stats, COLORS)
            show_distribution_visualization(output, config, env, logger)
            if logger: logger.info("✅ Dataset visualization displayed successfully")
        except Exception as e:
            if logger: logger.error(f"❌ Error displaying visualization: {str(e)}")
            display(create_status_indicator("error", f"{ICONS['error']} Error: {str(e)}"))
            
def register_event_handlers(ui_components: Dict[str, Any], config: Dict[str, Any], env=None, logger=None) -> Dict[str, Any]:
    """Register all event handlers for split config UI components."""
    for slider in ui_components.get('split_sliders', []):
        if hasattr(slider, 'observe'):
            slider.observe(lambda change: handle_slider_change(change, ui_components), names='value')

    for button, handler in [
        ('visualize_button', lambda b: handle_visualize_button(b, ui_components, config, env, logger)),
        ('save_button', lambda b: _handle_button_action('save', ui_components, config, env, logger)),
        ('reset_button', lambda b: _handle_button_action('reset', ui_components, config, env, logger))
    ]:
        if button in ui_components and ui_components[button]:
            ui_components[button].on_click(handler)

    ui_components['cleanup'] = lambda: (
        [s.unobserve_all() for s in ui_components.get('split_sliders', []) if hasattr(s, 'unobserve_all')],
        logger.info("🧹 UI split config event handlers cleaned up") if logger else None
    )[-1]