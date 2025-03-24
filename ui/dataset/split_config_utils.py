"""
File: smartcash/ui/dataset/split_config_utils.py
Deskripsi: Utilitas untuk konfigurasi split dataset dengan penanganan error yang lebih baik
"""

from typing import Dict, Any
import yaml
from pathlib import Path
from IPython.display import display, HTML, clear_output
from smartcash.dataset.utils.dataset_constants import DEFAULT_SPLITS, DEFAULT_SPLIT_RATIOS, DRIVE_DATASET_PATH, DRIVE_PREPROCESSED_PATH, DRIVE_BACKUP_PATH

# Default paths untuk dataset

class DatasetConfigManager:
    """Manages dataset configuration operations."""
    
    DEFAULT_CONFIG = {
        'data': {
            'split_ratios': DEFAULT_SPLIT_RATIOS,
            'stratified_split': True,
            'random_seed': 42,
            'backup_before_split': True,
            'backup_dir': DRIVE_BACKUP_PATH,
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
                if 'data' in loaded and isinstance(loaded['data'], dict):
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
    # Initialize config if needed
    if config is None:
        config = {'data': {}}
    elif 'data' not in config:
        config['data'] = {}
        
    data = config['data']

    # Update split ratios
    if ui_components and 'split_sliders' in ui_components and len(ui_components['split_sliders']) >= 3:
        train, val, test = [s.value for s in ui_components['split_sliders'][:3]]
        total = train + val + test
        factor = 100.0 / total if total else 1.0
        data['split_ratios'] = {
            'train': train * factor / 100.0,
            'valid': val * factor / 100.0,
            'test': test * factor / 100.0
        }

    # Update stratified split
    if ui_components and 'stratified' in ui_components:
        stratified = ui_components['stratified']
        if hasattr(stratified, 'value'):
            data['stratified_split'] = stratified.value
    
    # Update advanced options
    if ui_components and 'advanced_options' in ui_components and hasattr(ui_components['advanced_options'], 'children'):
        advanced_options = ui_components['advanced_options'].children
        if len(advanced_options) > 0 and hasattr(advanced_options[0], 'value'):
            data['random_seed'] = advanced_options[0].value
        if len(advanced_options) > 1 and hasattr(advanced_options[1], 'value'):
            data['backup_before_split'] = advanced_options[1].value
        if len(advanced_options) > 2 and hasattr(advanced_options[2], 'value'):
            data['backup_dir'] = advanced_options[2].value
    
    # Update data paths
    if ui_components and 'data_paths' in ui_components and hasattr(ui_components['data_paths'], 'children'):
        data_paths = ui_components['data_paths'].children
        if len(data_paths) > 1 and hasattr(data_paths[1], 'value'):
            data['dataset_path'] = data_paths[1].value
        if len(data_paths) > 2 and hasattr(data_paths[2], 'value'):
            data['preprocessed_path'] = data_paths[2].value
    else:
        # Default paths
        data['dataset_path'] = DRIVE_DATASET_PATH
        data['preprocessed_path'] = DRIVE_PREPROCESSED_PATH
        
    return config

def initialize_from_config(ui_components: Dict[str, Any], config: Dict[str, Any], env=None, logger=None) -> Dict[str, Any]:
    """Initialize UI components from configuration."""
    # Validasi input
    if ui_components is None:
        ui_components = {}
        
    try:
        # Load config if needed
        if config is None or not isinstance(config, dict) or 'data' not in config:
            config = DatasetConfigManager.load_config()
            
        data = config.get('data', {})

        # Update split sliders
        sliders = ui_components.get('split_sliders', [])
        if len(sliders) >= 3:
            ratios = data.get('split_ratios', DEFAULT_SPLIT_RATIOS)
            for i, key in enumerate(DEFAULT_SPLITS):
                sliders[i].value = ratios.get(key, 0.7 if i == 0 else 0.15) * 100

        # Update stratified checkbox
        if 'stratified' in ui_components:
            ui_components['stratified'].value = data.get('stratified_split', True)
        
        # Update advanced options
        advanced = ui_components.get('advanced_options', {})
        if hasattr(advanced, 'children'):
            children = advanced.children
            if len(children) >= 3:
                children[0].value = data.get('random_seed', 42)
                children[1].value = data.get('backup_before_split', True)
                children[2].value = data.get('backup_dir', 'data/splits_backup')

        # Update paths
        paths = ui_components.get('data_paths', {})
        if hasattr(paths, 'children'):
            children = paths.children
            if len(children) >= 3:
                children[1].value = data.get('dataset_path', DRIVE_DATASET_PATH)
                children[2].value = data.get('preprocessed_path', DRIVE_PREPROCESSED_PATH)

        # Update UI status
        _update_status_panels(ui_components, config, env)
        if logger: logger.info("✅ UI split config initialized successfully")
    except Exception as e:
        if logger: logger.error(f"❌ Error initializing UI: {str(e)}")
    return ui_components

def _update_status_panels(ui_components: Dict[str, Any], config: Dict[str, Any], env=None):
    """Update status panels and stats HTML."""
    # Guard against None inputs
    if ui_components is None:
        return
        
    from smartcash.ui.utils.constants import COLORS, ICONS
    
    # Check if Drive is mounted
    drive_path = None
    if env and hasattr(env, 'is_drive_mounted') and env.is_drive_mounted:
        drive_path = getattr(env, 'drive_path', '/content/drive/MyDrive')
    
    # Update status panel if available
    if 'status_panel' in ui_components and drive_path:
        ui_components['status_panel'].value = (
            f'<div style="padding:10px; background-color:{COLORS["alert_info_bg"]}; '
            f'color:{COLORS["alert_info_text"]}; margin:10px 0; border-radius:4px; '
            f'border-left:4px solid {COLORS["alert_info_text"]};">'
            f'<p style="margin:5px 0">{ICONS["info"]} Connected to Google Drive at <strong><code>{drive_path}</code></strong> 🟢</p>'
            '</div>'
        )

    # Update stats HTML if available
    if 'current_stats_html' in ui_components:
        dataset_path = DRIVE_DATASET_PATH
        if config and isinstance(config, dict) and 'data' in config and isinstance(config['data'], dict):
            dataset_path = config['data'].get('dataset_path', DRIVE_DATASET_PATH)
            
        ui_components['current_stats_html'].value = (
            f'<div style="text-align:center; padding:15px;">'
            f'<h3 style="color:{COLORS["dark"]}; margin-bottom:10px;">{ICONS["dataset"]} Dataset Info</h3>'
            f'<p style="color:{COLORS["dark"]};">Click <strong>Visualize Class Distribution</strong> to see dataset stats.</p>'
            f'<p style="color:{COLORS["muted"]};">Dataset path: {dataset_path}</p>'
            '</div>'
        )

def handle_slider_change(change, ui_components: Dict[str, Any]):
    """Handle slider percentage changes."""
    # Safety check
    if not change or not isinstance(change, dict) or 'name' not in change or change['name'] != 'value':
        return
        
    if ui_components is None or 'split_sliders' not in ui_components:
        return
        
    sliders = ui_components['split_sliders']
    if not sliders or len(sliders) < 3:
        return
    
    # Calculate total percentage
    total = sum(s.value for s in sliders)
    if abs(total - 100.0) <= 0.5:  # Close enough to 100%
        return

    # Identify which slider changed
    changed_idx = -1
    for i, s in enumerate(sliders):
        if s is change.get('owner'):
            changed_idx = i
            break
            
    if changed_idx == -1:
        return
        
    # Adjust other sliders proportionally
    remaining = 100.0 - change['new']
    other_total = sum(s.value for i, s in enumerate(sliders) if i != changed_idx)
    
    if other_total > 0:
        ratio = remaining / other_total
        for i, slider in enumerate(sliders):
            if i != changed_idx:
                slider.value *= ratio

def handle_visualize_button(b, ui_components: Dict[str, Any], config: Dict[str, Any], env=None, logger=None):
    """Handle dataset visualization button."""
    # Safety check
    if ui_components is None or 'output_box' not in ui_components:
        if logger: logger.error("❌ Output box tidak tersedia untuk visualisasi")
        return

    output = ui_components['output_box']
    with output:
        clear_output(wait=True)
        from smartcash.ui.utils.alert_utils import create_status_indicator
        from smartcash.ui.utils.constants import ICONS, COLORS
        display(create_status_indicator("info", f"{ICONS['chart']} Preparing dataset visualization..."))

        try:
            from smartcash.ui.dataset.split_config_visualization import (
                DatasetStatsManager, update_stats_cards, show_distribution_visualization
            )
            
            # Get dataset stats
            stats = DatasetStatsManager.get_stats(config, env, logger)
            
            # Update stats cards if available
            if 'current_stats_html' in ui_components:
                update_stats_cards(ui_components['current_stats_html'], stats, COLORS)
            
            # Show distribution visualization
            show_distribution_visualization(output, config, env, logger)
            
            if logger: logger.info("✅ Dataset visualization displayed successfully")
        except Exception as e:
            if logger: logger.error(f"❌ Error displaying visualization: {str(e)}")
            display(create_status_indicator("error", f"{ICONS['error']} Error: {str(e)}"))

def _handle_button_action(action: str, ui_components: Dict[str, Any], config: Dict[str, Any], env=None, logger=None):
    """Generic handler for button actions."""
    # Safety check
    if ui_components is None or 'output_box' not in ui_components:
        if logger: logger.error(f"❌ Output box tidak tersedia untuk aksi {action}")
        return

    from smartcash.ui.utils.alert_utils import create_status_indicator
    from smartcash.ui.utils.constants import COLORS, ICONS
    
    output = ui_components['output_box']
    
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
                config = DatasetConfigManager.load_config()
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

def register_event_handlers(ui_components: Dict[str, Any], config: Dict[str, Any], env=None, logger=None) -> Dict[str, Any]:
    """Register all event handlers for split config UI components."""
    # Safety check
    if ui_components is None:
        ui_components = {}
        if logger: logger.warning("⚠️ ui_components tidak boleh None, membuat dictionary kosong")
    
    # Register slider handlers
    for slider in ui_components.get('split_sliders', []):
        if hasattr(slider, 'observe'):
            slider.observe(lambda change: handle_slider_change(change, ui_components), names='value')

    # Register button handlers
    for button, handler in [
        ('visualize_button', lambda b: handle_visualize_button(b, ui_components, config, env, logger)),
        ('save_button', lambda b: _handle_button_action('save', ui_components, config, env, logger)),
        ('reset_button', lambda b: _handle_button_action('reset', ui_components, config, env, logger))
    ]:
        if button in ui_components and ui_components[button]:
            ui_components[button].on_click(handler)

    # Define cleanup function
    def cleanup():
        try:
            # Unobserve all sliders
            for s in ui_components.get('split_sliders', []):
                if hasattr(s, 'unobserve_all'):
                    s.unobserve_all()
            # Log cleanup
            if logger: logger.info("🧹 UI split config event handlers cleaned up")
            return True
        except Exception as e:
            if logger: logger.error(f"❌ Error during cleanup: {str(e)}")
            return False
    
    ui_components['cleanup'] = cleanup
    
    return ui_components