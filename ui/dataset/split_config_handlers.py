"""
File: smartcash/ui/dataset/split_config_handlers.py
Deskripsi: Event handlers untuk konfigurasi split dataset
"""

from typing import Dict, Any, Optional, Callable
from IPython.display import display, HTML, clear_output
import threading

from smartcash.ui.dataset.split_config_utils import (
    load_dataset_config, 
    save_dataset_config, 
    get_default_split_config,
    normalize_split_percentages
)
from smartcash.ui.dataset.split_config_visualization import (
    get_dataset_stats,
    get_class_distribution,
    show_class_distribution_visualization,
    update_stats_cards
)

def register_handlers(
    ui_components: Dict[str, Any],
    config: Dict[str, Any],
    env=None,
    logger=None
) -> Dict[str, Any]:
    """
    Register semua handler untuk komponen UI split config.
    
    Args:
        ui_components: Dictionary berisi komponen UI
        config: Konfigurasi aplikasi
        env: Environment manager
        logger: Logger untuk logging
        
    Returns:
        Dictionary ui_components yang telah diupdate
    """
    from smartcash.ui.utils.constants import ICONS, COLORS
    from smartcash.ui.components.alerts import create_status_indicator
    
    # Handler untuk slider changes
    def on_slider_change(change):
        """Ensure split percentages sum to 100%."""
        if change['name'] != 'value':
            return
            
        # Get current values
        train_pct = ui_components['split_sliders'][0].value
        val_pct = ui_components['split_sliders'][1].value
        test_pct = ui_components['split_sliders'][2].value
        total = train_pct + val_pct + test_pct
        
        # Auto-adjust if too far from 100%
        if abs(total - 100.0) > 0.5:
            # Find which slider was just changed
            changed_slider = None
            for i, slider in enumerate(ui_components['split_sliders']):
                if slider is change['owner']:
                    changed_slider = i
                    break
            
            if changed_slider is not None:
                # Adjust the other sliders proportionally
                remaining = 100.0 - change['new']
                other_sliders = [i for i in range(3) if i != changed_slider]
                other_total = ui_components['split_sliders'][other_sliders[0]].value + ui_components['split_sliders'][other_sliders[1]].value
                
                if other_total > 0:
                    ratio = remaining / other_total
                    ui_components['split_sliders'][other_sliders[0]].value = ui_components['split_sliders'][other_sliders[0]].value * ratio
                    ui_components['split_sliders'][other_sliders[1]].value = ui_components['split_sliders'][other_sliders[1]].value * ratio
    
    # Handler untuk save button
    def on_save_config(b):
        """Save current configuration to dataset_config.yaml."""
        with ui_components['output_box']:
            clear_output(wait=True)
            display(create_status_indicator("info", f"{ICONS['save']} Menyimpan konfigurasi split..."))
            
            try:
                # Get current values from UI
                train_pct = ui_components['split_sliders'][0].value
                val_pct = ui_components['split_sliders'][1].value
                test_pct = ui_components['split_sliders'][2].value
                stratified = ui_components['stratified'].value
                random_seed = ui_components['advanced_options'].children[0].value
                backup_before_split = ui_components['advanced_options'].children[1].value
                backup_dir = ui_components['advanced_options'].children[2].value
                
                # Normalize percentages
                split_ratios = normalize_split_percentages(train_pct, val_pct, test_pct)
                
                # Update config
                config['data']['split_ratios'] = split_ratios
                config['data']['stratified_split'] = stratified
                config['data']['random_seed'] = random_seed
                config['data']['backup_before_split'] = backup_before_split
                config['data']['backup_dir'] = backup_dir
                
                # Save to dataset_config.yaml
                success = save_dataset_config(config)
                
                # Try to save with config_manager too
                try:
                    from smartcash.ui.training_config.config_handler import get_config_manager
                    config_manager = get_config_manager()
                    if config_manager:
                        config_manager.save_config(config)
                except (ImportError, AttributeError):
                    pass
                
                if success:
                    display(create_status_indicator("success", 
                        f"{ICONS['success']} Konfigurasi split berhasil disimpan ke configs/dataset_config.yaml"))
                    
                    # Update status panel
                    from smartcash.ui.dataset.download_initialization import update_status_panel
                    update_status_panel(ui_components, "success", "Konfigurasi split berhasil disimpan")
                    
                    # Tampilkan kembali visualisasi
                    show_class_distribution_visualization(
                        ui_components['output_box'],
                        get_class_distribution(config, logger),
                        COLORS
                    )
                else:
                    display(create_status_indicator("error", 
                        f"{ICONS['error']} Gagal menyimpan konfigurasi. Pastikan direktori configs/ tersedia"))
                
            except Exception as e:
                if logger:
                    logger.error(f"❌ Error saat menyimpan konfigurasi: {str(e)}")
                display(create_status_indicator("error", f"{ICONS['error']} Error saat menyimpan konfigurasi: {str(e)}"))
    
    # Handler untuk reset button
    def on_reset_config(b):
        """Reset configuration to defaults."""
        with ui_components['output_box']:
            clear_output(wait=True)
            display(create_status_indicator("info", f"{ICONS['refresh']} Reset konfigurasi split ke default..."))
            
            try:
                # Get default config
                default_config = get_default_split_config()
                
                # Update current config
                config['data']['split_ratios'] = default_config['data']['split_ratios']
                config['data']['stratified_split'] = default_config['data']['stratified_split']
                config['data']['random_seed'] = default_config['data']['random_seed']
                config['data']['backup_before_split'] = default_config['data']['backup_before_split']
                
                # Update UI
                # Split sliders
                ui_components['split_sliders'][0].value = default_config['data']['split_ratios']['train'] * 100
                ui_components['split_sliders'][1].value = default_config['data']['split_ratios']['valid'] * 100
                ui_components['split_sliders'][2].value = default_config['data']['split_ratios']['test'] * 100
                
                # Stratified checkbox
                ui_components['stratified'].value = default_config['data']['stratified_split']
                
                # Advanced options
                ui_components['advanced_options'].children[0].value = default_config['data']['random_seed']
                ui_components['advanced_options'].children[1].value = default_config['data']['backup_before_split']
                
                display(create_status_indicator("success", 
                    f"{ICONS['success']} Konfigurasi split berhasil direset ke default"))
                
                # Update status panel
                from smartcash.ui.dataset.download_initialization import update_status_panel
                update_status_panel(ui_components, "success", "Konfigurasi split direset ke default")
                
                # Tampilkan kembali visualisasi
                show_class_distribution_visualization(
                    ui_components['output_box'],
                    get_class_distribution(config, logger),
                    COLORS
                )
            
            except Exception as e:
                if logger:
                    logger.error(f"❌ Error saat reset konfigurasi: {str(e)}")
                display(create_status_indicator("error", f"{ICONS['error']} Error saat reset konfigurasi: {str(e)}"))
    
    # Handler untuk perubahan config
    def on_config_change(change):
        """Handler for config changes to update visualization."""
        # Update visualization with a small delay to ensure UI is updated first
        threading.Timer(0.5, lambda: show_class_distribution_visualization(
            ui_components['output_box'],
            get_class_distribution(config, logger),
            COLORS
        )).start()
    
    # Function for cleanup
    def cleanup():
        """Cleanup resources."""
        # Unobserve slider events
        for slider in ui_components['split_sliders']:
            slider.unobserve(on_slider_change, names='value')
        
        # Unobserve config change events
        ui_components['stratified'].unobserve(on_config_change, names='value')
            
        if logger:
            logger.info(f"{ICONS['success']} Split config resources cleaned up")
    
    # Register all handlers
    for slider in ui_components['split_sliders']:
        slider.observe(on_slider_change, names='value')
        
    ui_components['save_button'].on_click(on_save_config)
    ui_components['reset_button'].on_click(on_reset_config)
    ui_components['stratified'].observe(on_config_change, names='value')
    
    # Add cleanup function
    ui_components['cleanup'] = cleanup
    
    return ui_components

def initialize_ui(
    ui_components: Dict[str, Any],
    config: Dict[str, Any],
    env=None,
    logger=None
) -> None:
    """
    Initialize UI dengan data dari konfigurasi.
    
    Args:
        ui_components: Dictionary berisi komponen UI
        config: Konfigurasi aplikasi
        env: Environment manager
        logger: Logger untuk logging
    """
    from smartcash.ui.utils.constants import COLORS
    
    try:
        # Coba load dataset config dan update config
        dataset_config = load_dataset_config()
        
        # Update config dengan dataset_config
        if 'data' in dataset_config:
            for key, value in dataset_config['data'].items():
                config['data'][key] = value
        
        # Update UI components dengan config
        # Split sliders
        split_ratios = config['data'].get('split_ratios', {'train': 0.7, 'valid': 0.15, 'test': 0.15})
        ui_components['split_sliders'][0].value = split_ratios.get('train', 0.7) * 100
        ui_components['split_sliders'][1].value = split_ratios.get('valid', 0.15) * 100
        ui_components['split_sliders'][2].value = split_ratios.get('test', 0.15) * 100
        
        # Stratified checkbox
        ui_components['stratified'].value = config['data'].get('stratified_split', True)
        
        # Advanced options
        ui_components['advanced_options'].children[0].value = config['data'].get('random_seed', 42)
        ui_components['advanced_options'].children[1].value = config['data'].get('backup_before_split', True)
        ui_components['advanced_options'].children[2].value = config['data'].get('backup_dir', 'data/splits_backup')
        
        # Update stats cards
        stats = get_dataset_stats(config, env, logger)
        update_stats_cards(ui_components['current_stats_html'], stats, COLORS)
        
        # Show class distribution
        show_class_distribution_visualization(
            ui_components['output_box'],
            get_class_distribution(config, logger),
            COLORS
        )
        
        if logger:
            logger.info(f"✅ UI konfigurasi split berhasil diinisialisasi")
            
    except Exception as e:
        if logger:
            logger.error(f"❌ Error saat inisialisasi UI: {str(e)}")