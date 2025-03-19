"""
File: smartcash/ui/dataset/split_config_handlers.py
Deskripsi: Event handlers untuk konfigurasi split dataset - handler utama untuk aksi tombol
"""

from typing import Dict, Any
from IPython.display import display, HTML, clear_output
import threading

from smartcash.ui.dataset.split_config_utils import (
    load_dataset_config, 
    save_dataset_config, 
    normalize_split_percentages,
    update_config_from_ui
)
from smartcash.ui.dataset.split_config_visualization import (
    show_class_distribution_visualization,
    get_class_distribution
)
from smartcash.ui.utils.constants import ICONS, COLORS
from smartcash.ui.components.alerts import create_status_indicator

def handle_save_config(ui_components: Dict[str, Any], config: Dict[str, Any], logger=None):
    """Handler untuk tombol simpan konfigurasi."""
    with ui_components['output_box']:
        clear_output(wait=True)
        display(create_status_indicator("info", f"{ICONS['save']} Menyimpan konfigurasi split..."))
        
        try:
            # Update config dari nilai UI
            config = update_config_from_ui(config, ui_components)
            
            # Simpan konfigurasi
            config_path = 'configs/dataset_config.yaml'
            success = save_dataset_config(config, config_path)
            
            # Coba simpan juga dengan config_manager jika tersedia
            try:
                from smartcash.ui.training_config.config_handler import get_config_manager
                config_manager = get_config_manager()
                if config_manager:
                    config_manager.save_config('configs/dataset_config.yaml', config)
            except:
                pass
            
            if success:
                display(create_status_indicator("success", f"{ICONS['success']} Konfigurasi split berhasil disimpan"))
                
                # Update status panel
                try:
                    from smartcash.ui.dataset.download_initialization import update_status_panel
                    update_status_panel(ui_components, "success", "Konfigurasi split berhasil disimpan")
                except:
                    pass
                
                # Tampilkan kembali visualisasi
                show_class_distribution_visualization(
                    ui_components['output_box'],
                    get_class_distribution(config, logger),
                    COLORS
                )
            else:
                display(create_status_indicator("error", f"{ICONS['error']} Gagal menyimpan konfigurasi"))
                
        except Exception as e:
            if logger: logger.error(f"❌ Error saat menyimpan konfigurasi: {str(e)}")
            display(create_status_indicator("error", f"{ICONS['error']} Error: {str(e)}"))

def handle_reset_config(ui_components: Dict[str, Any], config: Dict[str, Any], logger=None):
    """Handler untuk tombol reset konfigurasi."""
    with ui_components['output_box']:
        clear_output(wait=True)
        display(create_status_indicator("info", f"{ICONS['refresh']} Reset konfigurasi split ke default..."))
        
        try:
            # Reset config ke default
            default_config = load_dataset_config()
            
            # Update config saat ini
            config['data']['split_ratios'] = default_config['data']['split_ratios']
            config['data']['stratified_split'] = default_config['data']['stratified_split']
            config['data']['random_seed'] = default_config['data']['random_seed']
            config['data']['backup_before_split'] = default_config['data']['backup_before_split']
            
            # Update UI
            # Split sliders
            ui_components['split_sliders'][0].value = default_config['data']['split_ratios']['train'] * 100
            ui_components['split_sliders'][1].value = default_config['data']['split_ratios']['valid'] * 100
            ui_components['split_sliders'][2].value = default_config['data']['split_ratios']['test'] * 100
            
            # Stratified checkbox dan advanced options
            ui_components['stratified'].value = default_config['data']['stratified_split']
            ui_components['advanced_options'].children[0].value = default_config['data']['random_seed']
            ui_components['advanced_options'].children[1].value = default_config['data']['backup_before_split']
            
            display(create_status_indicator("success", f"{ICONS['success']} Konfigurasi split berhasil direset ke default"))
            
            # Update status panel
            try:
                from smartcash.ui.dataset.download_initialization import update_status_panel
                update_status_panel(ui_components, "success", "Konfigurasi split direset ke default")
            except:
                pass
            
            # Tampilkan kembali visualisasi
            show_class_distribution_visualization(
                ui_components['output_box'],
                get_class_distribution(config, logger),
                COLORS
            )
        
        except Exception as e:
            if logger: logger.error(f"❌ Error saat reset konfigurasi: {str(e)}")
            display(create_status_indicator("error", f"{ICONS['error']} Error saat reset: {str(e)}"))

def handle_slider_change(change, ui_components: Dict[str, Any]):
    """Handler untuk perubahan slider persentase split."""
    if change['name'] != 'value':
        return
        
    # Dapatkan nilai saat ini
    train_pct = ui_components['split_sliders'][0].value
    val_pct = ui_components['split_sliders'][1].value
    test_pct = ui_components['split_sliders'][2].value
    total = train_pct + val_pct + test_pct
    
    # Auto-adjust jika terlalu jauh dari 100%
    if abs(total - 100.0) > 0.5:
        # Cari slider yang baru saja diubah
        changed_slider = None
        for i, slider in enumerate(ui_components['split_sliders']):
            if slider is change['owner']:
                changed_slider = i
                break
        
        if changed_slider is not None:
            # Sesuaikan slider lain secara proporsional
            remaining = 100.0 - change['new']
            other_sliders = [i for i in range(3) if i != changed_slider]
            other_total = ui_components['split_sliders'][other_sliders[0]].value + ui_components['split_sliders'][other_sliders[1]].value
            
            if other_total > 0:
                ratio = remaining / other_total
                ui_components['split_sliders'][other_sliders[0]].value = ui_components['split_sliders'][other_sliders[0]].value * ratio
                ui_components['split_sliders'][other_sliders[1]].value = ui_components['split_sliders'][other_sliders[1]].value * ratio

def handle_config_change(change, ui_components: Dict[str, Any], config: Dict[str, Any], logger=None):
    """Handler untuk perubahan konfigurasi."""
    # Update visualisasi dengan delay kecil untuk memastikan UI update dulu
    def update_visualization():
        show_class_distribution_visualization(
            ui_components['output_box'],
            get_class_distribution(config, logger),
            COLORS
        )
    
    threading.Timer(0.5, update_visualization).start()

def register_handlers(ui_components: Dict[str, Any], config: Dict[str, Any], env=None, logger=None) -> Dict[str, Any]:
    """Register semua handler untuk komponen UI split config."""
    # Register slider handlers
    for slider in ui_components['split_sliders']:
        slider.observe(lambda change: handle_slider_change(change, ui_components), names='value')
    
    # Register button handlers
    ui_components['save_button'].on_click(lambda b: handle_save_config(ui_components, config, logger))
    ui_components['reset_button'].on_click(lambda b: handle_reset_config(ui_components, config, logger))
    
    # Register config change handler
    ui_components['stratified'].observe(
        lambda change: handle_config_change(change, ui_components, config, logger), 
        names='value'
    )
    
    # Tambahkan fungsi cleanup
    def cleanup():
        """Cleanup resources."""
        # Unobserve slider events
        for slider in ui_components['split_sliders']:
            slider.unobserve_all()
        
        # Unobserve config change events
        ui_components['stratified'].unobserve_all()
        
        if logger: logger.info(f"{ICONS['success']} Resources cleaned up")
    
    ui_components['cleanup'] = cleanup
    
    return ui_components