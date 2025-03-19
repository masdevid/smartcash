"""
File: smartcash/ui/dataset/split_config_handlers.py
Deskripsi: Event handlers untuk konfigurasi split dataset dengan dukungan drive
"""

from typing import Dict, Any
from IPython.display import display, HTML, clear_output
import threading
import yaml
import os
from pathlib import Path

from smartcash.ui.utils.constants import ICONS, COLORS
from smartcash.ui.components.alerts import create_status_indicator

def load_dataset_config(config_path: str = 'configs/dataset_config.yaml') -> Dict[str, Any]:
    """Load konfigurasi split dataset dari file."""
    default_config = {
        'data': {
            'split_ratios': {'train': 0.7, 'valid': 0.15, 'test': 0.15},
            'stratified_split': True,
            'random_seed': 42,
            'backup_before_split': True,
            'backup_dir': 'data/splits_backup',
            'use_drive': False,
            'drive_path': '',
            'local_clone_path': 'data_local',
            'sync_on_change': True
        }
    }
    
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                loaded_config = yaml.safe_load(f) or {}
                
            # Update dengan config yang dimuat
            if 'data' in loaded_config:
                for key, value in loaded_config['data'].items():
                    if key == 'split_ratios' and isinstance(value, dict):
                        default_config['data']['split_ratios'].update(value)
                    else:
                        default_config['data'][key] = value
    except Exception:
        pass
    
    return default_config

def save_dataset_config(config: Dict[str, Any], config_path: str = 'configs/dataset_config.yaml') -> bool:
    """Simpan konfigurasi dataset ke file."""
    try:
        # Buat config baru untuk disimpan (hindari reference)
        dataset_config = {'data': {}}
        
        # Salin konfigurasi yang diperlukan
        for key in ['split_ratios', 'stratified_split', 'random_seed', 'backup_before_split', 
                   'backup_dir', 'use_drive', 'drive_path', 'local_clone_path', 'sync_on_change']:
            if key in config.get('data', {}):
                dataset_config['data'][key] = config['data'][key]
        
        # Buat direktori jika belum ada
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Simpan ke file
        with open(config_path, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
            
        return True
    except Exception as e:
        print(f"❌ Error saat menyimpan konfigurasi: {str(e)}")
        return False

def update_config_from_ui(config: Dict[str, Any], ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Update konfigurasi dari komponen UI."""
    # Split values
    train_pct = ui_components['split_sliders'][0].value
    val_pct = ui_components['split_sliders'][1].value
    test_pct = ui_components['split_sliders'][2].value
    
    # Normalize percentages
    total = train_pct + val_pct + test_pct
    if abs(total - 100.0) > 0.001:
        factor = 100.0 / total
        train_pct *= factor
        val_pct *= factor
        test_pct *= factor
    
    # Update split ratios
    config['data']['split_ratios'] = {
        'train': train_pct / 100.0,
        'valid': val_pct / 100.0,
        'test': test_pct / 100.0
    }
    
    # Update stratified
    config['data']['stratified_split'] = ui_components['stratified'].value
    
    # Advanced settings
    config['data']['random_seed'] = ui_components['advanced_options'].children[0].value
    config['data']['backup_before_split'] = ui_components['advanced_options'].children[1].value
    config['data']['backup_dir'] = ui_components['advanced_options'].children[2].value
    
    # Drive settings if available
    if 'drive_options' in ui_components:
        drive_opts = ui_components['drive_options'].children
        config['data']['use_drive'] = drive_opts[0].value
        config['data']['drive_path'] = drive_opts[1].value
        config['data']['local_clone_path'] = drive_opts[2].value
        config['data']['sync_on_change'] = drive_opts[3].value
    
    return config

def update_ui_from_config(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Update komponen UI dari konfigurasi."""
    # Split sliders
    split_ratios = config.get('data', {}).get('split_ratios', {'train': 0.7, 'valid': 0.15, 'test': 0.15})
    ui_components['split_sliders'][0].value = split_ratios.get('train', 0.7) * 100
    ui_components['split_sliders'][1].value = split_ratios.get('valid', 0.15) * 100
    ui_components['split_sliders'][2].value = split_ratios.get('test', 0.15) * 100
    
    # Stratified checkbox
    ui_components['stratified'].value = config.get('data', {}).get('stratified_split', True)
    
    # Advanced options
    ui_components['advanced_options'].children[0].value = config.get('data', {}).get('random_seed', 42)
    ui_components['advanced_options'].children[1].value = config.get('data', {}).get('backup_before_split', True)
    ui_components['advanced_options'].children[2].value = config.get('data', {}).get('backup_dir', 'data/splits_backup')
    
    # Drive options jika tersedia
    if 'drive_options' in ui_components:
        ui_components['drive_options'].children[0].value = config.get('data', {}).get('use_drive', False)
        ui_components['drive_options'].children[1].value = config.get('data', {}).get('drive_path', '')
        ui_components['drive_options'].children[2].value = config.get('data', {}).get('local_clone_path', 'data_local')
        ui_components['drive_options'].children[3].value = config.get('data', {}).get('sync_on_change', True)

def handle_save_config(ui_components: Dict[str, Any], config: Dict[str, Any], env=None, logger=None):
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
            except ImportError:
                pass
            
            if success:
                display(create_status_indicator("success", f"{ICONS['success']} Konfigurasi split berhasil disimpan"))
                
                # Update status panel
                try:
                    from smartcash.ui.dataset.download_initialization import update_status_panel
                    update_status_panel(ui_components, "success", "Konfigurasi split berhasil disimpan")
                except ImportError:
                    pass
                
                # Sinkronkan drive jika perlu
                if config.get('data', {}).get('use_drive', False) and config.get('data', {}).get('sync_on_change', True):
                    display(create_status_indicator("info", f"{ICONS['sync']} Menyinkronkan data dari Google Drive..."))
                    
                    def sync_callback(status, message):
                        with ui_components['output_box']:
                            display(create_status_indicator(status, message))
                    
                    # Import dan jalankan sinkronisasi async
                    from smartcash.ui.utils.drive_detector import async_sync_drive
                    async_sync_drive(config, env, logger, sync_callback)
                
                # Tampilkan visualisasi
                from smartcash.ui.dataset.split_config_visualization import (
                    show_class_distribution_visualization,
                    get_class_distribution
                )
                show_class_distribution_visualization(
                    ui_components['output_box'],
                    get_class_distribution(config, env, logger),
                    COLORS,
                    logger
                )
            else:
                display(create_status_indicator("error", f"{ICONS['error']} Gagal menyimpan konfigurasi"))
                
        except Exception as e:
            if logger: logger.error(f"❌ Error saat menyimpan konfigurasi: {str(e)}")
            display(create_status_indicator("error", f"{ICONS['error']} Error: {str(e)}"))

def handle_reset_config(ui_components: Dict[str, Any], config: Dict[str, Any], env=None, logger=None):
    """Handler untuk tombol reset konfigurasi."""
    with ui_components['output_box']:
        clear_output(wait=True)
        display(create_status_indicator("info", f"{ICONS['refresh']} Reset konfigurasi split ke default..."))
        
        try:
            # Reset config ke default
            default_config = load_dataset_config()
            
            # Update config saat ini
            for key, value in default_config['data'].items():
                config['data'][key] = value
            
            # Update UI
            update_ui_from_config(ui_components, config)
            
            display(create_status_indicator("success", f"{ICONS['success']} Konfigurasi split berhasil direset ke default"))
            
            # Update status panel
            try:
                from smartcash.ui.dataset.download_initialization import update_status_panel
                update_status_panel(ui_components, "success", "Konfigurasi split direset ke default")
            except ImportError:
                pass
            
            # Tampilkan visualisasi
            from smartcash.ui.dataset.split_config_visualization import (
                show_class_distribution_visualization,
                get_class_distribution
            )
            show_class_distribution_visualization(
                ui_components['output_box'],
                get_class_distribution(config, env, logger),
                COLORS,
                logger
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

def handle_drive_sync_click(b, ui_components: Dict[str, Any], config: Dict[str, Any], env=None, logger=None):
    """Handler untuk tombol sinkronisasi Drive."""
    with ui_components['output_box']:
        clear_output(wait=True)
        display(create_status_indicator("info", f"{ICONS['sync']} Memulai sinkronisasi dari Google Drive..."))
        
        def sync_callback(status, message):
            with ui_components['output_box']:
                display(create_status_indicator(status, message))
        
        # Import dan jalankan sinkronisasi async
        try:
            from smartcash.ui.utils.drive_detector import async_sync_drive
            
            # Update config dulu dari UI
            config = update_config_from_ui(config, ui_components)
            
            # Jalankan sinkronisasi
            async_sync_drive(config, env, logger, sync_callback)
        except Exception as e:
            if logger: logger.error(f"❌ Error saat sinkronisasi drive: {str(e)}")
            display(create_status_indicator("error", f"{ICONS['error']} Error saat sinkronisasi: {str(e)}"))

def handle_drive_option_change(change, ui_components: Dict[str, Any]):
    """Handler untuk perubahan opsi drive."""
    if change['name'] != 'value':
        return
    
    # Aktivasi/deaktivasi field lain berdasarkan checkbox use_drive
    if change['owner'] is ui_components['drive_options'].children[0]:
        use_drive = change['new']
        # Aktifkan/nonaktifkan input lain
        for i in range(1, 5):
            ui_components['drive_options'].children[i].disabled = not use_drive

def register_handlers(ui_components: Dict[str, Any], config: Dict[str, Any], env=None, logger=None) -> Dict[str, Any]:
    """Register semua handler untuk komponen UI split config."""
    # Register slider handlers
    for slider in ui_components['split_sliders']:
        slider.observe(lambda change: handle_slider_change(change, ui_components), names='value')
    
    # Register button handlers
    ui_components['save_button'].on_click(lambda b: handle_save_config(ui_components, config, env, logger))
    ui_components['reset_button'].on_click(lambda b: handle_reset_config(ui_components, config, env, logger))
    
    # Register drive handlers jika tersedia
    if 'drive_options' in ui_components:
        # Sync button
        ui_components['drive_sync_button'].on_click(lambda b: handle_drive_sync_click(b, ui_components, config, env, logger))
        
        # Drive option checkbox (use_drive)
        ui_components['drive_options'].children[0].observe(lambda change: handle_drive_option_change(change, ui_components), names='value')
    
    # Tambahkan fungsi cleanup
    def cleanup():
        """Cleanup resources."""
        # Unobserve slider events
        for slider in ui_components['split_sliders']:
            slider.unobserve_all()
        
        # Unobserve drive events jika ada
        if 'drive_options' in ui_components:
            ui_components['drive_options'].children[0].unobserve_all()
        
        if logger: logger.info(f"{ICONS['success']} Resources cleaned up")
    
    ui_components['cleanup'] = cleanup
    
    return ui_components