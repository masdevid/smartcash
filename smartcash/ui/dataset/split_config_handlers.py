"""
File: smartcash/ui/dataset/split_config_handlers.py
Deskripsi: Event handlers untuk konfigurasi split dataset dengan error handling yang lebih baik
"""

from typing import Dict, Any, Optional, Callable
from IPython.display import display, HTML, clear_output
import yaml
import os
from pathlib import Path

def save_dataset_config(config: Dict[str, Any], config_path: str = 'configs/dataset_config.yaml') -> bool:
    """
    Simpan konfigurasi dataset ke file.
    
    Args:
        config: Konfigurasi yang akan disimpan
        config_path: Path ke file konfigurasi
        
    Returns:
        Boolean menunjukkan keberhasilan operasi
    """
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
    """
    Update konfigurasi dari komponen UI dengan validasi.
    
    Args:
        config: Konfigurasi yang akan diupdate
        ui_components: Dictionary berisi widget UI
        
    Returns:
        Dictionary konfigurasi yang telah diupdate
    """
    # Validasi komponen yang diperlukan
    if 'split_sliders' not in ui_components or not ui_components['split_sliders'] or len(ui_components['split_sliders']) < 3:
        return config
    
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
    if 'data' not in config:
        config['data'] = {}
    config['data']['split_ratios'] = {
        'train': train_pct / 100.0,
        'valid': val_pct / 100.0,
        'test': test_pct / 100.0
    }
    
    # Update stratified
    if 'stratified' in ui_components:
        config['data']['stratified_split'] = ui_components['stratified'].value
    
    # Advanced settings
    if ('advanced_options' in ui_components and hasattr(ui_components['advanced_options'], 'children') 
            and len(ui_components['advanced_options'].children) >= 3):
        config['data']['random_seed'] = ui_components['advanced_options'].children[0].value
        config['data']['backup_before_split'] = ui_components['advanced_options'].children[1].value
        config['data']['backup_dir'] = ui_components['advanced_options'].children[2].value
    
    # Drive settings if available
    if ('drive_options' in ui_components and hasattr(ui_components['drive_options'], 'children') 
            and len(ui_components['drive_options'].children) >= 4):
        config['data']['use_drive'] = ui_components['drive_options'].children[0].value
        config['data']['drive_path'] = ui_components['drive_options'].children[1].value
        config['data']['local_clone_path'] = ui_components['drive_options'].children[2].value
        config['data']['sync_on_change'] = ui_components['drive_options'].children[3].value
    
    return config

def handle_save_config(ui_components: Dict[str, Any], config: Dict[str, Any], env=None, logger=None):
    """
    Handler untuk tombol simpan konfigurasi.
    
    Args:
        ui_components: Dictionary berisi widget UI
        config: Konfigurasi aplikasi
        env: Environment manager
        logger: Logger untuk logging
    """
    if 'output_box' not in ui_components:
        return
        
    with ui_components['output_box']:
        clear_output(wait=True)
        from smartcash.ui.utils.constants import ICONS
        from smartcash.ui.components.alerts import create_status_indicator
        display(create_status_indicator("info", f"{ICONS['save']} Menyimpan konfigurasi split..."))
        
        try:
            # Update config dari nilai UI
            config = update_config_from_ui(config, ui_components)
            
            # Simpan konfigurasi
            config_path = 'configs/dataset_config.yaml'
            success = save_dataset_config(config, config_path)
            
            # Coba simpan juga dengan config_manager jika tersedia
            try:
                from smartcash.common.config import get_config_manager
                config_manager = get_config_manager()
                if config_manager:
                    config_manager.save_config('configs/dataset_config.yaml', config)
            except ImportError:
                pass
            
            if success:
                display(create_status_indicator("success", f"{ICONS['success']} Konfigurasi split berhasil disimpan"))
                
                # Update status panel
                try:
                    from smartcash.ui.utils.constants import COLORS
                    if 'status_panel' in ui_components:
                        ui_components['status_panel'].value = f"""
                        <div style="padding:10px; background-color:{COLORS['alert_success_bg']}; 
                                    color:{COLORS['alert_success_text']}; margin:10px 0; border-radius:4px; 
                                    border-left:4px solid {COLORS['alert_success_text']};">
                            <p style="margin:5px 0">{ICONS['success']} Konfigurasi split berhasil disimpan</p>
                        </div>
                        """
                except ImportError:
                    pass
                
                # Sinkronkan drive jika perlu
                if config.get('data', {}).get('use_drive', False) and config.get('data', {}).get('sync_on_change', True):
                    display(create_status_indicator("info", f"{ICONS['sync']} Menyinkronkan data dari Google Drive..."))
                    
                    def sync_callback(status, message):
                        with ui_components['output_box']:
                            display(create_status_indicator(status, message))
                    
                    # Import dan jalankan sinkronisasi async
                    try:
                        from smartcash.ui.utils.drive_detector import async_sync_drive
                        async_sync_drive(config, env, logger, sync_callback)
                    except ImportError as e:
                        if logger: logger.warning(f"⚠️ Tidak dapat sinkronisasi drive: {str(e)}")
                
                # Tampilkan visualisasi
                from smartcash.ui.dataset.split_config_visualization import (
                    show_class_distribution_visualization,
                    get_class_distribution
                )
                from smartcash.ui.utils.constants import COLORS
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
    """
    Handler untuk tombol reset konfigurasi.
    
    Args:
        ui_components: Dictionary berisi widget UI
        config: Konfigurasi aplikasi
        env: Environment manager
        logger: Logger untuk logging
    """
    if 'output_box' not in ui_components:
        return
        
    with ui_components['output_box']:
        clear_output(wait=True)
        from smartcash.ui.utils.constants import ICONS
        from smartcash.ui.components.alerts import create_status_indicator
        display(create_status_indicator("info", f"{ICONS['refresh']} Reset konfigurasi split ke default..."))
        
        try:
            # Reset config ke default
            from smartcash.ui.dataset.split_config_initialization import load_dataset_config
            default_config = load_dataset_config()
            
            # Update config saat ini
            if 'data' not in config:
                config['data'] = {}
            for key, value in default_config['data'].items():
                config['data'][key] = value
            
            # Update UI dari konfigurasi
            try:
                from smartcash.ui.dataset.split_config_initialization import _update_ui_from_config
                _update_ui_from_config(ui_components, config)
            except ImportError:
                # Fallback jika fungsi tidak tersedia
                if 'split_sliders' in ui_components and len(ui_components['split_sliders']) >= 3:
                    split_ratios = config.get('data', {}).get('split_ratios', {'train': 0.7, 'valid': 0.15, 'test': 0.15})
                    ui_components['split_sliders'][0].value = split_ratios.get('train', 0.7) * 100
                    ui_components['split_sliders'][1].value = split_ratios.get('valid', 0.15) * 100
                    ui_components['split_sliders'][2].value = split_ratios.get('test', 0.15) * 100
            
            display(create_status_indicator("success", f"{ICONS['success']} Konfigurasi split berhasil direset ke default"))
            
            # Update status panel
            try:
                from smartcash.ui.utils.constants import COLORS
                if 'status_panel' in ui_components:
                    ui_components['status_panel'].value = f"""
                    <div style="padding:10px; background-color:{COLORS['alert_success_bg']}; 
                                color:{COLORS['alert_success_text']}; margin:10px 0; border-radius:4px; 
                                border-left:4px solid {COLORS['alert_success_text']};">
                        <p style="margin:5px 0">{ICONS['success']} Konfigurasi split direset ke default</p>
                    </div>
                    """
            except ImportError:
                pass
            
            # Tampilkan visualisasi
            from smartcash.ui.dataset.split_config_visualization import (
                show_class_distribution_visualization,
                get_class_distribution
            )
            from smartcash.ui.utils.constants import COLORS
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
    """
    Handler untuk perubahan slider persentase split.
    
    Args:
        change: Event perubahan
        ui_components: Dictionary berisi widget UI
    """
    if change['name'] != 'value' or 'split_sliders' not in ui_components or len(ui_components['split_sliders']) < 3:
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
    """
    Handler untuk tombol sinkronisasi Drive.
    
    Args:
        b: Button event
        ui_components: Dictionary berisi widget UI
        config: Konfigurasi aplikasi
        env: Environment manager
        logger: Logger untuk logging
    """
    if 'output_box' not in ui_components:
        return
        
    with ui_components['output_box']:
        clear_output(wait=True)
        from smartcash.ui.utils.constants import ICONS
        from smartcash.ui.components.alerts import create_status_indicator
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
    """
    Handler untuk perubahan opsi drive.
    
    Args:
        change: Event perubahan
        ui_components: Dictionary berisi widget UI
    """
    if change['name'] != 'value' or 'drive_options' not in ui_components or not hasattr(ui_components['drive_options'], 'children'):
        return
    
    # Aktivasi/deaktivasi field lain berdasarkan checkbox use_drive
    if change['owner'] is ui_components['drive_options'].children[0]:
        use_drive = change['new']
        # Aktifkan/nonaktifkan input lain
        for i in range(1, min(5, len(ui_components['drive_options'].children))):
            ui_components['drive_options'].children[i].disabled = not use_drive

def register_handlers(ui_components: Dict[str, Any], config: Dict[str, Any], env=None, logger=None) -> Dict[str, Any]:
    """
    Register semua handler untuk komponen UI split config.
    
    Args:
        ui_components: Dictionary berisi widget UI
        config: Konfigurasi aplikasi
        env: Environment manager
        logger: Logger untuk logging
        
    Returns:
        Dictionary berisi widget UI dengan handler terdaftar
    """
    # Validasi komponen utama
    if 'split_sliders' not in ui_components or not ui_components['split_sliders']:
        if logger: logger.error("❌ Tidak dapat register handlers: komponen split_sliders tidak tersedia")
        return ui_components
    
    # Register slider handlers dengan validasi
    for slider in ui_components['split_sliders']:
        if hasattr(slider, 'observe'):
            slider.observe(lambda change: handle_slider_change(change, ui_components), names='value')
    
    # Register button handlers dengan validasi
    if 'save_button' in ui_components and ui_components['save_button']:
        ui_components['save_button'].on_click(lambda b: handle_save_config(ui_components, config, env, logger))
    
    if 'reset_button' in ui_components and ui_components['reset_button']:
        ui_components['reset_button'].on_click(lambda b: handle_reset_config(ui_components, config, env, logger))
    
    # Register drive handlers jika tersedia
    if 'drive_options' in ui_components and 'drive_sync_button' in ui_components:
        # Sync button
        ui_components['drive_sync_button'].on_click(lambda b: handle_drive_sync_click(b, ui_components, config, env, logger))
        
        # Drive option checkbox (use_drive)
        if hasattr(ui_components['drive_options'], 'children') and len(ui_components['drive_options'].children) > 0:
            ui_components['drive_options'].children[0].observe(
                lambda change: handle_drive_option_change(change, ui_components), 
                names='value'
            )
    
    # Tambahkan fungsi cleanup
    def cleanup():
        """Cleanup resources."""
        # Unobserve slider events
        if 'split_sliders' in ui_components:
            for slider in ui_components['split_sliders']:
                if hasattr(slider, 'unobserve_all'):
                    slider.unobserve_all()
        
        # Unobserve drive events jika ada
        if ('drive_options' in ui_components and hasattr(ui_components['drive_options'], 'children') 
                and len(ui_components['drive_options'].children) > 0):
            if hasattr(ui_components['drive_options'].children[0], 'unobserve_all'):
                ui_components['drive_options'].children[0].unobserve_all()
        
        if logger: logger.info(f"🧹 UI split config resources cleaned up")
    
    ui_components['cleanup'] = cleanup
    
    return ui_components