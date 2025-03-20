"""
File: smartcash/ui/dataset/split_config_utils.py
Deskripsi: Utilitas untuk konfigurasi split dataset termasuk event handlers dan inisialisasi
"""

from typing import Dict, Any, Optional
import yaml
import os
from pathlib import Path
from IPython.display import display, HTML, clear_output

def load_dataset_config(config_path: str = 'configs/dataset_config.yaml') -> Dict[str, Any]:
    """
    Load konfigurasi dataset dari file dengan fallback ke default.
    
    Args:
        config_path: Path ke file konfigurasi
        
    Returns:
        Dictionary berisi konfigurasi dataset
    """
    default_config = {
        'data': {
            'split_ratios': {'train': 0.7, 'valid': 0.15, 'test': 0.15},
            'stratified_split': True,
            'random_seed': 42,
            'backup_before_split': True,
            'backup_dir': 'data/splits_backup',
            'dataset_path': '/content/drive/MyDrive/SmartCash',
            'preprocessed_path': '/content/drive/MyDrive/SmartCash/preprocessed'
        }
    }
    
    try:
        if os.path.exists(config_path):
            import yaml
            with open(config_path, 'r') as f:
                loaded_config = yaml.safe_load(f) or {}
                
            # Update dengan config yang dimuat
            if 'data' in loaded_config:
                for key, value in loaded_config['data'].items():
                    if key == 'split_ratios' and isinstance(value, dict):
                        default_config['data']['split_ratios'].update(value)
                    else:
                        default_config['data'][key] = value
    except Exception as e:
        print(f"⚠️ Gagal memuat konfigurasi dari {config_path}: {str(e)}")
    
    return default_config

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
        # Buat config baru untuk disimpan
        dataset_config = {'data': {}}
        
        # Salin konfigurasi yang diperlukan
        for key in ['split_ratios', 'stratified_split', 'random_seed', 'backup_before_split', 
                   'backup_dir', 'dataset_path', 'preprocessed_path']:
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
    # Split sliders
    if 'split_sliders' in ui_components and len(ui_components['split_sliders']) >= 3:
        train_pct = ui_components['split_sliders'][0].value
        val_pct = ui_components['split_sliders'][1].value
        test_pct = ui_components['split_sliders'][2].value
        
        # Normalize percentages
        total = train_pct + val_pct + test_pct
        if total != 100.0:
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
    
    # Stratified checkbox
    if 'stratified' in ui_components:
        config['data']['stratified_split'] = ui_components['stratified'].value
    
    # Advanced settings
    if 'advanced_options' in ui_components and hasattr(ui_components['advanced_options'], 'children') and len(ui_components['advanced_options'].children) >= 3:
        config['data']['random_seed'] = ui_components['advanced_options'].children[0].value
        config['data']['backup_before_split'] = ui_components['advanced_options'].children[1].value
        config['data']['backup_dir'] = ui_components['advanced_options'].children[2].value
    
    # Path settings
    if 'data_paths' in ui_components and hasattr(ui_components['data_paths'], 'children') and len(ui_components['data_paths'].children) >= 3:
        config['data']['dataset_path'] = ui_components['data_paths'].children[1].value
        config['data']['preprocessed_path'] = ui_components['data_paths'].children[2].value
    
    return config

def initialize_from_config(ui_components: Dict[str, Any], config: Dict[str, Any], env=None, logger=None) -> Dict[str, Any]:
    """
    Initialize UI dengan data dari konfigurasi.
    
    Args:
        ui_components: Dictionary berisi komponen UI
        config: Konfigurasi aplikasi
        env: Environment manager
        logger: Logger untuk logging
        
    Returns:
        Dictionary berisi komponen UI yang telah diupdate
    """
    try:
        # Load dataset config
        dataset_config = load_dataset_config()
        
        # Update config dengan dataset_config
        if 'data' in dataset_config:
            for key, value in dataset_config['data'].items():
                if 'data' not in config:
                    config['data'] = {}
                config['data'][key] = value
        
        # Validasi dan update UI dari config
        if 'split_sliders' in ui_components and len(ui_components['split_sliders']) >= 3:
            split_ratios = config.get('data', {}).get('split_ratios', {'train': 0.7, 'valid': 0.15, 'test': 0.15})
            ui_components['split_sliders'][0].value = split_ratios.get('train', 0.7) * 100
            ui_components['split_sliders'][1].value = split_ratios.get('valid', 0.15) * 100
            ui_components['split_sliders'][2].value = split_ratios.get('test', 0.15) * 100
        
        # Stratified checkbox
        if 'stratified' in ui_components:
            ui_components['stratified'].value = config.get('data', {}).get('stratified_split', True)
        
        # Advanced options
        if 'advanced_options' in ui_components and hasattr(ui_components['advanced_options'], 'children') and len(ui_components['advanced_options'].children) >= 3:
            ui_components['advanced_options'].children[0].value = config.get('data', {}).get('random_seed', 42)
            ui_components['advanced_options'].children[1].value = config.get('data', {}).get('backup_before_split', True)
            ui_components['advanced_options'].children[2].value = config.get('data', {}).get('backup_dir', 'data/splits_backup')
        
        # Path settings
        is_colab = 'google.colab' in str(globals())
        drive_mounted = False
        drive_path = None
        
        # Cek drive dari environment manager
        if env and hasattr(env, 'is_drive_mounted') and env.is_drive_mounted:
            drive_mounted = True
            drive_path = str(env.drive_path) if hasattr(env, 'drive_path') else '/content/drive/MyDrive'
        
        if 'data_paths' in ui_components and hasattr(ui_components['data_paths'], 'children') and len(ui_components['data_paths'].children) >= 3:
            dataset_path = config.get('data', {}).get('dataset_path', '/content/drive/MyDrive/SmartCash' if drive_mounted else 'data')
            preprocessed_path = config.get('data', {}).get('preprocessed_path', '/content/drive/MyDrive/SmartCash/preprocessed' if drive_mounted else 'data/preprocessed')
            
            ui_components['data_paths'].children[1].value = dataset_path
            ui_components['data_paths'].children[2].value = preprocessed_path
        
        # Update status panel
        if 'status_panel' in ui_components and drive_mounted:
            from smartcash.ui.utils.constants import COLORS, ICONS
            ui_components['status_panel'].value = f"""
            <div style="padding:10px; background-color:{COLORS['alert_info_bg']}; 
                        color:{COLORS['alert_info_text']}; margin:10px 0; border-radius:4px; 
                        border-left:4px solid {COLORS['alert_info_text']};">
                <p style="margin:5px 0">{ICONS['info']} Terhubung ke Google Drive 🟢 | Path: {drive_path}</p>
            </div>
            """
        
        if logger: logger.info(f"✅ UI konfigurasi split berhasil diinisialisasi dari config")
        
    except Exception as e:
        if logger: logger.error(f"❌ Error saat inisialisasi UI: {str(e)}")
    
    return ui_components

def handle_save_config(b, ui_components: Dict[str, Any], config: Dict[str, Any], env=None, logger=None):
    """
    Handler untuk tombol simpan konfigurasi.
    
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
        from smartcash.ui.utils.alerts import create_status_indicator
        from smartcash.ui.utils.constants import ICONS
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
                from smartcash.ui.utils.constants import COLORS
                if 'status_panel' in ui_components:
                    ui_components['status_panel'].value = f"""
                    <div style="padding:10px; background-color:{COLORS['alert_success_bg']}; 
                                color:{COLORS['alert_success_text']}; margin:10px 0; border-radius:4px; 
                                border-left:4px solid {COLORS['alert_success_text']};">
                        <p style="margin:5px 0">{ICONS['success']} Konfigurasi split berhasil disimpan</p>
                    </div>
                    """
                
                # Refresh visualisasi
                from smartcash.ui.dataset.split_config_visualization import load_and_display_dataset_stats
                load_and_display_dataset_stats(ui_components, config, env, logger)
            else:
                display(create_status_indicator("error", f"{ICONS['error']} Gagal menyimpan konfigurasi"))
                
        except Exception as e:
            if logger: logger.error(f"❌ Error saat menyimpan konfigurasi: {str(e)}")
            display(create_status_indicator("error", f"{ICONS['error']} Error: {str(e)}"))

def handle_reset_config(b, ui_components: Dict[str, Any], config: Dict[str, Any], env=None, logger=None):
    """
    Handler untuk tombol reset konfigurasi.
    
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
        from smartcash.ui.utils.alerts import create_status_indicator
        from smartcash.ui.utils.constants import ICONS
        display(create_status_indicator("info", f"{ICONS['refresh']} Reset konfigurasi split ke default..."))
        
        try:
            # Reset config ke default
            default_config = load_dataset_config()
            
            # Update config saat ini
            if 'data' not in config:
                config['data'] = {}
            for key, value in default_config['data'].items():
                config['data'][key] = value
            
            # Update UI dari konfigurasi
            initialize_from_config(ui_components, config, env, logger)
            
            display(create_status_indicator("success", f"{ICONS['success']} Konfigurasi split berhasil direset ke default"))
            
            # Update status panel
            from smartcash.ui.utils.constants import COLORS
            if 'status_panel' in ui_components:
                ui_components['status_panel'].value = f"""
                <div style="padding:10px; background-color:{COLORS['alert_success_bg']}; 
                            color:{COLORS['alert_success_text']}; margin:10px 0; border-radius:4px; 
                            border-left:4px solid {COLORS['alert_success_text']};">
                    <p style="margin:5px 0">{ICONS['success']} Konfigurasi split direset ke default</p>
                </div>
                """
            
            # Refresh visualisasi
            from smartcash.ui.dataset.split_config_visualization import load_and_display_dataset_stats
            load_and_display_dataset_stats(ui_components, config, env, logger)
        
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

def register_event_handlers(ui_components: Dict[str, Any], config: Dict[str, Any], env=None, logger=None) -> Dict[str, Any]:
    """
    Register semua event handler untuk komponen UI split config.
    
    Args:
        ui_components: Dictionary berisi widget UI
        config: Konfigurasi aplikasi
        env: Environment manager
        logger: Logger untuk logging
        
    Returns:
        Dictionary berisi widget UI dengan handler terdaftar
    """
    # Register slider handlers dengan validasi
    if 'split_sliders' in ui_components:
        for slider in ui_components['split_sliders']:
            if hasattr(slider, 'observe'):
                slider.observe(lambda change: handle_slider_change(change, ui_components), names='value')
    
    # Register button handlers dengan validasi
    if 'save_button' in ui_components and ui_components['save_button']:
        ui_components['save_button'].on_click(lambda b: handle_save_config(b, ui_components, config, env, logger))
    
    if 'reset_button' in ui_components and ui_components['reset_button']:
        ui_components['reset_button'].on_click(lambda b: handle_reset_config(b, ui_components, config, env, logger))
    
    # Tambahkan fungsi cleanup
    def cleanup():
        """Cleanup resources event handlers."""
        # Unobserve slider events
        if 'split_sliders' in ui_components:
            for slider in ui_components['split_sliders']:
                if hasattr(slider, 'unobserve_all'):
                    slider.unobserve_all()
        if logger: logger.info(f"🧹 UI split config event handlers dibersihkan")
    
    ui_components['cleanup'] = cleanup
    
    return ui_components