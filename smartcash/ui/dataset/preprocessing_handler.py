"""
File: smartcash/ui/dataset/preprocessing_handler.py
Deskripsi: Handler untuk preprocessing dataset dengan integrasi visualisasi distribusi kelas
"""

from typing import Dict, Any, Optional
import os
from pathlib import Path
from IPython.display import display, clear_output

def setup_preprocessing_handlers(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """
    Setup handler untuk komponen UI preprocessing dataset.
    
    Args:
        ui_components: Dictionary komponen UI
        env: Environment manager
        config: Konfigurasi aplikasi
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    # Import komponen standard
    from smartcash.ui.utils.constants import COLORS, ICONS
    from smartcash.ui.utils.alert_utils import create_status_indicator
    
    # Import handler progress tracking
    from smartcash.ui.dataset.preprocessing_progress_handler import setup_progress_handler
    
    # Import handler baru untuk visualisasi distribusi kelas
    from smartcash.ui.dataset.preprocessing_visualization_integration import setup_preprocessing_visualization
    
    # Dapatkan logger atau buat baru
    logger = ui_components.get('logger')
    
    try:
        # Setup progress handler
        ui_components = setup_progress_handler(ui_components, env, config)
        
        # Setup handlers untuk tombol-tombol action
        if 'action_button' in ui_components:
            ui_components['action_button'].on_click(lambda b: handle_preprocessing_action(ui_components, env, config))
        
        # Setup handlers untuk tombol visualisasi jika ada
        if 'visualize_button' in ui_components:
            ui_components['visualize_button'].on_click(lambda b: handle_visualization(ui_components, env, config))
        
        # Setup integration dengan visualisasi distribusi kelas
        ui_components = setup_preprocessing_visualization(ui_components, env, config)
        
        # Setup handlers untuk tombol-tombol lainnya
        if 'save_button' in ui_components:
            ui_components['save_button'].on_click(lambda b: handle_save_config(ui_components, config))
        
        # Register cleanup function yang akan dijalankan saat cell di-reset
        def cleanup_resources():
            """Bersihkan resources yang digunakan."""
            if logger: logger.info(f"{ICONS.get('cleanup', '🧹')} Membersihkan resources preprocessing")
            
            # Unregister observer jika tersedia
            if 'observer_manager' in ui_components and 'observer_group' in ui_components:
                try:
                    ui_components['observer_manager'].unregister_group(ui_components['observer_group'])
                except Exception as e:
                    if logger: logger.debug(f"⚠️ Error unregister observer: {str(e)}")
        
        # Tambahkan cleanup function ke UI components
        ui_components['cleanup'] = cleanup_resources
        
        # Log success
        if logger: logger.info(f"{ICONS.get('success', '✅')} Preprocessing handler berhasil diinisialisasi")
        
    except Exception as e:
        # Tampilkan error
        if 'status' in ui_components:
            with ui_components['status']:
                clear_output(wait=True)
                display(create_status_indicator("error", f"{ICONS.get('error', '❌')} Error setup handler: {str(e)}"))
        
        # Log error
        if logger: logger.error(f"{ICONS.get('error', '❌')} Error setup preprocessing handler: {str(e)}")
    
    return ui_components

def handle_preprocessing_action(ui_components: Dict[str, Any], env=None, config=None) -> None:
    """
    Handler untuk aksi preprocessing utama.
    
    Args:
        ui_components: Dictionary komponen UI
        env: Environment manager
        config: Konfigurasi aplikasi
    """
    from smartcash.ui.utils.constants import ICONS
    from smartcash.ui.utils.alert_utils import create_status_indicator
    
    logger = ui_components.get('logger')
    
    # Tampilkan pesan memulai
    with ui_components['status']:
        clear_output(wait=True)
        display(create_status_indicator("info", f"{ICONS.get('processing', '🔄')} Memulai preprocessing dataset..."))
    
    try:
        # Menggunakan preprocessing manager atau service
        from smartcash.dataset.services.preprocessing.preprocessing_service import PreprocessingService
        
        # Dapatkan parameter dari UI
        params = get_preprocessing_params(ui_components)
        
        # Buat instance preprocessing service
        preprocessing_service = PreprocessingService(config, logger=logger)
        
        # Panggil metode preprocessing
        result = preprocessing_service.preprocess_dataset(**params)
        
        # Update UI dengan hasil
        if result.get('status') == 'success':
            # Tampilkan hasil sukses
            with ui_components['status']:
                clear_output(wait=True)
                display(create_status_indicator("success", f"{ICONS.get('success', '✅')} Preprocessing dataset berhasil"))
            
            # Update status panel
            if 'status_panel' in ui_components:
                ui_components['status_panel'].value = create_status_indicator(
                    "success", 
                    f"{ICONS.get('success', '✅')} Preprocessing selesai: {result.get('processed_files', 0)} file diproses"
                ).value
                
            # Tampilkan tombol visualisasi
            enable_visualization_buttons(ui_components)
            
            # Log success
            if logger: logger.success(f"{ICONS.get('success', '✅')} Preprocessing selesai: {result.get('processed_files', 0)} file diproses")
            
            # Notifikasi distribusi kelas
            analyze_class_distribution(ui_components, result.get('output_dir'))
        else:
            # Tampilkan error
            with ui_components['status']:
                clear_output(wait=True)
                display(create_status_indicator("error", f"{ICONS.get('error', '❌')} Error: {result.get('message', 'Unknown error')}"))
            
            # Log error
            if logger: logger.error(f"{ICONS.get('error', '❌')} Preprocessing error: {result.get('message')}")
    except Exception as e:
        # Tampilkan error
        with ui_components['status']:
            clear_output(wait=True)
            display(create_status_indicator("error", f"{ICONS.get('error', '❌')} Error: {str(e)}"))
        
        # Log error
        if logger: logger.error(f"{ICONS.get('error', '❌')} Error preprocessing: {str(e)}")

def handle_visualization(ui_components: Dict[str, Any], env=None, config=None) -> None:
    """
    Handler untuk aksi visualisasi dataset.
    
    Args:
        ui_components: Dictionary komponen UI
        env: Environment manager
        config: Konfigurasi aplikasi
    """
    from smartcash.ui.utils.constants import ICONS
    from smartcash.ui.utils.alert_utils import create_status_indicator
    
    logger = ui_components.get('logger')
    
    # Tampilkan pesan memulai
    with ui_components['status']:
        clear_output(wait=True)
        display(create_status_indicator("info", f"{ICONS.get('chart', '📊')} Mempersiapkan visualisasi dataset..."))
    
    try:
        # Dapatkan jalur dataset dari config
        preprocessed_dir = config.get('preprocessing', {}).get('output_dir', 'data/preprocessed')
        
        # Panggil visualisasi kelas dari modul visualisasi
        from smartcash.ui.dataset.preprocessing_visualization_integration import analyze_preprocessing_effect
        analyze_preprocessing_effect(ui_components, preprocessed_dir)
    except Exception as e:
        # Tampilkan error
        with ui_components['status']:
            clear_output(wait=True)
            display(create_status_indicator("error", f"{ICONS.get('error', '❌')} Error visualisasi: {str(e)}"))
        
        # Log error
        if logger: logger.error(f"{ICONS.get('error', '❌')} Error visualisasi: {str(e)}")

def handle_save_config(ui_components: Dict[str, Any], config: Dict[str, Any] = None) -> None:
    """
    Handler untuk menyimpan konfigurasi preprocessing.
    
    Args:
        ui_components: Dictionary komponen UI
        config: Konfigurasi aplikasi
    """
    from smartcash.ui.utils.constants import ICONS
    from smartcash.ui.utils.alert_utils import create_status_indicator
    
    logger = ui_components.get('logger')
    
    try:
        # Update config dari UI
        updated_config = update_config_from_ui(ui_components, config)
        
        # Simpan konfigurasi
        config_path = "configs/preprocessing_config.yaml"
        
        # Simpan config dengan utilitas standar
        import yaml
        from pathlib import Path
        
        # Pastikan direktori ada
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Simpan config
        with open(config_path, 'w') as f:
            yaml.dump(updated_config, f, default_flow_style=False)
        
        # Tampilkan pesan sukses
        with ui_components['status']:
            clear_output(wait=True)
            display(create_status_indicator("success", f"{ICONS.get('success', '✅')} Konfigurasi berhasil disimpan ke {config_path}"))
        
        # Log success
        if logger: logger.success(f"{ICONS.get('success', '✅')} Konfigurasi preprocessing berhasil disimpan ke {config_path}")
    except Exception as e:
        # Tampilkan error
        with ui_components['status']:
            clear_output(wait=True)
            display(create_status_indicator("error", f"{ICONS.get('error', '❌')} Error: {str(e)}"))
        
        # Log error
        if logger: logger.error(f"{ICONS.get('error', '❌')} Error menyimpan konfigurasi: {str(e)}")

def get_preprocessing_params(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ekstrak parameter preprocessing dari komponen UI.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary berisi parameter preprocessing
    """
    # Default params
    params = {
        'input_dir': 'data',
        'output_dir': 'data/preprocessed',
        'resize': True,
        'target_size': (640, 640),
        'normalize': True,
        'augment': False,
        'validate': True
    }
    
    # Extract params dari UI jika tersedia
    if 'preprocessing_options' in ui_components:
        options = ui_components['preprocessing_options']
        
        # Ekstrak parameter dari UI component
        # (implementasi ini mungkin berbeda tergantung struktur UI Anda)
        
        # Contoh ekstraksi params dari preprocessing_options
        if hasattr(options, 'children'):
            children = options.children
            
            # Ekstrak parameter berdasarkan index child atau ID
            for i, child in enumerate(children):
                if hasattr(child, 'value'):
                    # Parameter input_dir
                    if hasattr(child, 'description') and child.description == 'Input Dir:':
                        params['input_dir'] = child.value
                    
                    # Parameter output_dir
                    elif hasattr(child, 'description') and child.description == 'Output Dir:':
                        params['output_dir'] = child.value
                    
                    # Parameter resize
                    elif hasattr(child, 'description') and 'resize' in child.description.lower():
                        params['resize'] = child.value
                    
                    # Parameter normalize
                    elif hasattr(child, 'description') and 'normalize' in child.description.lower():
                        params['normalize'] = child.value
                    
                    # Parameter augment
                    elif hasattr(child, 'description') and 'augment' in child.description.lower():
                        params['augment'] = child.value
                    
                    # Parameter validate
                    elif hasattr(child, 'description') and 'validate' in child.description.lower():
                        params['validate'] = child.value
                        
    return params

def update_config_from_ui(ui_components: Dict[str, Any], config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Update konfigurasi dari komponen UI.
    
    Args:
        ui_components: Dictionary komponen UI
        config: Konfigurasi aplikasi
        
    Returns:
        Dictionary berisi konfigurasi yang diupdate
    """
    # Inisialisasi config jika None
    if config is None:
        config = {}
    
    # Pastikan ada key preprocessing
    if 'preprocessing' not in config:
        config['preprocessing'] = {}
    
    # Extract param dari UI seperti pada get_preprocessing_params
    params = get_preprocessing_params(ui_components)
    
    # Update config dengan params
    config['preprocessing'].update(params)
    
    return config

def enable_visualization_buttons(ui_components: Dict[str, Any]) -> None:
    """
    Aktifkan tombol-tombol visualisasi.
    
    Args:
        ui_components: Dictionary komponen UI
    """
    # Aktifkan tombol visualisasi jika ada
    if 'visualize_button' in ui_components:
        ui_components['visualize_button'].layout.display = 'block'
    
    # Aktifkan tombol visualisasi distribusi kelas jika ada
    if 'distribution_button' in ui_components:
        ui_components['distribution_button'].layout.display = 'block'
    
    # Aktifkan container tombol visualisasi jika ada
    if 'visualization_button_container' in ui_components:
        ui_components['visualization_button_container'].layout.display = 'flex'

def analyze_class_distribution(ui_components: Dict[str, Any], output_dir: str) -> None:
    """
    Analisis distribusi kelas setelah preprocessing.
    
    Args:
        ui_components: Dictionary komponen UI
        output_dir: Path output direktori
    """
    logger = ui_components.get('logger')
    
    try:
        # Coba import fungsi analisis dari visualisasi
        from smartcash.ui.dataset.preprocessing_visualization_integration import analyze_preprocessing_effect
        
        # Log bahwa analisis distribusi kelas sedang dilakukan
        if logger: logger.info(f"📊 Menganalisis distribusi kelas di {output_dir}")
        
        # Jalankan analisis distribusi kelas
        analyze_preprocessing_effect(ui_components, output_dir)
    except ImportError:
        # Jika tidak tersedia, skip
        if logger: logger.debug("ℹ️ Module visualisasi distribusi kelas tidak tersedia, skipping analisis")
    except Exception as e:
        # Log error analisis distribusi
        if logger: logger.warning(f"⚠️ Gagal analisis distribusi kelas: {str(e)}")