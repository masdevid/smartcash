"""
File: smartcash/ui/dataset/preprocessing_handler.py
Deskripsi: Handler untuk preprocessing dataset dengan integrasi visualisasi yang ditingkatkan
"""

from typing import Dict, Any
import os
from pathlib import Path
from IPython.display import display, clear_output

def setup_preprocessing_handlers(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """
    Setup semua handler untuk komponen UI preprocessing dataset.
    
    Args:
        ui_components: Dictionary komponen UI
        env: Environment manager
        config: Konfigurasi aplikasi
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    # Import komponen standard
    from smartcash.ui.utils.constants import ICONS

    # Dapatkan logger
    logger = ui_components.get('logger')
    
    try:
        # Persiapan awal - sertakan nilai awal di ui_components
        preprocessed_dir = config.get('preprocessing', {}).get('output_dir', 'data/preprocessed')
        data_dir = config.get('data', {}).get('dir', 'data')
        
        # Update ui_components dengan path standar yang diperlukan
        ui_components.update({
            'data_dir': data_dir,
            'preprocessed_dir': preprocessed_dir,
        })
        
        # Setup config handler
        from smartcash.ui.dataset.preprocessing_config_handler import setup_preprocessing_config_handler
        ui_components = setup_preprocessing_config_handler(ui_components, config, env)
        
        # Setup tombol handler
        from smartcash.ui.dataset.preprocessing_click_handler import setup_click_handlers
        ui_components = setup_click_handlers(ui_components, env, config)
        
        # Setup progress handler
        from smartcash.ui.dataset.preprocessing_progress_handler import setup_progress_handler
        ui_components = setup_progress_handler(ui_components, env, config)
        
        # Setup visualisasi handler
        from smartcash.ui.dataset.preprocessing_visualization_integration import setup_visualization_handlers
        ui_components = setup_visualization_handlers(ui_components, env, config)
        
        # Setup cleanup handler
        from smartcash.ui.dataset.preprocessing_cleanup_handler import setup_cleanup_handler
        ui_components = setup_cleanup_handler(ui_components, env, config)
        
        # Setup summary handler
        from smartcash.ui.dataset.preprocessing_summary_handler import setup_summary_handler
        ui_components = setup_summary_handler(ui_components, env, config)
        
        # Cek status data preprocessed yang sudah ada
        preprocessed_path = Path(preprocessed_dir)
        images_path = preprocessed_path / 'train' / 'images'
        is_preprocessed = preprocessed_path.exists() and images_path.exists() and any(images_path.glob('*.jpg'))
        
        if is_preprocessed:
            from smartcash.ui.dataset.preprocessing_initialization import update_status_panel
            
            # Update status panel
            update_status_panel(
                ui_components,
                "success",
                f"{ICONS['success']} Dataset preprocessed sudah tersedia di: {preprocessed_dir}"
            )
            
            # Tampilkan tombol cleanup dan visualisasi
            ui_components['cleanup_button'].layout.display = 'block'
            ui_components['visualization_buttons'].layout.display = 'flex'
            
            if logger: logger.info(f"{ICONS['folder']} Dataset preprocessed terdeteksi di: {os.path.abspath(preprocessed_dir)}")
            
            # Update summary jika tersedia
            try:
                from smartcash.ui.dataset.preprocessing_summary_handler import generate_preprocessing_summary
                summary = generate_preprocessing_summary(preprocessed_dir)
                
                if 'update_summary' in ui_components and callable(ui_components['update_summary']):
                    ui_components['update_summary'](summary)
            except Exception as e:
                if logger: logger.debug(f"{ICONS['info']} {str(e)}")
        
        # Setup dataset manager jika belum ada
        if 'dataset_manager' not in ui_components:
            try:
                from smartcash.ui.utils.fallback_utils import get_dataset_manager
                dataset_manager = get_dataset_manager(config, logger)
                if dataset_manager:
                    ui_components['dataset_manager'] = dataset_manager
                    
                    # Register callback progress jika ada
                    if 'register_progress_callback' in ui_components and callable(ui_components['register_progress_callback']):
                        ui_components['register_progress_callback'](dataset_manager)
            except ImportError:
                if logger: logger.debug(f"{ICONS['info']} Dataset manager tidak tersedia")
        
        # Setup observer
        try:
            from smartcash.ui.handlers.observer_handler import setup_observer_handlers
            ui_components = setup_observer_handlers(ui_components, "preprocessing_observers")
        except ImportError:
            pass
        
        # Setup cleanup function
        def cleanup_resources():
            """Bersihkan resources yang digunakan."""
            if logger: logger.info(f"{ICONS['cleanup']} Membersihkan resources preprocessing")
            
            # Unregister observer
            if 'observer_manager' in ui_components and 'observer_group' in ui_components:
                try:
                    ui_components['observer_manager'].unregister_group(ui_components['observer_group'])
                except Exception as e:
                    if logger: logger.debug(f"{ICONS['warning']} Error unregister observer: {str(e)}")
        
        # Tambahkan cleanup function ke UI components
        ui_components['cleanup'] = cleanup_resources
        
        # Log success
        if logger: logger.info(f"{ICONS['success']} Handler preprocessing berhasil diinisialisasi")
        
    except Exception as e:
        # Tampilkan error
        from smartcash.ui.utils.alert_utils import create_status_indicator
        
        if 'status' in ui_components:
            with ui_components['status']:
                clear_output(wait=True)
                display(create_status_indicator("error", f"{ICONS['error']} Error setup handler: {str(e)}"))
        
        # Log error
        if logger: logger.error(f"{ICONS['error']} Error setup preprocessing handler: {str(e)}")
    
    return ui_components

def get_preprocessing_dataset_manager(config: Dict[str, Any], logger=None):
    """
    Dapatkan dataset manager untuk preprocessing dengan fallback yang terstandarisasi.
    
    Args:
        config: Konfigurasi aplikasi
        logger: Logger
        
    Returns:
        Dataset manager instance atau None
    """
    # Gunakan fallback_utils untuk konsistensi
    from smartcash.ui.utils.fallback_utils import get_dataset_manager
    return get_dataset_manager(config, logger)

def get_preprocessing_stats(preprocessed_dir: str) -> Dict[str, Any]:
    """
    Dapatkan statistik dataset preprocessed dengan pendekatan one-liner.
    
    Args:
        preprocessed_dir: Direktori dataset preprocessed
        
    Returns:
        Dictionary statistik dataset
    """
    stats = {
        'splits': {},
        'total': {
            'images': 0,
            'labels': 0
        },
        'valid': False
    }
    
    # Cek setiap split dengan list comprehension
    for split in ['train', 'valid', 'test']:
        split_dir = Path(preprocessed_dir) / split
        if not split_dir.exists():
            stats['splits'][split] = {'exists': False, 'images': 0, 'labels': 0, 'complete': False}
            continue
            
        # Hitung gambar dan label
        images_dir, labels_dir = split_dir / 'images', split_dir / 'labels'
        num_images = len(list(images_dir.glob('*.jpg'))) + len(list(images_dir.glob('*.npy'))) if images_dir.exists() else 0
        num_labels = len(list(labels_dir.glob('*.txt'))) if labels_dir.exists() else 0
        
        # Update statistik
        stats['splits'][split] = {
            'exists': True,
            'images': num_images,
            'labels': num_labels,
            'complete': num_images > 0 and num_labels > 0 and num_images == num_labels
        }
        
        # Update total
        stats['total']['images'] += num_images
        stats['total']['labels'] += num_labels
    
    # Dataset dianggap valid jika minimal ada 1 split dengan data lengkap
    stats['valid'] = any(split_info.get('complete', False) for split_info in stats['splits'].values())
    
    return stats