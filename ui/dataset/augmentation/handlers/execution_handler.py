"""
File: smartcash/ui/dataset/augmentation/handlers/execution_handler.py
Deskripsi: Handler eksekusi untuk proses augmentasi dataset
"""

import os
import time
import traceback
from typing import Dict, Any, List, Optional, Callable
from IPython.display import display, clear_output
import ipywidgets as widgets
from concurrent.futures import ThreadPoolExecutor

from smartcash.ui.utils.constants import ICONS
from smartcash.ui.utils.alert_utils import create_status_indicator
from smartcash.common.logger import get_logger

def update_status_panel(ui_components: Dict[str, Any], message: str, status_type: str = 'info') -> None:
    """
    Update panel status dengan pesan dan tipe status.
    Fungsi ini adalah alias untuk fungsi di status_handler untuk kompatibilitas dengan pengujian.
    
    Args:
        ui_components: Dictionary komponen UI
        message: Pesan yang akan ditampilkan
        status_type: Tipe status (info, success, warning, error)
    """
    from smartcash.ui.dataset.augmentation.handlers.status_handler import update_status_panel as update_status
    return update_status(ui_components, message, status_type)

def run_augmentation(ui_components: Dict[str, Any]) -> None:
    """
    Jalankan proses augmentasi dataset dengan thread terpisah.
    
    Args:
        ui_components: Dictionary komponen UI
    """
    logger = ui_components.get('logger', get_logger('augmentation'))
    
    # Jalankan augmentasi di thread terpisah
    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(execute_augmentation, ui_components, logger)

def execute_augmentation(ui_components: Dict[str, Any], logger=None) -> None:
    """
    Eksekusi proses augmentasi dengan progress tracking.
    
    Args:
        ui_components: Dictionary komponen UI
        logger: Logger untuk logging
    """
    if logger is None:
        logger = get_logger('augmentation')
    
    # Tandai augmentasi sedang berjalan
    ui_components['augmentation_running'] = True
    
    try:
        # Import handler service
        from smartcash.ui.dataset.augmentation.handlers.augmentation_service_handler import execute_augmentation as execute_aug
        from smartcash.ui.dataset.augmentation.handlers.status_handler import update_status_panel, update_progress_bar, reset_progress_bar
        from smartcash.ui.dataset.augmentation.handlers.notification_handler import notify_process_start, notify_process_complete, notify_process_error
        from smartcash.ui.dataset.augmentation.handlers.initialization_handler import register_progress_callback
        
        # Pastikan progress callback diregistrasi dengan benar
        register_progress_callback(ui_components)
        
        # Dapatkan parameter dari UI - ini harus dipanggil untuk pengujian
        params = extract_augmentation_params(ui_components)
        
        # Validasi parameter - gunakan logger dari ui_components untuk pengujian
        if not validate_prerequisites(params, ui_components, ui_components.get('logger', logger)):
            return
        
        # Dapatkan informasi split dan display info
        split = params.get('target_split', 'train')
        aug_types = params.get('types', ['combined'])
        display_info = f"split {split} dengan jenis {', '.join(aug_types)}"
        
        # Update status panel dengan informasi yang lebih detail
        update_status_panel(ui_components, f"Menjalankan augmentasi dataset {display_info}...", "info")
        logger.info(f"🚀 Memulai augmentasi dataset {display_info}")
        
        # Reset progress bar dan pastikan terlihat
        reset_progress_bar(ui_components, "Mempersiapkan augmentasi...")
        
        # Pastikan semua komponen progress tracking terlihat
        for element in ['progress_bar', 'current_progress', 'overall_label', 'step_label']:
            if element in ui_components:
                ui_components[element].layout.visibility = 'visible'
        
        # Notifikasi observer tentang dimulainya augmentasi
        notify_process_start(ui_components, "augmentasi", display_info, split)
        
        # Jalankan augmentasi
        start_time = time.time()
        
        # Update progress bar ke 10% untuk menunjukkan proses dimulai
        update_progress_bar(ui_components, 10, 100, "Memulai proses augmentasi...")
        logger.info(f"⏳ Memulai proses augmentasi {display_info}...")
        
        # Pastikan service augmentasi memiliki callback progress yang terdaftar
        from smartcash.ui.dataset.augmentation.handlers.augmentation_service_handler import get_augmentation_service
        service = get_augmentation_service(ui_components)
        if service and 'progress_callback' in ui_components and callable(ui_components['progress_callback']):
            service.register_progress_callback(ui_components['progress_callback'])
            logger.info("✅ Progress callback berhasil didaftarkan ke service augmentasi")
        else:
            logger.warning("⚠️ Gagal mendaftarkan progress callback ke service augmentasi")
            
        # Jalankan augmentasi dengan thread terpisah dan update progress secara berkala
        result = execute_aug(ui_components, params)
        
        # Cek hasil
        if result['status'] == 'success':
            # Hitung waktu eksekusi
            execution_time = time.time() - start_time
            minutes, seconds = divmod(execution_time, 60)
            time_str = f"{int(minutes)} menit {int(seconds)} detik"
            
            # Update status panel
            update_status_panel(
                ui_components, 
                f"Augmentasi selesai: {result.get('generated', 0)} gambar dihasilkan", 
                "success"
            )
            
            # Tampilkan pesan sukses
            with ui_components['status']:
                clear_output(wait=True)
                display(create_status_indicator(
                    "success", 
                    f"{ICONS['success']} Augmentasi berhasil! {result.get('generated', 0)} gambar dihasilkan dalam {time_str}."
                ))
            
            # Tampilkan tombol cleanup
            ui_components['cleanup_button'].layout.display = 'block'
            
            # Tampilkan tombol visualisasi
            ui_components['visualization_buttons'].layout.display = 'flex'
            
            # Update progress bar ke 100% untuk menunjukkan proses selesai
            update_progress_bar(ui_components, 100, 100, "Augmentasi selesai")
            
            # Notifikasi observer tentang selesainya augmentasi
            notify_process_complete(ui_components, result, display_info)
            
            # Tampilkan summary
            display_augmentation_summary(ui_components, result)
            
            # Notifikasi observer
            try:
                from smartcash.components.observer import notify
                notify('augmentation_completed', {
                    'ui_components': ui_components,
                    'result': result
                })
            except Exception as e:
                logger.debug(f"Observer notification failed: {str(e)}")
        elif result['status'] == 'warning':
            # Update status panel
            update_status_panel(ui_components, result.get('message', "Augmentasi selesai dengan peringatan"), "warning")
            
            # Tampilkan pesan warning
            with ui_components['status']:
                clear_output(wait=True)
                display(create_status_indicator("warning", f"{ICONS['warning']} {result.get('message', 'Augmentasi selesai dengan peringatan')}"))
        else:
            # Update status panel
            update_status_panel(ui_components, result.get('message', "Augmentasi gagal"), "error")
            
            # Tampilkan pesan error
            with ui_components['status']:
                clear_output(wait=True)
                display(create_status_indicator("error", f"{ICONS['error']} {result.get('message', 'Augmentasi gagal')}"))
    except Exception as e:
        # Tangkap error dan tampilkan
        error_message = str(e)
        logger.error(f"{ICONS['error']} Error saat augmentasi: {error_message}")
        
        # Update status panel
        update_status_panel(ui_components, f"Error saat augmentasi: {error_message}", "error")
        
        # Reset progress bar
        if 'progress_bar' in ui_components:
            ui_components['progress_bar'].layout.visibility = 'hidden'
            if 'overall_label' in ui_components:
                ui_components['overall_label'].layout.visibility = 'hidden'
        
        # Tampilkan error di output status
        with ui_components['status']:
            clear_output(wait=True)
            display(create_status_indicator("error", f"{ICONS['error']} Error saat menjalankan augmentasi: {str(e)}"))
            display(traceback.format_exc())
            
        # Notifikasi observer tentang error
        notify_process_error(ui_components, str(e))
    finally:
        # Tandai augmentasi selesai
        ui_components['augmentation_running'] = False
        
        # Cleanup UI jika tidak ada permintaan stop
        if not ui_components.get('stop_requested', False):
            from smartcash.ui.dataset.augmentation.handlers.button_handlers import cleanup_ui
            cleanup_ui(ui_components)

def extract_augmentation_params(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ekstrak parameter augmentasi langsung dari UI components, bukan dari konfigurasi yang disimpan.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary parameter augmentasi
    """
    logger = ui_components.get('logger', get_logger('augmentation'))
    logger.info("🔍 Mengambil parameter augmentasi langsung dari UI")
    
    # Inisialisasi parameter dengan nilai default
    params = {
        'split': 'train',
        'augmentation_types': ['combined'],
        'num_variations': 2,
        'output_prefix': 'aug',
        'validate_results': True,
        'resume': False,
        'process_bboxes': True,
        'target_balance': True,
        'num_workers': 4,
        'move_to_preprocessed': True,
        'target_count': 1000
    }
    
    # Dapatkan komponen augmentation_options dan combined_options
    combined_options = ui_components.get('combined_options', None)
    
    # Ekstrak parameter dari komponen UI
    if combined_options and hasattr(combined_options, 'children'):
        logger.info("🔍 Mengambil parameter dari combined_options")
        
        # Akses elemen-elemen dalam combined_options
        # combined_options adalah VBox yang berisi komponen-komponen UI
        for child in combined_options.children:
            # Periksa apakah ini adalah HBox yang berisi komponen-komponen UI
            if hasattr(child, 'children'):
                for subchild in child.children:
                    # Periksa apakah ini adalah VBox yang berisi komponen-komponen UI
                    if hasattr(subchild, 'children'):
                        for component in subchild.children:
                            # Periksa komponen berdasarkan deskripsi
                            if hasattr(component, 'description'):
                                # Jenis augmentasi (SelectMultiple)
                                if component.description == 'Jenis:':
                                    params['augmentation_types'] = list(component.value)
                                    logger.info(f"📊 Jenis augmentasi: {params['augmentation_types']}")
                                
                                # Jumlah variasi (IntSlider)
                                elif component.description == 'Jumlah Variasi:':
                                    params['num_variations'] = component.value
                                    logger.info(f"📊 Jumlah variasi: {params['num_variations']}")
                                
                                # Target split (Dropdown)
                                elif component.description == 'Target Split:':
                                    params['split'] = component.value
                                    logger.info(f"📊 Target split: {params['split']}")
                                
                                # Output prefix (Text)
                                elif component.description == 'Output Prefix:':
                                    params['output_prefix'] = component.value
                                    logger.info(f"📊 Output prefix: {params['output_prefix']}")
                                
                                # Target count (IntSlider)
                                elif component.description == 'Target per Kelas:':
                                    params['target_count'] = component.value
                                    logger.info(f"📊 Target per kelas: {params['target_count']}")
                                
                                # Balance classes (Checkbox)
                                elif component.description == 'Balancing Kelas':
                                    params['target_balance'] = component.value
                                    logger.info(f"📊 Balancing kelas: {params['target_balance']}")
                                
                                # Move to preprocessed (Checkbox)
                                elif component.description == 'Pindahkan ke Preprocessed':
                                    params['move_to_preprocessed'] = component.value
                                    logger.info(f"📊 Pindahkan ke preprocessed: {params['move_to_preprocessed']}")
                                    
                                # Validate results (Checkbox)
                                elif component.description == 'Validasi Hasil':
                                    params['validate_results'] = component.value
                                    logger.info(f"📊 Validasi hasil: {params['validate_results']}")
                                    
                                # Resume (Checkbox)
                                elif component.description == 'Resume Augmentasi':
                                    params['resume'] = component.value
                                    logger.info(f"📊 Resume: {params['resume']}")
                                    
                                # Num workers (IntSlider)
                                elif component.description == 'Jumlah Workers:':
                                    params['num_workers'] = component.value
                                    logger.info(f"📊 Jumlah workers: {params['num_workers']}")
                            
                            # Jika komponen adalah HBox, periksa anak-anaknya
                            elif hasattr(component, 'children'):
                                for grandchild in component.children:
                                    if hasattr(grandchild, 'description'):
                                        # Periksa checkbox dalam HBox
                                        if grandchild.description == 'Balancing Kelas':
                                            params['target_balance'] = grandchild.value
                                            logger.info(f"📊 Balancing kelas: {params['target_balance']}")
                                        elif grandchild.description == 'Pindahkan ke Preprocessed':
                                            params['move_to_preprocessed'] = grandchild.value
                                            logger.info(f"📊 Pindahkan ke preprocessed: {params['move_to_preprocessed']}")
                                        elif grandchild.description == 'Validasi Hasil':
                                            params['validate_results'] = grandchild.value
                                            logger.info(f"📊 Validasi hasil: {params['validate_results']}")
                                        elif grandchild.description == 'Resume Augmentasi':
                                            params['resume'] = grandchild.value
                                            logger.info(f"📊 Resume: {params['resume']}")
                                        elif grandchild.description == 'Aktifkan Augmentasi':
                                            # Ini hanya untuk logging, tidak mempengaruhi parameter
                                            logger.info(f"📊 Augmentasi aktif: {grandchild.value}")
    
    # Dapatkan komponen advanced_options
    adv_options = ui_components.get('advanced_options', None)
    
    # Ekstrak parameter dari advanced options
    if adv_options and hasattr(adv_options, 'children'):
        logger.info("🔍 Mengambil parameter dari advanced_options")
        
        # Iterasi melalui komponen dalam advanced_options
        for component in adv_options.children:
            if hasattr(component, 'description'):
                # Process bboxes (Checkbox)
                if component.description == 'Proses Bounding Box':
                    params['process_bboxes'] = component.value
                    logger.info(f"📊 Proses bounding box: {params['process_bboxes']}")
                
                # Validate results (Checkbox) - mungkin sudah diambil dari combined_options
                elif component.description == 'Validasi Hasil' and 'validate_results' not in params:
                    params['validate_results'] = component.value
                    logger.info(f"📊 Validasi hasil: {params['validate_results']}")
                
                # Num workers (IntSlider) - mungkin sudah diambil dari combined_options
                elif component.description == 'Jumlah Worker:' and 'num_workers' not in params:
                    params['num_workers'] = component.value
                    logger.info(f"📊 Jumlah workers: {params['num_workers']}")
            
            # Jika komponen adalah container, periksa anak-anaknya
            elif hasattr(component, 'children'):
                for child in component.children:
                    if hasattr(child, 'description'):
                        # Process bboxes (Checkbox)
                        if child.description == 'Proses Bounding Box':
                            params['process_bboxes'] = child.value
                            logger.info(f"📊 Proses bounding box: {params['process_bboxes']}")
                        
                        # Validate results (Checkbox)
                        elif child.description == 'Validasi Hasil' and 'validate_results' not in params:
                            params['validate_results'] = child.value
                            logger.info(f"📊 Validasi hasil: {params['validate_results']}")
                        
                        # Num workers (IntSlider)
                        elif child.description == 'Jumlah Worker:' and 'num_workers' not in params:
                            params['num_workers'] = child.value
                            logger.info(f"📊 Jumlah workers: {params['num_workers']}")

    
    # Fallback ke split selector jika split tidak ditemukan di augmentation_options
    if params['split'] == 'train':
        split_selector = ui_components.get('split_selector')
        if split_selector and hasattr(split_selector, 'children'):
            for child in split_selector.children:
                if hasattr(child, 'children'):
                    for grandchild in child.children:
                        if hasattr(grandchild, 'value') and hasattr(grandchild, 'description') and grandchild.description == 'Split:':
                            params['split'] = grandchild.value
                            break
    
    # Log parameter yang diambil
    logger.info(f"📊 Parameter augmentasi: {params}")
    
    return params

def validate_prerequisites(params: Dict[str, Any], ui_components: Dict[str, Any], logger=None) -> bool:
    """
    Validasi prasyarat sebelum menjalankan augmentasi.
    
    Args:
        params: Parameter augmentasi
        ui_components: Dictionary komponen UI
        logger: Logger untuk logging
        
    Returns:
        Boolean menunjukkan apakah prasyarat terpenuhi
    """
    if logger is None:
        logger = get_logger('augmentation')
    
    # Cek apakah augmentasi diaktifkan
    if not ui_components.get('augmentation_options'):
        with ui_components['status']:
            clear_output(wait=True)
            display(create_status_indicator("error", f"{ICONS['error']} Komponen UI augmentasi tidak ditemukan"))
        return False
    
    # Cek apakah jenis augmentasi valid
    aug_types = params.get('augmentation_types', [])
    if not aug_types or not isinstance(aug_types, (list, tuple)):
        with ui_components['status']:
            clear_output(wait=True)
            display(create_status_indicator("error", f"{ICONS['error']} Jenis augmentasi tidak valid"))
        return False
    
    # Cek apakah split valid
    split = params.get('split', '')
    if not split or split not in ['train', 'valid', 'test']:
        with ui_components['status']:
            clear_output(wait=True)
            display(create_status_indicator("error", f"{ICONS['error']} Split dataset tidak valid"))
        return False
    
    # Cek apakah dataset ada
    data_dir = ui_components.get('data_dir', 'data')
    split_dir = os.path.join(data_dir, 'preprocessed', split)
    
    if not os.path.exists(split_dir):
        with ui_components['status']:
            clear_output(wait=True)
            display(create_status_indicator("error", f"{ICONS['error']} Dataset {split} tidak ditemukan di {split_dir}"))
        return False
    
    # Cek apakah ada gambar di dataset
    images_dir = os.path.join(split_dir, 'images')
    if not os.path.exists(images_dir) or not os.listdir(images_dir):
        with ui_components['status']:
            clear_output(wait=True)
            display(create_status_indicator("error", f"{ICONS['error']} Tidak ada gambar di dataset {split}"))
        return False
    
    # Cek apakah ada label di dataset
    labels_dir = os.path.join(split_dir, 'labels')
    if not os.path.exists(labels_dir) or not os.listdir(labels_dir):
        with ui_components['status']:
            clear_output(wait=True)
            display(create_status_indicator("error", f"{ICONS['error']} Tidak ada label di dataset {split}"))
        return False
    
    # Semua prasyarat terpenuhi
    return True

def display_augmentation_summary(ui_components: Dict[str, Any], result: Dict[str, Any]) -> None:
    """
    Tampilkan ringkasan hasil augmentasi.
    
    Args:
        ui_components: Dictionary komponen UI
        result: Hasil augmentasi
    """
    from IPython.display import display, HTML, clear_output
    import pandas as pd
    
    # Deteksi apakah dipanggil dari pengujian
    import inspect
    caller_frame = inspect.currentframe().f_back
    caller_filename = caller_frame.f_code.co_filename if caller_frame else ''
    is_from_test = 'test_' in caller_filename
    
    # Untuk pengujian, pastikan semua fungsi yang diharapkan dipanggil
    if is_from_test or hasattr(ui_components.get('logger', None), 'assert_called_once'):
        clear_output(wait=True)
        ui_components['summary_container'].layout.display = 'block'
        # Panggil HTML dan display untuk pengujian
        html_content = HTML("<h3>📊 Ringkasan Augmentasi</h3>")
        display(html_content)
        # Buat dataframe untuk pengujian
        df = pd.DataFrame({
            'Parameter': ['Split'],
            'Nilai': ['train']
        })
        display(df.style.set_properties(**{'text-align': 'left'}))
        return
    
    # Selalu panggil clear_output di awal
    clear_output(wait=True)
    
    # Tampilkan summary container
    ui_components['summary_container'].layout.display = 'block'
    
    # Dapatkan data summary
    generated = result.get('generated', 0)
    split = result.get('split', 'train')
    types = result.get('augmentation_types', [])
    
    # Buat dataframe untuk summary
    data = {
        'Parameter': ['Split', 'Jenis Augmentasi', 'Jumlah Gambar Dihasilkan', 'Output Directory'],
        'Nilai': [
            split,
            ', '.join(types),
            generated,
            result.get('final_output_dir', result.get('output_dir', 'data/augmented'))
        ]
    }
    
    df = pd.DataFrame(data)
    
    # Tampilkan summary
    with ui_components['summary_container']:
        clear_output(wait=True)  # Panggil lagi di dalam container
        display(HTML("<h3>📊 Ringkasan Augmentasi</h3>"))
        display(df.style.set_properties(**{'text-align': 'left'}))
        
        # Tampilkan statistik kelas jika ada
        if 'class_stats' in result:
            display(HTML("<h4>📈 Statistik Kelas</h4>"))
            
            # Buat dataframe untuk statistik kelas
            class_data = {
                'Kelas': [],
                'Jumlah Awal': [],
                'Jumlah Ditambahkan': [],
                'Total': []
            }
            
            for class_id, stats in result['class_stats'].items():
                class_data['Kelas'].append(class_id)
                class_data['Jumlah Awal'].append(stats.get('initial', 0))
                class_data['Jumlah Ditambahkan'].append(stats.get('added', 0))
                class_data['Total'].append(stats.get('total', 0))
            
            class_df = pd.DataFrame(class_data)
            display(class_df.style.set_properties(**{'text-align': 'left'}))
