"""
File: smartcash/ui/dataset/augmentation/utils/validation_utils.py
Deskripsi: Utilitas validasi untuk augmentasi dataset dengan ekstraksi parameter yang diperbaiki
"""

import os
from typing import Dict, Any, List, Tuple, Optional
from smartcash.ui.dataset.augmentation.utils.logger_helper import log_message

def validate_augmentation_parameters(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validasi parameter augmentasi dari UI dan return dict dengan params yang valid.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dict dengan 'valid' (bool), 'message' (str), dan 'params' (dict) jika valid
    """
    try:
        # Extract parameters dari UI dengan multiple fallback
        params = {}
        
        # Validasi jenis augmentasi - cek beberapa kemungkinan nama
        aug_types = None
        for key in ['augmentation_types', 'aug_types', 'types']:
            if key in ui_components and hasattr(ui_components[key], 'value'):
                aug_types = ui_components[key].value
                break
        
        # Jika masih None, cari dalam nested components
        if aug_types is None:
            aug_types = _find_augmentation_types_in_components(ui_components)
        
        if not aug_types:
            log_message(ui_components, "🔍 Debug: Mencari augmentation_types dalam UI components", "debug")
            for key, value in ui_components.items():
                if hasattr(value, 'value') and isinstance(value.value, (list, tuple)):
                    log_message(ui_components, f"🔍 Found list/tuple in {key}: {value.value}", "debug")
            return {'valid': False, 'message': 'Jenis augmentasi tidak ditemukan dalam UI'}
        
        params['types'] = list(aug_types) if isinstance(aug_types, (list, tuple)) else [aug_types]
        log_message(ui_components, f"✅ Jenis augmentasi ditemukan: {params['types']}", "debug")
        
        # Validasi num_variations dengan fallback
        num_variations = _get_widget_value(ui_components, ['num_variations', 'variations', 'jumlah_variasi'], 2)
        if num_variations <= 0:
            return {'valid': False, 'message': 'Jumlah variasi harus lebih dari 0'}
        params['num_variations'] = num_variations
        
        # Validasi target_count dengan fallback
        target_count = _get_widget_value(ui_components, ['target_count', 'count', 'target'], 1000)
        if target_count <= 0:
            return {'valid': False, 'message': 'Target count harus lebih dari 0'}
        params['target_count'] = target_count
        
        # Validasi output_prefix dengan fallback
        output_prefix = _get_widget_value(ui_components, ['output_prefix', 'prefix'], 'aug')
        if not output_prefix or not str(output_prefix).strip():
            return {'valid': False, 'message': 'Output prefix tidak boleh kosong'}
        params['output_prefix'] = str(output_prefix).strip()
        
        # Validasi split target dengan fallback
        split_target = _get_widget_value(ui_components, ['target_split', 'split_target', 'split'], 'train')
        if split_target not in ['train', 'valid', 'test']:
            return {'valid': False, 'message': f'Split target tidak valid: {split_target}'}
        params['split_target'] = split_target
        
        # Boolean options dengan fallback
        params['balance_classes'] = _get_widget_value(ui_components, ['balance_classes'], False)
        params['validate_results'] = _get_widget_value(ui_components, ['validate_results'], True)
        params['process_bboxes'] = True  # Always true untuk augmentasi
        
        # Validasi dataset availability
        data_dir = ui_components.get('data_dir', 'data')
        dataset_valid, dataset_msg = validate_dataset_availability(ui_components, split_target, data_dir)
        if not dataset_valid:
            return {'valid': False, 'message': dataset_msg}
        
        # Validasi disk space
        output_dir = ui_components.get('augmented_dir', 'data/augmented')
        disk_valid, disk_msg = validate_disk_space(ui_components, output_dir, 1000)
        if not disk_valid:
            log_message(ui_components, f"⚠️ {disk_msg}", "warning")
        
        # Set additional params
        params['output_dir'] = output_dir
        params['data_dir'] = data_dir
        params['num_workers'] = 4  # Default workers
        
        log_message(ui_components, f"✅ Parameter validasi berhasil: {params}", "debug")
        
        return {
            'valid': True,
            'message': f'Parameter valid untuk augmentasi {split_target}',
            'params': params
        }
        
    except Exception as e:
        error_msg = f"Error validasi parameter: {str(e)}"
        log_message(ui_components, error_msg, "error", "❌")
        return {'valid': False, 'message': error_msg}

def _find_augmentation_types_in_components(ui_components: Dict[str, Any]) -> Any:
    """
    Cari augmentation_types dalam nested components.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Value dari augmentation_types jika ditemukan
    """
    # Cari dalam basic_options, advanced_options, atau augmentation_types container
    containers = ['basic_options', 'advanced_options', 'augmentation_types']
    
    for container_name in containers:
        if container_name in ui_components:
            container = ui_components[container_name]
            result = _search_widget_in_container(container, ['augmentation_types', 'types', 'jenis'])
            if result is not None:
                return result
    
    return None

def _search_widget_in_container(container, target_descriptions: List[str]) -> Any:
    """
    Search widget dalam container berdasarkan description.
    
    Args:
        container: Widget container
        target_descriptions: List description yang dicari
        
    Returns:
        Value widget jika ditemukan
    """
    if not hasattr(container, 'children'):
        return None
    
    for child in container.children:
        # Cek description
        if hasattr(child, 'description') and hasattr(child, 'value'):
            desc = child.description.lower().replace(':', '').replace(' ', '_')
            if any(target in desc for target in target_descriptions):
                return child.value
        
        # Cek tipe widget khusus
        if hasattr(child, 'value'):
            import ipywidgets as widgets
            if isinstance(child, widgets.SelectMultiple):
                return child.value
        
        # Rekursif search
        result = _search_widget_in_container(child, target_descriptions)
        if result is not None:
            return result
    
    return None

def _get_widget_value(ui_components: Dict[str, Any], possible_keys: List[str], default_value: Any) -> Any:
    """
    Dapatkan nilai widget dengan multiple fallback keys.
    
    Args:
        ui_components: Dictionary komponen UI
        possible_keys: List kemungkinan key
        default_value: Nilai default
        
    Returns:
        Nilai widget atau default
    """
    for key in possible_keys:
        if key in ui_components and hasattr(ui_components[key], 'value'):
            return ui_components[key].value
    
    return default_value

def validate_augmentation_config(ui_components: Dict[str, Any], config: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validasi konfigurasi augmentasi.
    
    Args:
        ui_components: Dictionary komponen UI
        config: Konfigurasi augmentasi
        
    Returns:
        Tuple (is_valid, error_message)
    """
    # Validasi augmentasi diaktifkan
    if not config.get('enabled', True):
        return False, "Augmentasi tidak diaktifkan"
    
    # Validasi jenis augmentasi
    aug_types = config.get('types', [])
    if not aug_types or not isinstance(aug_types, (list, tuple)):
        return False, "Jenis augmentasi tidak valid"
    
    valid_types = ['combined', 'position', 'lighting']
    invalid_types = [t for t in aug_types if t not in valid_types]
    if invalid_types:
        return False, f"Jenis augmentasi tidak valid: {', '.join(invalid_types)}"
    
    # Validasi num_variations
    num_variations = config.get('num_variations', 0)
    if not isinstance(num_variations, int) or num_variations <= 0:
        return False, "Jumlah variasi harus berupa integer positif"
    
    if num_variations > 10:
        return False, "Jumlah variasi maksimal adalah 10"
    
    # Validasi target_count
    target_count = config.get('target_count', 0)
    if not isinstance(target_count, int) or target_count <= 0:
        return False, "Target count harus berupa integer positif"
    
    if target_count > 10000:
        return False, "Target count maksimal adalah 10000"
    
    # Validasi output_prefix
    output_prefix = config.get('output_prefix', '')
    if not output_prefix or not isinstance(output_prefix, str):
        return False, "Output prefix tidak boleh kosong"
    
    # Validasi karakter output_prefix
    if not output_prefix.replace('_', '').replace('-', '').isalnum():
        return False, "Output prefix hanya boleh berisi huruf, angka, underscore, dan dash"
    
    # Validasi output_dir
    output_dir = config.get('output_dir', '')
    if not output_dir:
        return False, "Output directory tidak boleh kosong"
    
    return True, ""

def validate_dataset_availability(ui_components: Dict[str, Any], split: str, data_dir: str = 'data') -> Tuple[bool, str]:
    """
    Validasi ketersediaan dataset untuk augmentasi.
    
    Args:
        ui_components: Dictionary komponen UI
        split: Split dataset (train, valid, test)
        data_dir: Direktori data
        
    Returns:
        Tuple (is_available, error_message)
    """
    # Validasi split
    if split not in ['train', 'valid', 'test']:
        return False, f"Split '{split}' tidak valid. Harus salah satu dari: train, valid, test"
    
    # Path ke direktori split
    split_dir = os.path.join(data_dir, 'preprocessed', split)
    
    # Cek direktori split
    if not os.path.exists(split_dir):
        return False, f"Direktori dataset {split} tidak ditemukan: {split_dir}"
    
    # Cek direktori images
    images_dir = os.path.join(split_dir, 'images')
    if not os.path.exists(images_dir):
        return False, f"Direktori images tidak ditemukan: {images_dir}"
    
    # Cek direktori labels
    labels_dir = os.path.join(split_dir, 'labels')
    if not os.path.exists(labels_dir):
        return False, f"Direktori labels tidak ditemukan: {labels_dir}"
    
    # Hitung jumlah file
    try:
        image_files = [f for f in os.listdir(images_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
        label_files = [f for f in os.listdir(labels_dir) 
                      if f.lower().endswith('.txt')]
        
        if not image_files:
            return False, f"Tidak ada file gambar di {images_dir}"
        
        if not label_files:
            return False, f"Tidak ada file label di {labels_dir}"
        
        # Cek rasio image:label
        if len(image_files) != len(label_files):
            log_message(ui_components, 
                       f"Jumlah gambar ({len(image_files)}) tidak sama dengan label ({len(label_files)})", 
                       "warning", "⚠️")
        
        return True, f"Dataset {split} siap: {len(image_files)} gambar, {len(label_files)} label"
        
    except Exception as e:
        return False, f"Error saat memeriksa dataset: {str(e)}"

def validate_disk_space(ui_components: Dict[str, Any], output_dir: str, estimated_size_mb: int = 1000) -> Tuple[bool, str]:
    """
    Validasi ruang disk yang tersedia.
    
    Args:
        ui_components: Dictionary komponen UI
        output_dir: Direktori output
        estimated_size_mb: Estimasi ukuran output dalam MB
        
    Returns:
        Tuple (has_enough_space, message)
    """
    try:
        import shutil
        
        # Dapatkan ruang disk yang tersedia
        free_space_bytes = shutil.disk_usage(output_dir).free
        free_space_mb = free_space_bytes / (1024 * 1024)
        
        # Tambahkan buffer 20%
        required_space_mb = estimated_size_mb * 1.2
        
        if free_space_mb < required_space_mb:
            return False, f"Ruang disk tidak cukup. Tersedia: {free_space_mb:.0f}MB, Dibutuhkan: {required_space_mb:.0f}MB"
        
        return True, f"Ruang disk cukup: {free_space_mb:.0f}MB tersedia"
        
    except Exception as e:
        # Jika tidak bisa cek disk space, anggap OK
        log_message(ui_components, f"Tidak bisa cek ruang disk: {str(e)}", "warning", "⚠️")
        return True, "Status ruang disk tidak dapat dipastikan"

def validate_ui_components(ui_components: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validasi komponen UI yang diperlukan.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Tuple (is_valid, missing_components)
    """
    required_components = [
        'progress_bar', 'progress_container', 'log_output', 
        'status_panel', 'confirmation_area'
    ]
    
    missing_components = []
    
    for component in required_components:
        if component not in ui_components:
            missing_components.append(component)
    
    return len(missing_components) == 0, missing_components

def get_validation_summary(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Dapatkan ringkasan validasi lengkap.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary ringkasan validasi
    """
    summary = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'info': []
    }
    
    # Validasi parameter
    param_result = validate_augmentation_parameters(ui_components)
    if not param_result['valid']:
        summary['is_valid'] = False
        summary['errors'].append(f"Parameter: {param_result['message']}")
    
    # Validasi komponen UI
    ui_valid, missing_components = validate_ui_components(ui_components)
    if not ui_valid:
        summary['warnings'].append(f"Komponen UI hilang: {', '.join(missing_components)}")
    
    # Validasi ruang disk
    output_dir = ui_components.get('augmented_dir', 'data/augmented')
    disk_valid, disk_msg = validate_disk_space(ui_components, output_dir)
    if not disk_valid:
        summary['warnings'].append(f"Ruang disk: {disk_msg}")
    else:
        summary['info'].append(disk_msg)
    
    return summary