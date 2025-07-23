"""
File: smartcash/dataset/preprocessor/api/__init__.py
Deskripsi: Updated preprocessing API dengan clean interface dan FileNamingManager integration
"""

from .preprocessing_api import (
    preprocess_dataset,
    get_preprocessing_status,
    validate_dataset_structure,
    validate_filenames,
    get_preprocessing_preview
)

from .normalization_api import (
    normalize_for_yolo,
    denormalize_for_visualization,
    batch_normalize_for_yolo,
    transform_coordinates_for_yolo,
    create_normalizer,
    get_normalization_info,
    list_available_presets,
    normalize_image_file,
    denormalize_npy_file
)

from .samples_api import (
    get_samples,
    generate_sample_previews,
    get_class_samples,
    get_samples_summary
)

from .stats_api import (
    get_dataset_stats,
    get_file_stats,
    get_class_distribution_stats,
    export_stats_report
)

from .cleanup_api import (
    cleanup_preprocessing_files,
    get_cleanup_preview,
    get_cleanup_summary,
    cleanup_empty_directories
)

from .readiness_api import (
    check_service_readiness,
    check_existing_data,
    get_preprocessing_directory_info
)

# Main API exports dengan focus pada core functionality
__all__ = [
    # Core preprocessing
    'preprocess_dataset',
    'get_preprocessing_status',
    'get_preprocessing_preview',
    
    # Normalization (standalone)
    'normalize_for_yolo',
    'denormalize_for_visualization',
    'transform_coordinates_for_yolo',
    
    # Samples & Analysis
    'get_samples',
    'get_dataset_stats',
    
    # Cleanup
    'cleanup_preprocessing_files',
    'get_cleanup_preview',
    
    # Readiness checks
    'check_service_readiness',
    'check_existing_data',
    'get_preprocessing_directory_info'
]

# === Enhanced API dengan progress integration ===

def preprocess_with_progress(config=None, ui_components=None, splits=None):
    """🚀 Enhanced preprocessing dengan automatic progress integration"""
    def progress_handler(level, current, total, message):
        tracker = ui_components.get('progress_tracker') if ui_components else None
        if tracker:
            progress_percent = int((current / total) * 100) if total > 0 else 0
            if level == 'overall':
                tracker.update_overall(progress_percent, message)
            elif level == 'current':
                tracker.update_current(progress_percent, message)
    
    return preprocess_dataset(config, progress_handler, ui_components, splits)

def cleanup_with_progress(data_dir, target='preprocessed', ui_components=None, confirm=False):
    """🧹 Enhanced cleanup dengan automatic progress integration"""
    def progress_handler(level, current, total, message):
        tracker = ui_components.get('progress_tracker') if ui_components else None
        if tracker:
            progress_percent = int((current / total) * 100) if total > 0 else 0
            if level == 'overall':
                tracker.update_overall(progress_percent, message)
            elif level == 'current':
                tracker.update_current(progress_percent, message)
    
    return cleanup_preprocessing_files(
        data_dir, target, confirm=confirm, 
        progress_callback=progress_handler, ui_components=ui_components
    )

def get_comprehensive_status(config=None):
    """📊 Comprehensive status dengan all information needed"""
    status = get_preprocessing_status(config)
    if status['success']:
        # Add additional insights
        stats = get_dataset_stats(config.get('data', {}).get('dir', 'data') if config else 'data')
        if stats['success']:
            status['dataset_insights'] = {
                'total_files': stats['overview']['total_files'],
                'main_banknotes': stats['main_banknotes']['total_objects'],
                'file_types': stats['file_types'],
                'recommendations': _generate_recommendations(stats)
            }
    return status

def _generate_recommendations(stats):
    """💡 Generate processing recommendations"""
    recommendations = []
    
    # File balance check
    raw_count = stats['file_types']['raw_images']
    preprocessed_count = stats['file_types']['preprocessed_npy']
    
    if raw_count > 0 and preprocessed_count == 0:
        recommendations.append("🔄 Ready untuk preprocessing - belum ada file yang diproses")
    elif preprocessed_count > 0 and preprocessed_count < raw_count:
        recommendations.append("⚠️ Preprocessing sebagian - ada file raw yang belum diproses")
    elif preprocessed_count >= raw_count:
        recommendations.append("✅ Preprocessing lengkap - siap untuk training")
    
    # Main banknotes balance
    main_count = stats['main_banknotes']['total_objects']
    if main_count > 1000:
        recommendations.append("🎯 Dataset besar - pertimbangkan sampling untuk testing")
    elif main_count < 100:
        recommendations.append("📈 Dataset kecil - pertimbangkan augmentation")
    
    return recommendations

# === Quick utilities untuk common patterns ===

def quick_preprocess_train_valid(data_dir='data', target_size=[640, 640]):
    """⚡ Quick preprocessing untuk train+valid dengan default settings"""
    config = {
        'data': {'dir': data_dir},
        'preprocessing': {
            'target_splits': ['train', 'valid'],
            'normalization': {'target_size': target_size}
        }
    }
    return preprocess_dataset(config)

def quick_sample_preview(data_dir='data', max_samples=5):
    """⚡ Quick sample generation untuk preview"""
    return get_samples(f"{data_dir}/preprocessed", 'train', max_samples)

def quick_cleanup_all(data_dir='data', confirm=False):
    """⚡ Quick cleanup semua preprocessing files"""
    return cleanup_preprocessing_files(data_dir, 'both', confirm=confirm)

# === Batch operations ===

def batch_normalize_directory(input_dir, output_dir, preset='default', max_workers=4):
    """📦 Batch normalize semua images dalam directory"""
    from pathlib import Path
    from concurrent.futures import ThreadPoolExecutor
    from ..core.file_processor import FileProcessor
    
    fp = FileProcessor()
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Scan image files
    image_files = fp.scan_files(input_path, 'rp_')
    
    def process_single_file(img_file):
        try:
            result = normalize_image_file(img_file, None, preset)
            return result['success']
        except:
            return False
    
    # Process dengan ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_single_file, image_files))
    
    success_count = sum(results)
    return {
        'success': True,
        'total_processed': len(image_files),
        'success_count': success_count,
        'error_count': len(image_files) - success_count,
        'output_dir': str(output_path)
    }

def batch_denormalize_samples(input_dir, output_dir, max_samples_per_class=3):
    """📦 Batch generate sample previews"""
    return generate_sample_previews(
        input_dir, output_dir, 
        splits=['train', 'valid'], 
        max_per_class=max_samples_per_class
    )

# === Configuration helpers ===

def create_preprocessing_config(data_dir='data', target_size=[640, 640], splits=['train', 'valid']):
    """⚙️ Create preprocessing config dengan sane defaults"""
    return {
        'data': {
            'dir': data_dir,
            'preprocessed_dir': f'{data_dir}/preprocessed'
        },
        'preprocessing': {
            'target_splits': splits,
            'validation': {
                'enabled': False,
                'filename_pattern': True,
                'auto_fix': True
            },
            'normalization': {
                'target_size': target_size,
                'preserve_aspect_ratio': True,
                'pixel_range': [0, 1]
            }
        },
        'performance': {
            'batch_size': 32,
            'use_gpu': True,
            'threading': {
                'io_workers': 8,
                'cpu_workers': None
            }
        }
    }

def create_yolo_preset_config(preset_name='yolov5s', data_dir='data'):
    """🎯 Create config untuk specific YOLO preset"""
    from ..config.defaults import NORMALIZATION_PRESETS
    
    if preset_name not in NORMALIZATION_PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}")
    
    preset_config = NORMALIZATION_PRESETS[preset_name]
    
    return {
        'data': {'dir': data_dir},
        'preprocessing': {
            'target_splits': ['train', 'valid'],
            'normalization': preset_config
        }
    }

# === Validation helpers ===

def validate_preprocessing_readiness(data_dir='data'):
    """✅ Comprehensive readiness check"""
    checks = {
        'structure': validate_dataset_structure(data_dir),
        'status': get_preprocessing_status({'data': {'dir': data_dir}}),
        'files': get_file_stats(data_dir, 'raw')
    }
    
    all_ready = all(check.get('success', False) for check in checks.values())
    
    return {
        'ready': all_ready,
        'checks': checks,
        'message': "✅ Siap untuk preprocessing" if all_ready else "❌ Ada masalah yang perlu diperbaiki"
    }

# === Export functionality ===

def export_preprocessing_report(data_dir, output_file='preprocessing_report.json'):
    """📄 Export comprehensive preprocessing report"""
    from pathlib import Path
    import json
    
    report = {
        'readiness': validate_preprocessing_readiness(data_dir),
        'current_stats': get_dataset_stats(data_dir),
        'file_analysis': get_file_stats(data_dir, 'all'),
        'class_distribution': get_class_distribution_stats(data_dir)
    }
    
    output_path = Path(output_file)
    try:
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        return {
            'success': True,
            'output_file': str(output_path),
            'message': f"✅ Report exported ke {output_path}"
        }
    except Exception as e:
        return {
            'success': False,
            'message': f"❌ Export failed: {str(e)}"
        }