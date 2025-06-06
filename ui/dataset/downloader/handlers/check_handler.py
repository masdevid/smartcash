"""
File: smartcash/ui/dataset/downloader/handlers/check_handler.py
Deskripsi: Check handler untuk pengecekan existing dataset dan summary singkat
"""

from typing import Dict, Any
from pathlib import Path

def setup_check_handler(ui_components: Dict[str, Any], config: Dict[str, Any], logger) -> None:
    """Setup check button handler untuk pengecekan existing dataset"""
    check_button = ui_components.get('check_button')
    if not check_button:
        return
    
    def on_check_click(button):
        """Handle check button click dengan dataset existence check"""
        logger.info("🔍 Checking existing dataset...")
        
        # Check existing dataset
        dataset_status = _check_existing_dataset(logger)
        
        # Display summary
        _display_dataset_summary(dataset_status, logger)
    
    check_button.on_click(on_check_click)
    logger.debug("✅ Check handler siap")

def _check_existing_dataset(logger) -> Dict[str, Any]:
    """Check existing dataset di berbagai lokasi dengan file counts"""
    data_path = Path('data')
    
    # Check standard dataset structure
    dataset_dirs = ['train', 'valid', 'test']
    dataset_status = {
        'data_dir_exists': data_path.exists(),
        'splits': {},
        'downloads': _check_downloads_dir(),
        'total_files': 0,
        'has_dataset': False
    }
    
    for split in dataset_dirs:
        split_status = _check_split_directory(data_path / split)
        dataset_status['splits'][split] = split_status
        dataset_status['total_files'] += split_status['total_files']
        
        if split_status['has_data']:
            dataset_status['has_dataset'] = True
    
    return dataset_status

def _check_split_directory(split_path: Path) -> Dict[str, Any]:
    """Check single split directory dengan image dan label counts"""
    if not split_path.exists():
        return {'exists': False, 'has_data': False, 'images': 0, 'labels': 0, 'total_files': 0}
    
    images_dir = split_path / 'images'
    labels_dir = split_path / 'labels'
    
    # Count files dengan one-liner
    image_count = len(list(images_dir.glob('*'))) if images_dir.exists() else 0
    label_count = len(list(labels_dir.glob('*.txt'))) if labels_dir.exists() else 0
    
    return {
        'exists': True,
        'has_data': image_count > 0 or label_count > 0,
        'images': image_count,
        'labels': label_count,
        'total_files': image_count + label_count,
        'images_dir_exists': images_dir.exists(),
        'labels_dir_exists': labels_dir.exists()
    }

def _check_downloads_dir() -> Dict[str, Any]:
    """Check downloads directory untuk temporary files"""
    downloads_path = Path('data/downloads')
    
    if not downloads_path.exists():
        return {'exists': False, 'files': 0, 'size_mb': 0}
    
    try:
        all_files = list(downloads_path.rglob('*'))
        files_only = [f for f in all_files if f.is_file()]
        total_size = sum(f.stat().st_size for f in files_only if f.exists())
        
        return {
            'exists': True,
            'files': len(files_only),
            'size_mb': total_size / (1024 * 1024)
        }
    except Exception:
        return {'exists': True, 'files': 0, 'size_mb': 0, 'error': True}

def _display_dataset_summary(dataset_status: Dict[str, Any], logger) -> None:
    """Display dataset summary dengan clear status messages"""
    if not dataset_status['data_dir_exists']:
        logger.info("📁 Data directory belum ada - siap untuk download pertama")
        return
    
    if not dataset_status['has_dataset']:
        logger.info("📂 Data directory kosong - siap untuk download")
        return
    
    # Dataset exists - show summary
    logger.info("📊 Dataset Summary:")
    
    # Split summary dengan one-liner formatting
    for split_name, split_info in dataset_status['splits'].items():
        if split_info['has_data']:
            logger.info(f"  • {split_name.title()}: {split_info['images']} images, {split_info['labels']} labels")
        else:
            logger.info(f"  • {split_name.title()}: Kosong")
    
    # Total summary
    total_files = dataset_status['total_files']
    logger.info(f"📈 Total: {total_files} files")
    
    # Downloads directory info
    downloads = dataset_status['downloads']
    if downloads['exists'] and downloads['files'] > 0:
        logger.info(f"💾 Downloads: {downloads['files']} files ({downloads['size_mb']:.1f}MB)")
    
    # Overall status
    if dataset_status['has_dataset']:
        logger.success("✅ Dataset sudah ada - download akan menimpa data existing")
    else:
        logger.info("🆕 Siap untuk download dataset baru")

# Export
__all__ = ['setup_check_handler']