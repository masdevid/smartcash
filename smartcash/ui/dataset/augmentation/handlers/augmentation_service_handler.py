"""
File: smartcash/ui/dataset/augmentation/handlers/augmentation_service_handler.py
Deskripsi: Handler service untuk augmentasi dataset
"""

import os
import time
from typing import Dict, Any, List, Optional
from smartcash.common.logger import get_logger
from concurrent.futures import ThreadPoolExecutor
from tqdm.auto import tqdm

def get_augmentation_service(ui_components: Dict[str, Any], config: Dict[str, Any] = None, logger=None):
    """
    Dapatkan augmentation service.
    
    Args:
        ui_components: Dictionary komponen UI
        config: Konfigurasi aplikasi
        logger: Logger untuk logging
        
    Returns:
        Augmentation service instance
    """
    if logger is None:
        logger = get_logger('augmentation')
    
    # Cek apakah sudah ada di ui_components
    if 'augmentation_service' in ui_components and ui_components['augmentation_service']:
        return ui_components['augmentation_service']
    
    try:
        # Import service
        from smartcash.dataset.services.augmentor.augmentation_service import AugmentationService
        
        # Dapatkan konfigurasi
        if config is None:
            from smartcash.ui.dataset.augmentation.handlers.config_handler import get_config_from_ui
            config = get_config_from_ui(ui_components)
        
        # Buat instance service dengan ui_components untuk NotificationManager
        service = AugmentationService(
            config=config,
            data_dir=ui_components.get('data_dir', 'data'),
            logger=logger,
            num_workers=config.get('augmentation', {}).get('num_workers', 4),
            ui_components=ui_components  # Teruskan ui_components untuk NotificationManager
        )
        
        # Cek apakah ada NotificationManager di ui_components
        from smartcash.ui.dataset.augmentation.utils.notification_manager import NotificationManager
        notification_manager = None
        
        # Jika tidak ada, buat instance baru
        if not hasattr(ui_components, 'notification_manager') or ui_components.get('notification_manager') is None:
            try:
                notification_manager = NotificationManager(ui_components)
                ui_components['notification_manager'] = notification_manager
            except Exception as e:
                logger.warning(f"⚠️ Tidak dapat membuat NotificationManager: {str(e)}")
        else:
            notification_manager = ui_components.get('notification_manager')
            
        # Register NotificationManager ke service
        if notification_manager:
            service.register_progress_callback(notification_manager=notification_manager)
        # Fallback ke callback lama jika ada
        elif 'progress_callback' in ui_components and callable(ui_components['progress_callback']):
            service.register_progress_callback(ui_components['progress_callback'])
        
        # Simpan ke ui_components
        ui_components['augmentation_service'] = service
        
        return service
    except Exception as e:
        logger.error(f"❌ Error saat membuat augmentation service: {str(e)}")
        return None

def execute_augmentation(ui_components: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Eksekusi augmentasi dataset.
    
    Args:
        ui_components: Dictionary komponen UI
        params: Parameter augmentasi
        
    Returns:
        Dictionary hasil augmentasi
    """
    logger = ui_components.get('logger', get_logger('augmentation'))
    
    try:
        # Dapatkan service
        service = get_augmentation_service(ui_components)
        
        if not service:
            return {
                'status': 'error',
                'message': 'Gagal membuat augmentation service'
            }
        
        # Dapatkan parameter
        split = params.get('split', 'train')
        augmentation_types = params.get('augmentation_types', ['combined'])
        num_variations = params.get('num_variations', 2)
        output_prefix = params.get('output_prefix', 'aug')
        validate_results = params.get('validate_results', True)
        resume = params.get('resume', False)
        process_bboxes = params.get('process_bboxes', True)
        target_balance = params.get('target_balance', True)
        num_workers = params.get('num_workers', 4)
        move_to_preprocessed = params.get('move_to_preprocessed', True)
        target_count = params.get('target_count', 1000)
        
        # Jalankan augmentasi
        logger.info(f"🚀 Memulai augmentasi dataset {split} dengan jenis: {', '.join(augmentation_types)}")
        
        # Cek apakah stop diminta
        if ui_components.get('stop_requested', False):
            return {
                'status': 'warning',
                'message': 'Augmentasi dibatalkan oleh pengguna'
            }
        
        # Jalankan augmentasi
        result = service.augment_dataset(
            split=split,
            augmentation_types=augmentation_types,
            num_variations=num_variations,
            output_prefix=output_prefix,
            validate_results=validate_results,
            resume=resume,
            process_bboxes=process_bboxes,
            target_balance=target_balance,
            num_workers=num_workers,
            move_to_preprocessed=move_to_preprocessed,
            target_count=target_count
        )
        
        # Tambahkan parameter ke hasil
        result['split'] = split
        result['augmentation_types'] = augmentation_types
        
        return result
    except Exception as e:
        logger.error(f"❌ Error saat menjalankan augmentasi: {str(e)}")
        
        return {
            'status': 'error',
            'message': f'Error saat menjalankan augmentasi: {str(e)}'
        }

def execute_augmentation_with_tracking(ui_components: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Eksekusi augmentasi dataset dengan tracking progress yang lebih detail.
    
    Args:
        ui_components: Dictionary komponen UI
        params: Parameter augmentasi
        
    Returns:
        Dictionary hasil augmentasi
    """
    # Pastikan logger tersedia untuk debugging
    logger = ui_components.get('logger', get_logger('augmentation'))
    logger.info("🔍 Memulai augmentasi dengan tracking detail")
    
    try:
        # Dapatkan service
        service = get_augmentation_service(ui_components)
        
        if not service:
            return {
                'status': 'error',
                'message': 'Gagal membuat augmentation service'
            }
        
        # Dapatkan parameter
        split = params.get('split', 'train')
        augmentation_types = params.get('augmentation_types', ['combined'])
        num_variations = params.get('num_variations', 2)
        output_prefix = params.get('output_prefix', 'aug')
        validate_results = params.get('validate_results', True)
        resume = params.get('resume', False)
        process_bboxes = params.get('process_bboxes', True)
        target_balance = params.get('target_balance', True)
        num_workers = params.get('num_workers', 4)
        move_to_preprocessed = params.get('move_to_preprocessed', True)
        target_count = params.get('target_count', 1000)
        
        # Jalankan augmentasi
        logger.info(f"🚀 Memulai augmentasi dataset {split} dengan jenis: {', '.join(augmentation_types)}")
        
        # Pastikan progress callback diregistrasi dengan benar
        from smartcash.ui.dataset.augmentation.handlers.initialization_handler import register_progress_callback
        register_progress_callback(ui_components)
        
        # Pastikan semua komponen progress tracking terlihat dan diinisialisasi dengan benar
        if 'progress_bar' in ui_components:
            ui_components['progress_bar'].layout.visibility = 'visible'
            ui_components['progress_bar'].value = 0
            ui_components['progress_bar'].max = 100
        
        if 'current_progress' in ui_components:
            ui_components['current_progress'].layout.visibility = 'visible'
            ui_components['current_progress'].value = 0
            ui_components['current_progress'].max = 100
        
        if 'overall_label' in ui_components:
            ui_components['overall_label'].layout.visibility = 'visible'
            ui_components['overall_label'].value = f"Memulai augmentasi dataset {split}..."
        
        if 'step_label' in ui_components:
            ui_components['step_label'].layout.visibility = 'visible'
            ui_components['step_label'].value = "Menganalisis dataset..."
        
        # Cek apakah stop diminta
        if ui_components.get('stop_requested', False):
            return {
                'status': 'warning',
                'message': 'Augmentasi dibatalkan oleh pengguna'
            }
        
        # Tambahkan callback untuk update UI dari thread terpisah
        def progress_update_callback(progress=None, total=None, message=None, status='info', **kwargs):
            # Log pemanggilan callback untuk debugging
            logger.debug(f"🔍 Progress callback dipanggil: progress={progress}, total={total}, message={message}")
            
            try:
                # Update progress bar
                if progress is not None and total is not None and 'progress_bar' in ui_components:
                    ui_components['progress_bar'].max = total
                    ui_components['progress_bar'].value = progress
                    logger.debug(f"📊 Progress bar diupdate langsung: {progress}/{total}")
                
                # Update current progress jika ada
                if 'current_progress' in kwargs and 'current_total' in kwargs and 'current_progress' in ui_components:
                    ui_components['current_progress'].max = kwargs['current_total']
                    ui_components['current_progress'].value = kwargs['current_progress']
                
                # Update label
                if message and 'overall_label' in ui_components:
                    ui_components['overall_label'].value = message
            except Exception as e:
                logger.warning(f"⚠️ Error saat mengupdate UI: {str(e)}")
            
            # Log progress
            if message:
                if status == 'error':
                    logger.error(message)
                elif status == 'warning':
                    logger.warning(message)
                else:
                    logger.info(message)
            
            # Cek apakah stop diminta
            if ui_components.get('stop_requested', False):
                return False
            
            # Lanjutkan proses
            return True
        
        # Register callback tambahan untuk memastikan UI diupdate
        service.register_progress_callback(progress_update_callback)
        
        # Log awal proses
        logger.info(f"⏳ Memulai proses augmentasi dengan {num_workers} worker...")
        
        # Jalankan augmentasi
        result = service.augment_dataset(
            split=split,
            augmentation_types=augmentation_types,
            num_variations=num_variations,
            output_prefix=output_prefix,
            validate_results=validate_results,
            resume=resume,
            process_bboxes=process_bboxes,
            target_balance=target_balance,
            num_workers=num_workers,
            move_to_preprocessed=move_to_preprocessed,
            target_count=target_count
        )
        
        # Tambahkan parameter ke hasil
        result['split'] = split
        result['augmentation_types'] = augmentation_types
        
        # Log hasil akhir
        if result.get('status') == 'success':
            logger.info(f"✅ Augmentasi selesai: {result.get('generated', 0)} gambar dihasilkan")
        else:
            logger.warning(f"⚠️ Augmentasi tidak selesai dengan sempurna: {result.get('message', '')}")
        
        return result
    except Exception as e:
        logger.error(f"❌ Error saat menjalankan augmentasi: {str(e)}")
        
        return {
            'status': 'error',
            'message': f'Error saat menjalankan augmentasi: {str(e)}'
        }

def run_augmentation(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Jalankan augmentasi dataset.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary hasil augmentasi
    """
    logger = ui_components.get('logger', get_logger('augmentation'))
    
    try:
        # Dapatkan konfigurasi dari UI
        from smartcash.ui.dataset.augmentation.handlers.config_handler import get_config_from_ui
        config = get_config_from_ui(ui_components)
        
        # Dapatkan parameter augmentasi
        augmentation_config = config.get('augmentation', {})
        
        # Dapatkan split yang dipilih
        split = ui_components.get('split_selector', {}).value if hasattr(ui_components.get('split_selector', {}), 'value') else 'train'
        
        # Buat parameter untuk augmentasi
        params = {
            'split': split,
            'augmentation_types': augmentation_config.get('types', ['combined']),
            'num_variations': augmentation_config.get('num_variations', 2),
            'output_prefix': 'aug',
            'validate_results': True,
            'resume': False,
            'process_bboxes': True,
            'target_balance': augmentation_config.get('target_balance', True),
            'num_workers': augmentation_config.get('num_workers', 4),
            'move_to_preprocessed': False,  # Akan dipindahkan secara manual nanti
            'target_count': augmentation_config.get('target_count', 1000)
        }
        
        # Jalankan augmentasi dengan tracking
        return execute_augmentation_with_tracking(ui_components, params)
    except Exception as e:
        logger.error(f"❌ Error saat menjalankan augmentasi: {str(e)}")
        
        return {
            'status': 'error',
            'message': f'Error saat menjalankan augmentasi: {str(e)}'
        }

def copy_augmented_to_preprocessed(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Salin hasil augmentasi ke direktori preprocessed.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary hasil penyalinan
    """
    logger = ui_components.get('logger', get_logger('augmentation'))
    
    try:
        # Dapatkan direktori data
        data_dir = ui_components.get('data_dir', 'data')
        
        # Dapatkan split yang dipilih
        split = ui_components.get('split_selector', {}).value if hasattr(ui_components.get('split_selector', {}), 'value') else 'train'
        
        # Dapatkan direktori sumber dan tujuan
        source_dir = os.path.join(data_dir, 'augmented')
        target_dir = os.path.join(data_dir, 'preprocessed', split)
        
        # Periksa apakah direktori sumber ada
        if not os.path.exists(source_dir):
            return {
                'status': 'error',
                'message': f'Direktori augmentasi tidak ditemukan: {source_dir}',
                'error': 'DirectoryNotFound'
            }
        
        # Periksa apakah direktori tujuan ada
        if not os.path.exists(target_dir):
            return {
                'status': 'error',
                'message': f'Dataset {split} tidak ditemukan di {target_dir}',
                'error': 'DirectoryNotFound'
            }
        
        # Salin gambar dan label
        import shutil
        import glob
        
        # Dapatkan daftar gambar dan label
        image_files = glob.glob(os.path.join(source_dir, 'images', '*.jpg')) + \
                     glob.glob(os.path.join(source_dir, 'images', '*.png')) + \
                     glob.glob(os.path.join(source_dir, 'images', '*.jpeg'))
        
        label_files = glob.glob(os.path.join(source_dir, 'labels', '*.txt'))
        
        # Buat direktori tujuan jika belum ada
        os.makedirs(os.path.join(target_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(target_dir, 'labels'), exist_ok=True)
        
        # Salin gambar dan label
        num_images_copied = 0
        num_labels_copied = 0
        
        for img_file in image_files:
            img_name = os.path.basename(img_file)
            target_path = os.path.join(target_dir, 'images', img_name)
            shutil.copy2(img_file, target_path)
            num_images_copied += 1
        
        for label_file in label_files:
            label_name = os.path.basename(label_file)
            target_path = os.path.join(target_dir, 'labels', label_name)
            shutil.copy2(label_file, target_path)
            num_labels_copied += 1
        
        return {
            'status': 'success',
            'message': f'Berhasil menyalin {num_images_copied} gambar dan {num_labels_copied} label ke {target_dir}',
            'num_images': num_images_copied,
            'num_labels': num_labels_copied
        }
    except Exception as e:
        logger.error(f"❌ Error saat menyalin hasil augmentasi: {str(e)}")
        
        return {
            'status': 'error',
            'message': f'Error saat menyalin hasil augmentasi: {str(e)}',
            'error': str(e)
        }
