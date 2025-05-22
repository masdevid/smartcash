"""
File: smartcash/ui/dataset/augmentation/handlers/augmentation_executor.py
Deskripsi: Executor untuk menjalankan proses augmentasi dengan symlink yang benar
"""

import os
import time
from typing import Dict, Any, Callable, Optional, List
from tqdm.auto import tqdm

from smartcash.common.logger import get_logger
from smartcash.common.constants.paths import COLAB_PATH, DRIVE_PATH
from smartcash.ui.dataset.augmentation.utils.logger_helper import log_message

class AugmentationExecutor:
    """Executor untuk proses augmentasi dengan path symlink yang benar."""
    
    def __init__(self, ui_components: Dict[str, Any]):
        """
        Inisialisasi executor.
        
        Args:
            ui_components: Dictionary komponen UI
        """
        self.ui_components = ui_components
        self.logger = ui_components.get('logger', get_logger())
        self.progress_callback: Optional[Callable] = None
        self.stop_requested = False
        
        # Deteksi environment
        self.is_colab = self._detect_colab_environment()
        
    def _detect_colab_environment(self) -> bool:
        """Deteksi apakah berjalan di Colab."""
        try:
            import google.colab
            return True
        except ImportError:
            return False
            
    def set_progress_callback(self, callback: Callable) -> None:
        """Set callback untuk progress updates."""
        self.progress_callback = callback
        
    def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Eksekusi augmentasi dengan parameter yang diberikan.
        
        Args:
            params: Parameter augmentasi
            
        Returns:
            Dictionary hasil augmentasi
        """
        try:
            # Setup direktori dengan symlink yang benar
            setup_result = self._setup_directories_with_symlink(params)
            if not setup_result['success']:
                return {'status': 'error', 'message': setup_result['message']}
            
            # Load dataset
            dataset_info = self._load_dataset_info(params)
            if not dataset_info['success']:
                return {'status': 'error', 'message': dataset_info['message']}
            
            # Jalankan augmentasi
            return self._run_augmentation(params, dataset_info['data'])
            
        except Exception as e:
            self.logger.error(f"🔥 Error executor: {str(e)}")
            return {'status': 'error', 'message': f'Error executor: {str(e)}'}
    
    def _setup_directories_with_symlink(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Setup direktori output menggunakan struktur symlink yang benar."""
        try:
            split = params.get('split_target', 'train')
            
            if self.is_colab:
                # Di Colab: gunakan path symlink yang benar
                # /content/data/augmented/split (otomatis ke Drive via symlink)
                data_base = f"{COLAB_PATH}/data"  # /content/data (symlink ke Drive)
                output_dir = os.path.join(data_base, 'augmented', split)
                
                # Cek apakah symlink aktif
                data_symlink_active = os.path.islink(data_base) and os.path.exists(data_base)
                
                if data_symlink_active:
                    actual_target = os.path.realpath(data_base)
                    log_message(self.ui_components, f"🔗 Data symlink aktif: {data_base} -> {actual_target}", "info")
                    storage_type = "Google Drive (via symlink)"
                else:
                    log_message(self.ui_components, f"⚠️ Data symlink tidak aktif, menggunakan direktori lokal", "warning")
                    storage_type = "Local (symlink belum setup)"
                
                params['storage_type'] = storage_type
                params['uses_symlink'] = data_symlink_active
                
            else:
                # Local development
                data_dir = self.ui_components.get('data_dir', 'data')
                output_dir = os.path.join(data_dir, 'augmented', split)
                params['storage_type'] = "Local"
                params['uses_symlink'] = False
            
            # Buat direktori output (otomatis ke Drive jika symlink aktif)
            os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
            os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)
            
            params['output_dir'] = output_dir
            
            log_message(self.ui_components, f"📁 Output directory: {output_dir}", "info")
            log_message(self.ui_components, f"📍 Storage: {params['storage_type']}", "info")
            
            return {'success': True}
            
        except Exception as e:
            return {'success': False, 'message': f'Error setup direktori: {str(e)}'}
    
    def _load_dataset_info(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Load informasi dataset untuk augmentasi."""
        try:
            split = params.get('split_target', 'train')
            
            if self.is_colab:
                # Di Colab: gunakan path symlink
                data_base = f"{COLAB_PATH}/data"  # /content/data (symlink)
                dataset_path = os.path.join(data_base, 'preprocessed', split)
            else:
                # Local development
                data_dir = self.ui_components.get('data_dir', 'data')
                dataset_path = os.path.join(data_dir, 'preprocessed', split)
            
            images_path = os.path.join(dataset_path, 'images')
            labels_path = os.path.join(dataset_path, 'labels')
            
            # Load daftar file
            if not os.path.exists(images_path):
                return {'success': False, 'message': f'Direktori images tidak ditemukan: {images_path}'}
            if not os.path.exists(labels_path):
                return {'success': False, 'message': f'Direktori labels tidak ditemukan: {labels_path}'}
            
            image_files = [f for f in os.listdir(images_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            label_files = [f for f in os.listdir(labels_path) 
                          if f.lower().endswith('.txt')]
            
            # Match gambar dengan label
            matched_files = []
            for img_file in image_files:
                base_name = os.path.splitext(img_file)[0]
                label_file = f"{base_name}.txt"
                if label_file in label_files:
                    matched_files.append({
                        'image': os.path.join(images_path, img_file),
                        'label': os.path.join(labels_path, label_file),
                        'base_name': base_name
                    })
            
            if not matched_files:
                return {'success': False, 'message': 'Tidak ada pasangan gambar-label yang valid'}
            
            log_message(self.ui_components, f"📊 Dataset loaded: {len(matched_files)} pasangan file", "info")
            
            return {
                'success': True,
                'data': {
                    'files': matched_files,
                    'total_files': len(matched_files),
                    'images_path': images_path,
                    'labels_path': labels_path,
                    'output_dir': params['output_dir']
                }
            }
            
        except Exception as e:
            return {'success': False, 'message': f'Error load dataset: {str(e)}'}
    
    def _run_augmentation(self, params: Dict[str, Any], dataset_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Jalankan proses augmentasi dengan output melalui symlink.
        
        Args:
            params: Parameter augmentasi
            dataset_info: Informasi dataset
            
        Returns:
            Dictionary hasil augmentasi
        """
        files = dataset_info['files']
        num_variations = params.get('num_variations', 2)
        total_tasks = len(files) * num_variations
        
        # Progress tracking
        generated_count = 0
        processed_count = 0
        
        output_dir = params['output_dir']
        storage_type = params.get('storage_type', 'Unknown')
        uses_symlink = params.get('uses_symlink', False)
        
        log_message(self.ui_components, f"🚀 Memulai augmentasi {total_tasks} tasks", "info")
        log_message(self.ui_components, f"📍 Storage: {storage_type}", "info")
        
        # Gunakan tqdm untuk progress bar
        with tqdm(total=total_tasks, desc="🔄 Augmentasi", unit="gambar", colour="green") as pbar:
            
            # Process secara sequential (tanpa multiprocessing untuk Colab)
            for file_info in files:
                for variation in range(num_variations):
                    
                    # Cek stop request
                    if self.ui_components.get('stop_requested', False):
                        break
                    
                    try:
                        # Augmentasi single image (output otomatis ke Drive jika symlink aktif)
                        result = self._augment_single_image(
                            file_info, params, variation, output_dir
                        )
                        
                        if result['success']:
                            generated_count += 1
                        
                        processed_count += 1
                        pbar.update(1)
                        
                        # Update progress callback
                        if self.progress_callback:
                            continue_processing = self.progress_callback(
                                processed_count, total_tasks, 
                                f"Generated: {generated_count}/{total_tasks}"
                            )
                            if not continue_processing:
                                break
                        
                    except Exception as e:
                        self.logger.error(f"🔥 Error task: {str(e)}")
                        pbar.update(1)
                        processed_count += 1
                
                # Break outer loop jika stop requested
                if self.ui_components.get('stop_requested', False):
                    break
        
        # Hasil akhir
        if self.ui_components.get('stop_requested', False):
            return {
                'status': 'cancelled',
                'message': 'Augmentasi dibatalkan',
                'generated_images': generated_count,
                'processed': processed_count,
                'storage_type': storage_type
            }
        
        success_rate = (generated_count / total_tasks) * 100 if total_tasks > 0 else 0
        
        # Message dengan info storage
        result_msg = f'Augmentasi selesai: {generated_count}/{total_tasks} ({success_rate:.1f}%)'
        if uses_symlink:
            result_msg += f' | Disimpan ke Google Drive via symlink'
        
        return {
            'status': 'success' if success_rate > 80 else 'warning',
            'message': result_msg,
            'generated_images': generated_count,
            'processed': processed_count,
            'success_rate': success_rate,
            'output_dir': output_dir,
            'storage_type': storage_type,
            'uses_symlink': uses_symlink
        }
    
    def _augment_single_image(self, file_info: Dict[str, Any], params: Dict[str, Any], 
                             variation: int, output_dir: str) -> Dict[str, Any]:
        """
        Augmentasi single image dengan output melalui symlink.
        
        Args:
            file_info: Info file (image, label, base_name)
            params: Parameter augmentasi
            variation: Nomor variasi (0, 1, 2, ...)
            output_dir: Direktori output (otomatis ke Drive jika symlink aktif)
            
        Returns:
            Dictionary hasil augmentasi
        """
        try:
            # Import augmentation service
            from smartcash.dataset.services.augmentor.augmentation_service import AugmentationService
            
            # Buat service instance
            service = AugmentationService()
            
            # Generate augmented image (output otomatis ke Drive jika symlink aktif)
            result = service.augment_single_file(
                image_path=file_info['image'],
                label_path=file_info['label'],
                output_dir=output_dir,
                types=params.get('types', ['combined']),
                variation_id=variation,
                prefix=params.get('output_prefix', 'aug')
            )
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'file': file_info['base_name']
            }
    
    def stop(self) -> None:
        """Hentikan proses augmentasi."""
        self.stop_requested = True
        self.ui_components['stop_requested'] = True
        log_message(self.ui_components, "⏹️ Permintaan stop diterima", "warning")

# Fungsi utility untuk compatibility dengan handler lama
def run_augmentation(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Jalankan augmentasi dengan executor.
    Compatibility function untuk handler yang sudah ada.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary hasil augmentasi
    """
    from smartcash.ui.dataset.augmentation.handlers.augmentation_handler import get_augmentation_config_from_ui
    
    # Dapatkan konfigurasi dari UI
    config = get_augmentation_config_from_ui(ui_components)
    
    # Buat executor
    executor = AugmentationExecutor(ui_components)
    
    # Jalankan augmentasi
    return executor.execute(config)

def process_augmentation_result(ui_components: Dict[str, Any], result: Dict[str, Any]) -> None:
    """
    Process hasil augmentasi dan update UI.
    Compatibility function untuk handler yang sudah ada.
    
    Args:
        ui_components: Dictionary komponen UI
        result: Hasil augmentasi
    """
    from smartcash.ui.dataset.augmentation.handlers.augmentation_handler import _handle_augmentation_result
    _handle_augmentation_result(ui_components, result)