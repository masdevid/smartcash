"""
File: smartcash/ui/dataset/augmentation/handlers/augmentation_executor.py
Deskripsi: Executor untuk menjalankan proses augmentasi dengan progress tracking dan tqdm
"""

import os
import time
from typing import Dict, Any, Callable, Optional, List
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from tqdm.auto import tqdm

from smartcash.common.logger import get_logger
from smartcash.ui.dataset.augmentation.utils.logger_helper import log_message

class AugmentationExecutor:
    """Executor untuk proses augmentasi dengan progress tracking."""
    
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
            # Setup direktori dan path
            setup_result = self._setup_directories(params)
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
    
    def _setup_directories(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Setup direktori output untuk augmentasi."""
        try:
            data_dir = self.ui_components.get('data_dir', 'data')
            split = params.get('split_target', 'train')
            
            # Direktori output
            output_dir = os.path.join(data_dir, 'augmented', split)
            os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
            os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)
            
            params['output_dir'] = output_dir
            log_message(self.ui_components, f"📁 Setup direktori: {output_dir}", "info")
            
            return {'success': True}
            
        except Exception as e:
            return {'success': False, 'message': f'Error setup direktori: {str(e)}'}
    
    def _load_dataset_info(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Load informasi dataset untuk augmentasi."""
        try:
            data_dir = self.ui_components.get('data_dir', 'data')
            split = params.get('split_target', 'train')
            
            # Path dataset
            dataset_path = os.path.join(data_dir, 'preprocessed', split)
            images_path = os.path.join(dataset_path, 'images')
            labels_path = os.path.join(dataset_path, 'labels')
            
            # Load daftar file
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
            
            log_message(self.ui_components, f"📊 Dataset loaded: {len(matched_files)} file pairs", "info")
            
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
        Jalankan proses augmentasi dengan tqdm dan multiprocessing.
        
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
        
        log_message(self.ui_components, f"🚀 Memulai augmentasi {total_tasks} tasks", "info")
        
        # Gunakan tqdm untuk progress bar
        with tqdm(total=total_tasks, desc="🔄 Augmentasi", unit="gambar", colour="green") as pbar:
            
            # Gunakan ProcessPoolExecutor untuk CPU-intensive augmentation
            max_workers = min(params.get('num_workers', 4), os.cpu_count())
            
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                
                # Submit semua tasks
                future_to_file = {}
                for file_info in files:
                    for variation in range(num_variations):
                        if self.ui_components.get('stop_requested', False):
                            break
                        
                        future = executor.submit(
                            self._augment_single_image,
                            file_info, params, variation
                        )
                        future_to_file[future] = (file_info, variation)
                
                # Process hasil
                for future in as_completed(future_to_file):
                    if self.ui_components.get('stop_requested', False):
                        break
                    
                    try:
                        result = future.result()
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
        
        # Hasil akhir
        if self.ui_components.get('stop_requested', False):
            return {
                'status': 'cancelled',
                'message': 'Augmentasi dibatalkan',
                'generated_images': generated_count,
                'processed': processed_count
            }
        
        success_rate = (generated_count / total_tasks) * 100 if total_tasks > 0 else 0
        
        return {
            'status': 'success' if success_rate > 80 else 'warning',
            'message': f'Augmentasi selesai: {generated_count}/{total_tasks} ({success_rate:.1f}%)',
            'generated_images': generated_count,
            'processed': processed_count,
            'success_rate': success_rate,
            'output_dir': params['output_dir']
        }
    
    def _augment_single_image(self, file_info: Dict[str, Any], params: Dict[str, Any], variation: int) -> Dict[str, Any]:
        """
        Augmentasi single image dengan parameter tertentu.
        
        Args:
            file_info: Info file (image, label, base_name)
            params: Parameter augmentasi
            variation: Nomor variasi (0, 1, 2, ...)
            
        Returns:
            Dictionary hasil augmentasi
        """
        try:
            # Import augmentation service
            from smartcash.dataset.services.augmentor.augmentation_service import AugmentationService
            
            # Buat service instance (per-process)
            service = AugmentationService()
            
            # Generate augmented image
            result = service.augment_single_file(
                image_path=file_info['image'],
                label_path=file_info['label'],
                output_dir=params['output_dir'],
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