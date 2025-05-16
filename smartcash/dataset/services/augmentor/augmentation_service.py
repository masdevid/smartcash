"""
File: smartcash/dataset/services/augmentor/augmentation_service.py
Deskripsi: Layanan augmentasi dataset dengan prioritisasi sampel yang direfactor untuk modularitas
"""

import os, time
from typing import Dict, List, Any, Optional, Callable, Union
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm.auto import tqdm

from smartcash.common.logger import get_logger
from smartcash.dataset.services.augmentor.pipeline_factory import AugmentationPipelineFactory
from smartcash.dataset.services.augmentor.bbox_augmentor import BBoxAugmentor
from smartcash.dataset.services.augmentor.class_balancer import ClassBalancer

# Import helper modules yang dipisah
from smartcash.dataset.services.augmentor.helpers.path_helper import setup_paths
from smartcash.dataset.services.augmentor.helpers.augmentation_executor import execute_augmentation_with_tracking
from smartcash.dataset.services.augmentor.helpers.validation_helper import validate_input_files
from smartcash.dataset.utils.move_utils import move_files_to_preprocessed

class AugmentationService:
    """Layanan augmentasi dataset dengan prioritisasi kelas dan pelacakan dinamis"""
    
    def __init__(self, config: Dict = None, data_dir: str = 'data', logger=None, num_workers: int = None):
        """Inisialisasi AugmentationService dengan parameter utama."""
        self.config = config or {}
        self.data_dir = data_dir
        self.logger = logger or get_logger("augmentation_service")
        self.num_workers = num_workers if num_workers is not None else self.config.get('augmentation', {}).get('num_workers', 4)
        self.logger.info(f"🔧 Menggunakan {self.num_workers} worker untuk augmentasi")
        
        # Inisialisasi komponen-komponen utama dengan one-liner
        self.pipeline_factory = AugmentationPipelineFactory(self.config, self.logger)
        self.bbox_augmentor = BBoxAugmentor(self.config, self.logger)
        self.class_balancer = ClassBalancer(self.config, self.logger)
        
        self._stop_signal = False
        self._progress_callbacks = []
    
    def register_progress_callback(self, callback: Callable) -> None:
        """Register callback untuk progress tracking."""
        if callback and callable(callback): 
            self._progress_callbacks.append(callback)
    
    def report_progress(self, progress: int = None, total: int = None, message: str = None, status: str = 'info', **kwargs) -> None:
        """Laporkan progress dengan callback dan hindari duplikasi parameter."""
        for callback in self._progress_callbacks:
            try:
                # Buat params bersih dengan one-liner
                explicit_params = ['progress', 'total', 'message', 'status', 'current_progress', 'current_total', 'class_id']
                filtered_kwargs = {k: v for k, v in kwargs.items() if k not in explicit_params}
                
                # Gabungkan parameter yang ada nilainya dengan one-liner
                params = {k: v for k, v in {
                    'message': message, 'status': status, 
                    'progress': progress, 'total': total,
                    'current_progress': kwargs.get('current_progress'), 
                    'current_total': kwargs.get('current_total'),
                    'class_id': kwargs.get('class_id')
                }.items() if v is not None}
                
                # Gabungkan dan panggil callback
                params.update(filtered_kwargs)
                callback(**params)
            except Exception as e: 
                self.logger.warning(f"⚠️ Error pada progress callback: {str(e)}")
    
    def augment_dataset(
        self, 
        split: str = 'train', 
        augmentation_types: List[str] = None, 
        num_variations: int = 2,
        output_prefix: str = 'aug', 
        validate_results: bool = True, 
        resume: bool = False,
        process_bboxes: bool = True, 
        target_balance: bool = True, 
        num_workers: int = None,
        move_to_preprocessed: bool = True, 
        target_count: int = 1000
    ) -> Dict[str, Any]:
        """Augmentasi dataset dengan prioritisasi kelas dan pelacakan mencapai target."""
        # Reset stop signal dan setup paths
        self._stop_signal = False
        start_time = time.time()
        
        # Gunakan helper untuk setup paths
        paths = setup_paths(self.config, split)
        
        # Validasi input files dengan helper
        image_files, validation_result = validate_input_files(
            paths['images_input_dir'], 
            self.config.get('preprocessing', {}).get('file_prefix', 'rp'),
            self.logger
        )
        
        # Return error jika validasi gagal
        if not validation_result['success']:
            self.logger.warning(f"⚠️ {validation_result['message']}")
            return {"status": "error", "message": validation_result['message']}
            
        # Persiapkan balancing dengan prioritisasi kelas
        class_data = self._prepare_balancing(image_files, paths, target_count, target_balance)
        if not class_data:
            return {"status": "info", "message": "Tidak ada kelas yang memerlukan augmentasi", "generated": 0}
        
        # Gunakan helper untuk augmentasi dengan pelacakan dinamis
        augmentation_result = execute_augmentation_with_tracking(
            self, class_data, augmentation_types, num_variations, output_prefix,
            validate_results, process_bboxes, num_workers or self.num_workers, 
            paths, split, target_count, start_time
        )
        
        # Handle move files jika diperlukan
        if move_to_preprocessed and augmentation_result['status'] == 'success':
            self.report_progress(
                message=f"🔄 Memindahkan {augmentation_result['generated']} file ke direktori preprocessed",
                status="info", step=2
            )
            
            move_success = move_files_to_preprocessed(
                paths['images_output_dir'], paths['labels_output_dir'],
                output_prefix, paths['final_output_dir'], split, self.logger
            )
            
            # Update path output jika berhasil di-move
            if move_success:
                augmentation_result['final_output_dir'] = paths['final_output_dir']
        
        return augmentation_result
    
    def _prepare_balancing(self, image_files: List[str], paths: Dict, target_count: int, target_balance: bool) -> Optional[Dict[str, Any]]:
        """Persiapkan balancing kelas dengan prioritisasi optimal menggunakan ClassBalancer."""
        if not target_balance:
            self.logger.info("ℹ️ Balancing kelas dinonaktifkan, menggunakan semua file")
            # Return format compatible data untuk mode non-balancing
            return {
                'files_by_class': {},
                'selected_files': image_files,
                'class_counts': {}
            }
            
        # Laporkan progres analisis
        self.report_progress(
            message="🔍 Menganalisis distribusi kelas untuk balancing optimal",
            status="info", step=0
        )
        
        try:
            # Gunakan class_balancer dengan prioritisasi kelas
            result = self.class_balancer.prepare_balanced_dataset(
                image_files=image_files,
                labels_dir=paths['labels_input_dir'],
                target_count=target_count,
                progress_callback=lambda *args, **kwargs: self.report_progress(*args, **kwargs),
                filter_unknown=True
            )
            
            # Cek hasil balancing
            total_files = len(result.get('selected_files', []))
            classes_needing = result.get('classes_to_augment', 0)
            total_needed = result.get('total_needed', 0)
            
            # Log distribusi kelas saat ini untuk debugging
            self.logger.info(f"📊 Distribusi kelas saat ini: {result.get('class_counts', {})}")
            self.logger.info(f"📊 Hasil balancing: {classes_needing} kelas perlu ditambah {total_needed} instance")
            self.logger.info(f"🎯 Menggunakan {total_files} file untuk augmentasi")
            
            # Cek apakah ada file yang terpilih
            if total_files == 0:
                self.logger.info("⚠️ Tidak ada file yang terpilih untuk augmentasi (semua kelas sudah memenuhi target)")
                
                # Jika tidak ada file yang terpilih, gunakan beberapa file untuk memaksa augmentasi
                # Ini memastikan bahwa augmentasi tetap dilakukan meskipun target sudah terpenuhi
                self.logger.info("🔄 Memaksa augmentasi dengan menggunakan beberapa file dari setiap kelas")
                
                # Gunakan maksimal 10 file per kelas atau 20% dari total file, mana yang lebih kecil
                class_count = len(result.get('class_counts', {}))
                if class_count > 0:
                    max_files_per_class = min(10, max(1, int(len(image_files) * 0.2 / class_count)))
                else:
                    max_files_per_class = 10
                
                # Ambil beberapa file dari setiap kelas
                forced_files = []
                for class_id, files in result.get('files_metadata', {}).items():
                    class_files = list(files.keys())[:max_files_per_class]
                    forced_files.extend(class_files)
                    self.logger.info(f"📑 Memaksa augmentasi untuk kelas {class_id} dengan {len(class_files)} file")
                
                # Update result dengan file yang dipaksa
                result['selected_files'] = forced_files
                result['forced_augmentation'] = True
                
                # Jika masih tidak ada file yang terpilih, gunakan semua file
                if not forced_files:
                    self.logger.warning("⚠️ Tidak dapat memaksa augmentasi, menggunakan semua file")
                    result['selected_files'] = image_files
                    result['forced_augmentation'] = True
                
                self.logger.info(f"🎯 Menggunakan {len(result['selected_files'])} file untuk augmentasi yang dipaksa")
                
            return result
        except Exception as e:
            self.logger.warning(f"⚠️ Error saat balancing kelas: {str(e)}. Menggunakan semua file.")
            # Return format compatible data untuk mode error fallback
            return {
                'files_by_class': {},
                'selected_files': image_files,
                'class_counts': {}
            }
    
    def stop_processing(self):
        """Hentikan proses augmentasi yang sedang berjalan."""
        self._stop_signal = True
        self.logger.info("🛑 Menghentikan proses augmentasi...")