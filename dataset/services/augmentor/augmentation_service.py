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
    
    def __init__(self, config: Dict = None, data_dir: str = 'data', logger=None, num_workers: int = None, ui_components: Dict[str, Any] = None):
        """Inisialisasi AugmentationService dengan parameter utama."""
        self.config = config or {}
        self.data_dir = data_dir
        self.ui_components = ui_components or {}
        
        # Gunakan logger dengan konteks augmentation untuk mencegah interferensi dengan log download
        if logger and hasattr(logger, 'bind'):
            try:
                self.logger = logger.bind(context="augmentation_only")
            except Exception as e:
                # Fallback jika bind tidak tersedia
                self.logger = logger or get_logger("augmentation_service")
        else:
            self.logger = logger or get_logger("augmentation_service")
            
        self.num_workers = num_workers if num_workers is not None else self.config.get('augmentation', {}).get('num_workers', 4)
        self.logger.info(f"🔧 Menggunakan {self.num_workers} worker untuk augmentasi")
        
        # Inisialisasi komponen-komponen utama dengan one-liner
        self.pipeline_factory = AugmentationPipelineFactory(self.config, self.logger)
        self.bbox_augmentor = BBoxAugmentor(self.config, self.logger)
        self.class_balancer = ClassBalancer(self.config, self.logger)
        
        self._stop_signal = False
        self._progress_callbacks = []
        self._notification_manager = None
        
        # Inisialisasi notification manager jika ui_components tersedia
        if ui_components:
            try:
                # Import notification manager hanya jika diperlukan
                from smartcash.ui.dataset.augmentation.utils.notification_manager import get_notification_manager
                self._notification_manager = get_notification_manager(ui_components)
                self.logger.info("✅ NotificationManager berhasil diinisialisasi")
            except ImportError as e:
                self.logger.warning(f"⚠️ NotificationManager tidak tersedia: {str(e)}")
            except Exception as e:
                self.logger.warning(f"⚠️ Error saat inisialisasi NotificationManager: {str(e)}")
    
    def register_progress_callback(self, callback: Callable = None, notification_manager = None) -> None:
        """Register callback untuk progress tracking atau notification manager.
        
        Args:
            callback: Fungsi callback untuk progress tracking (pola lama)
            notification_manager: Instance NotificationManager (pola baru)
        """
        if notification_manager:
            self._notification_manager = notification_manager
            self.logger.info("✅ NotificationManager berhasil diregistrasi")
        elif callback and callable(callback): 
            self._progress_callbacks.append(callback)
            self.logger.info("✅ Progress callback berhasil diregistrasi")
    
    def report_progress(self, progress: int = None, total: int = None, message: str = None, status: str = 'info', **kwargs) -> None:
        """Laporkan progress dengan callback atau notification manager.
        
        Args:
            progress: Nilai progress (0-100)
            total: Total item yang diproses
            message: Pesan yang akan ditampilkan
            status: Tipe status ('info', 'success', 'warning', 'error')
            **kwargs: Parameter tambahan
        """
        # Gunakan notification manager jika tersedia
        if self._notification_manager:
            try:
                # Deteksi jenis notifikasi berdasarkan parameter dan keyword
                if kwargs.get('process_start', False) or kwargs.get('start', False):
                    # Notifikasi proses dimulai
                    process_name = kwargs.get('process_name', 'augmentation')
                    display_info = message or kwargs.get('display_info', '')
                    split = kwargs.get('split')
                    self._notification_manager.notify_process_start(process_name, display_info, split)
                    return
                    
                elif kwargs.get('process_complete', False) or kwargs.get('complete', False):
                    # Notifikasi proses selesai
                    result = kwargs.get('result', {})
                    display_info = message or kwargs.get('display_info', '')
                    self._notification_manager.notify_process_complete(result, display_info)
                    return
                    
                elif status == 'error' or kwargs.get('error', False):
                    # Notifikasi error
                    error_message = message or kwargs.get('error_message', 'Unknown error')
                    self._notification_manager.notify_process_error(error_message)
                    return
                    
                elif progress is not None:
                    # Update progress bar
                    self._notification_manager.update_progress(progress, message or '')
                    return
                    
                else:
                    # Update status panel
                    self._notification_manager.update_status(status, message or '')
                    return
                    
            except Exception as e:
                self.logger.error(f"❌ Error saat menggunakan NotificationManager: {str(e)}")
                # Fallback ke callback lama jika notification manager gagal
        
        # Fallback ke callback lama jika notification manager tidak tersedia atau gagal
        for callback in self._progress_callbacks:
            try:
                # Buat params bersih dengan one-liner
                explicit_params = ['progress', 'total', 'message', 'status', 'current_progress', 'current_total', 'class_id', 'module_type', 'context']
                filtered_kwargs = {k: v for k, v in kwargs.items() if k not in explicit_params}
                
                # Gabungkan parameter yang ada nilainya dengan one-liner
                params = {k: v for k, v in {
                    'message': message, 'status': status, 
                    'progress': progress, 'total': total,
                    'current_progress': kwargs.get('current_progress'), 
                    'current_total': kwargs.get('current_total'),
                    'class_id': kwargs.get('class_id'),
                    # Tambahkan konteks dan modul untuk memudahkan filter
                    'module_type': kwargs.get('module_type', 'augmentation'),
                    'context': kwargs.get('context', 'augmentation_only')
                }.items() if v is not None}
                
                # Gabungkan dengan filtered_kwargs
                params.update(filtered_kwargs)
                
                # Panggil callback dengan parameter yang ada
                callback(**params)
            except Exception as e:
                self.logger.error(f"❌ Error saat memanggil progress callback: {str(e)}")
    
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
        
        # Notifikasi awal proses augmentasi
        self.report_progress(
            message=f"Memulai augmentasi dataset {split}",
            status="info",
            process_start=True,
            process_name="augmentation",
            display_info=f"{split} ({len(augmentation_types or [])} jenis)",
            split=split
        )
        
        # Validasi input files dengan helper
        image_files, validation_result = validate_input_files(
            paths['images_input_dir'], 
            self.config.get('preprocessing', {}).get('file_prefix', 'rp'),
            self.logger
        )
        
        # Return error jika validasi gagal
        if not validation_result['success']:
            error_message = validation_result['message']
            self.logger.warning(f"⚠️ {error_message}")
            self.report_progress(
                message=error_message,
                status="error",
                error=True,
                error_message=error_message
            )
            return {"status": "error", "message": error_message}
            
        # Persiapkan balancing dengan prioritisasi kelas
        class_data = self._prepare_balancing(image_files, paths, target_count, target_balance)
        if not class_data:
            return {"status": "info", "message": "Tidak ada kelas yang memerlukan augmentasi", "generated": 0}
        
        # Eksekusi augmentasi dengan tracking
        augmentation_result = execute_augmentation_with_tracking(
            service=self,
            class_data=class_data,
            augmentation_types=augmentation_types,
            num_variations=max(1, num_variations),  # Pastikan minimal 1 variasi
            output_prefix=output_prefix,
            validate_results=False,  # Nonaktifkan validasi untuk memastikan semua gambar diproses
            process_bboxes=process_bboxes,
            n_workers=num_workers or self.num_workers,
            paths=paths,
            split=split,
            target_count=target_count,
            start_time=start_time
        )
        
        # Handle move files jika diperlukan
        if move_to_preprocessed and augmentation_result['status'] == 'success':
            self.report_progress(
                message=f"Memindahkan {augmentation_result['generated']} file ke direktori preprocessed",
                status="info", 
                step=2,
                module_type="augmentation",
                context="augmentation_only",
                silent=True  # Tambahkan flag silent untuk mengurangi output log yang tidak perlu
            )
            
            move_success = move_files_to_preprocessed(
                paths['images_output_dir'], paths['labels_output_dir'],
                output_prefix, paths['final_output_dir'], split, self.logger
            )
            
            # Update path output jika berhasil di-move
            if move_success:
                augmentation_result['final_output_dir'] = paths['final_output_dir']
        
        # Notifikasi akhir proses augmentasi
        if augmentation_result['status'] == 'success':
            self.report_progress(
                message=f"Augmentasi berhasil dengan {augmentation_result['generated']} gambar",
                status="success",
                process_complete=True,
                result=augmentation_result,
                display_info=f"{split} ({augmentation_result['generated']} gambar)"
            )
        else:
            self.report_progress(
                message=augmentation_result.get('message', 'Augmentasi gagal'),
                status="error",
                error=True,
                error_message=augmentation_result.get('message', 'Augmentasi gagal')
            )
        
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
            message="Menganalisis distribusi kelas untuk balancing optimal",
            status="info", 
            step=0,
            progress=10,  # Berikan nilai progress awal
            current_progress=1,
            current_total=10
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