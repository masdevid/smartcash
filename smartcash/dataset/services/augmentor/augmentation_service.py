"""
File: smartcash/dataset/services/augmentor/augmentation_service.py
Deskripsi: Layanan augmentasi dataset dengan integrasi observer pattern
"""

import os, time
from typing import Dict, List, Any, Optional, Callable, Union
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm.auto import tqdm

from smartcash.common.logger import get_logger
from smartcash.common.exceptions import DatasetError
from smartcash.common.config import SimpleConfigManager, get_config_manager
from smartcash.dataset.services.augmentor.pipeline_factory import AugmentationPipelineFactory
from smartcash.dataset.services.augmentor.bbox_augmentor import BBoxAugmentor
from smartcash.dataset.services.augmentor.class_balancer import ClassBalancer
from smartcash.dataset.services.augmentor.dataset_augmentor import DatasetAugmentor
from smartcash.dataset.utils.dataset_constants import DEFAULT_AUGMENTED_DIR

# Import helper modules yang dipisah
from smartcash.dataset.services.augmentor.helpers.path_helper import setup_paths
from smartcash.dataset.services.augmentor.helpers.augmentation_executor import execute_augmentation_with_tracking
from smartcash.dataset.services.augmentor.helpers.validation_helper import validate_input_files
from smartcash.dataset.utils.move_utils import move_files_to_preprocessed

class AugmentationService:
    """Layanan augmentasi dataset dengan integrasi observer pattern."""
    
    def __init__(self, config: Dict[str, Any], data_dir: str = 'data', logger=None, num_workers: int = 4, ui_components: Dict[str, Any] = None):
        """
        Inisialisasi AugmentationService.
        
        Args:
            config: Konfigurasi augmentasi
            data_dir: Direktori data
            logger: Logger untuk logging
            num_workers: Jumlah worker untuk parallel processing
            ui_components: Komponen UI untuk notifikasi
        """
        self.config = config
        self.data_dir = Path(data_dir)
        self.logger = logger or get_logger(__name__)
        self.num_workers = num_workers
        self.ui_components = ui_components
        self.config_manager = get_config_manager()
        
        # Setup paths
        self.augmented_dir = Path(config.get('augmentation', {}).get('output_dir', DEFAULT_AUGMENTED_DIR))
        
        # Ensure directories exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.augmented_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize augmentor
        self._augmentor = None
        
        # Log initialization
        self.logger.info(f"✅ AugmentationService diinisialisasi:")
        self.logger.info(f"  - Data dir: {self.data_dir}")
        self.logger.info(f"  - Augmented dir: {self.augmented_dir}")
        
        self._stop_signal = False
        self._progress_callbacks = []
        self._notification_manager = None
        
        # Inisialisasi notification manager jika ui_components tersedia
        if ui_components:
            try:
                # Import notification manager hanya jika diperlukan
                from smartcash.ui.dataset.augmentation.utils.notification_manager import NotificationManager
                
                # Cek apakah notification_manager sudah ada di ui_components
                if 'notification_manager' in ui_components and ui_components['notification_manager'] is not None:
                    self._notification_manager = ui_components['notification_manager']
                else:
                    # Buat instance baru jika belum ada
                    self._notification_manager = NotificationManager(ui_components)
                    ui_components['notification_manager'] = self._notification_manager
                    
                self.logger.info("✅ NotificationManager berhasil diinisialisasi")
            except ImportError as e:
                self.logger.warning(f"⚠️ NotificationManager tidak tersedia: {str(e)}")
            except Exception as e:
                self.logger.warning(f"⚠️ Error saat inisialisasi NotificationManager: {str(e)}")
    
    @property
    def augmentor(self) -> DatasetAugmentor:
        """Lazy initialization of augmentor."""
        if self._augmentor is None:
            # Get config from config manager
            config = self.config_manager.get_config()
            
            # Create integrated config
            integrated_config = {
                'augmentation': {
                    'types': config.get('augmentation', {}).get('types', ['combined']),
                    'num_variations': config.get('augmentation', {}).get('num_variations', 2),
                    'output_dir': str(self.augmented_dir),
                    'process_bboxes': config.get('augmentation', {}).get('process_bboxes', True),
                    'target_balance': config.get('augmentation', {}).get('target_balance', True),
                    'num_workers': self.num_workers,
                    'target_count': config.get('augmentation', {}).get('target_count', 1000)
                },
                'data': {
                    'dir': str(self.data_dir)
                }
            }
            
            # Initialize augmentor
            self._augmentor = DatasetAugmentor(
                config=integrated_config,
                logger=self.logger
            )
            
            # Register progress callback if ui_components exists
            if self.ui_components and 'progress_callback' in self.ui_components:
                self._augmentor.register_progress_callback(
                    self.ui_components['progress_callback']
                )
        
        return self._augmentor
    
    def register_progress_callback(self, callback: Callable) -> None:
        """
        Register progress callback untuk tracking progress.
        
        Args:
            callback: Fungsi callback untuk progress tracking
        """
        if self.ui_components:
            self.ui_components['progress_callback'] = callback
            if self._augmentor:
                self._augmentor.register_progress_callback(callback)
    
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
        """
        Augmentasi dataset dan simpan hasilnya.
        
        Args:
            split: Split dataset ('train', 'val', 'test')
            augmentation_types: Jenis augmentasi yang akan digunakan
            num_variations: Jumlah variasi per gambar
            output_prefix: Prefix untuk file output
            validate_results: Validasi hasil augmentasi
            resume: Lanjutkan augmentasi yang terhenti
            process_bboxes: Proses bounding box
            target_balance: Balance kelas target
            num_workers: Jumlah worker untuk parallel processing
            move_to_preprocessed: Pindahkan hasil ke direktori preprocessed
            target_count: Target jumlah gambar per kelas
            
        Returns:
            Dict: Statistik hasil augmentasi
        """
        try:
            # Get latest config
            config = self.config_manager.get_config()
            
            # Update parameters from config if not specified
            if augmentation_types is None:
                augmentation_types = config.get('augmentation', {}).get('types', ['combined'])
            if num_workers is None:
                num_workers = config.get('augmentation', {}).get('num_workers', self.num_workers)
            
            # Get augmentor and run augmentation
            result = self.augmentor.augment_dataset(
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
            
            return result
            
        except Exception as e:
            error_msg = f"Error saat augmentasi dataset: {str(e)}"
            self.logger.error(error_msg)
            raise DatasetError(error_msg)
    
    def cleanup(self):
        """Cleanup resources."""
        self._augmentor = None
        self.ui_components = None
    
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