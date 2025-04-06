"""
File: smartcash/dataset/services/downloader/file_processor.py
Deskripsi: Komponen untuk memproses file dataset, ZIP, dan strukturisasi dengan optimasi performa
"""

import os
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List, Union, Optional, Any, Callable
from concurrent.futures import ThreadPoolExecutor
from tqdm.auto import tqdm
import time

from smartcash.common.logger import get_logger
from smartcash.dataset.utils.dataset_constants import DEFAULT_SPLITS
from smartcash.common.io import (
    ensure_dir, copy_files, move_files, extract_zip as wrapper_extract_zip
)
from smartcash.dataset.services.downloader.notification_utils import notify_service_event

class DownloadFileProcessor:
    """Processor untuk operasi file dataset dengan dukungan ZIP dan restrukturisasi."""
    
    def __init__(self, logger=None, num_workers: int = None, observer_manager=None):
        """
        Inisialisasi file processor dengan optimasi thread count.
        """
        self.logger = logger or get_logger("file_processor")
        # Optimasi: gunakan CPU count sebagai default tapi batasi maksimum
        self.num_workers = min(num_workers or os.cpu_count() or 4, 16)
        self.observer_manager = observer_manager
        self.logger.info(f"📂 DownloadFileProcessor diinisialisasi dengan {self.num_workers} workers")
    
    def process_zip_file(
        self,
        zip_path: Union[str, Path],
        output_dir: Union[str, Path],
        extract_only: bool = False,
        remove_zip: bool = False,
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """
        Proses file ZIP dataset dengan ekstraksi dan strukturisasi.
        """
        start_time = time.time()
        zip_path, output_path = Path(zip_path), Path(output_dir)
        
        notify_service_event("zip_processing", "start", self, self.observer_manager,
                          message="Memulai proses file ZIP", zip_file=str(zip_path))
        
        self.logger.info(f"📦 Memproses file ZIP dataset: {zip_path}")
        
        # Validasi cepat file ZIP
        if not self._validate_zip_file(zip_path):
            msg = f"File ZIP tidak valid: {zip_path}"
            self.logger.error(f"❌ {msg}")
            notify_service_event("zip_processing", "error", self, self.observer_manager, message=msg)
            return {"status": "error", "message": msg}
        
        # Setup direktori
        ensure_dir(output_path)
        tmp_extract_dir = output_path.with_name(f"{output_path.name}_extract_temp")
        ensure_dir(tmp_extract_dir)
        
        try:
            # Step 1: Ekstraksi
            notify_service_event("zip_processing", "progress", self, self.observer_manager,
                              step="extract", message="Ekstraksi file ZIP",
                              progress=1, total_steps=3, current_step=1)
            
            extraction_result = self._extract_zip(zip_path, tmp_extract_dir, remove_zip, show_progress)
            
            if extraction_result.get("errors", 0) > 0:
                raise ValueError("Error saat ekstraksi file ZIP")
                
            self.logger.info(f"✅ Ekstraksi selesai: {extraction_result.get('extracted', 0)} file")
            
            # Step 2: Strukturisasi (jika diperlukan)
            if not extract_only:
                notify_service_event("zip_processing", "progress", self, self.observer_manager,
                                  step="structure", message="Menyesuaikan struktur dataset",
                                  progress=2, total_steps=3, current_step=2)
                
                structure_fixed = self._fix_dataset_structure(tmp_extract_dir)
                
                # Step 3: Pindahkan ke output dir
                notify_service_event("zip_processing", "progress", self, self.observer_manager,
                                  step="move", message=f"Memindahkan dataset ke {output_path}",
                                  progress=3, total_steps=3, current_step=3)
                
                copy_result = self._copy_dataset(
                    tmp_extract_dir, output_path, clear_target=True, show_progress=show_progress
                )
            else:
                # Jika hanya ekstrak, langsung pindahkan semua konten
                self._copy_directory_contents(tmp_extract_dir, output_path, show_progress)
            
            # Step 4: Dapatkan statistik dan selesaikan
            stats = self._get_dataset_stats(output_path)
            elapsed_time = time.time() - start_time
            
            notify_service_event("zip_processing", "complete", self, self.observer_manager,
                              message=f"Proses file ZIP selesai: {stats.get('total_images', 0)} gambar",
                              duration=elapsed_time)
            
            self.logger.success(
                f"✅ Proses file ZIP selesai ({elapsed_time:.1f}s)\n"
                f"   • Total file: {extraction_result.get('extracted', 0)}\n"
                f"   • Gambar: {stats.get('total_images', 0)}\n"
                f"   • Label: {stats.get('total_labels', 0)}\n"
                f"   • Output: {output_path}"
            )
            
            return {
                "status": "success",
                "output_dir": str(output_path),
                "file_count": extraction_result.get('extracted', 0),
                "stats": stats,
                "duration": elapsed_time
            }
            
        except Exception as e:
            notify_service_event("zip_processing", "error", self, self.observer_manager,
                              message=f"Error memproses file ZIP: {str(e)}")
                              
            self.logger.error(f"❌ Error memproses file ZIP: {str(e)}")
            return {"status": "error", "message": str(e)}
        
        finally:
            # Cleanup temporary dir
            if tmp_extract_dir.exists():
                shutil.rmtree(tmp_extract_dir, ignore_errors=True)
    
    def export_to_local(
        self,
        source_dir: Union[str, Path],
        target_dir: Union[str, Path],
        show_progress: bool = True,
        num_workers: Optional[int] = None
    ) -> Dict[str, int]:
        """
        Export dataset dari format Roboflow ke struktur lokal standar.
        """
        source_path, target_path = Path(source_dir), Path(target_dir)
        self.logger.info(f"📤 Export dataset dari {source_path} ke {target_path}")
        
        return self._copy_dataset(
            source_path, 
            target_path,
            clear_target=True,
            show_progress=show_progress,
            num_workers=num_workers or self.num_workers
        )
    
    def _validate_zip_file(self, zip_path: Path) -> bool:
        """Validasi file ZIP cepat."""
        try:
            return zip_path.is_file() and zipfile.is_zipfile(zip_path)
        except Exception:
            return False
    
    def _extract_zip(
        self, 
        zip_path: Path, 
        extract_dir: Path, 
        remove_zip: bool, 
        show_progress: bool
    ) -> Dict[str, int]:
        """Ekstrak file ZIP menggunakan wrapper."""
        # Gunakan utility yang sudah ada dari file_wrapper
        return wrapper_extract_zip(
            zip_path=zip_path,
            output_dir=extract_dir,
            remove_zip=remove_zip,
            show_progress=show_progress
        )
    
    def _fix_dataset_structure(self, dataset_dir: Path) -> bool:
        """
        Perbaiki struktur dataset agar sesuai format YOLO standar.
        Optimasi dengan deteksi struktur cerdas dan penanganan split.
        """
        dataset_path = Path(dataset_dir)
        
        # Deteksi struktur yang sudah benar
        if self._is_valid_yolo_structure(dataset_path):
            self.logger.info("✅ Struktur dataset sudah sesuai standar YOLO")
            return True
        
        # Cek struktur dengan split tapi tanpa subdirektori images/labels
        if all((dataset_path / split).exists() for split in DEFAULT_SPLITS):
            self.logger.info("🔄 Menata ulang struktur dataset ke format standar YOLO")
            
            # Perbaiki struktur setiap split secara paralel
            with ThreadPoolExecutor(max_workers=len(DEFAULT_SPLITS)) as executor:
                futures = {}
                for split in DEFAULT_SPLITS:
                    futures[split] = executor.submit(
                        self._organize_split_directory, dataset_path / split
                    )
                # Tunggu semua selesai
                for split, future in futures.items():
                    if future.result():
                        self.logger.debug(f"✅ Split {split} berhasil diorganisir")
                        
            self.logger.success("✅ Struktur dataset berhasil diperbaiki")
            return True
            
        # Cek struktur flat (images dan labels di root tanpa split)
        elif (dataset_path / 'images').exists() and (dataset_path / 'labels').exists():
            self.logger.info("🔄 Memigrasikan struktur flat ke struktur split (train)")
            
            # Buat split train dengan subdirektori
            train_img_dir = dataset_path / 'train' / 'images'
            train_label_dir = dataset_path / 'train' / 'labels'
            ensure_dir(train_img_dir)
            ensure_dir(train_label_dir)
            
            # Copy files
            copy_files(dataset_path / 'images', train_img_dir)
            copy_files(
                source_dir=dataset_path / 'labels',
                target_dir=train_label_dir,
                file_list=[f for f in (dataset_path / 'labels').glob('*.txt') 
                         if f.stem.lower() not in {'readme', 'classes', 'data'}]
            )
            
            self.logger.success("✅ Struktur dataset berhasil diperbaiki")
            return True
            
        self.logger.warning("⚠️ Tidak dapat mengenali struktur dataset")
        return False
    
    def _organize_split_directory(self, split_dir: Path) -> bool:
        """Mengorganisir satu direktori split dengan pemindahan file ke subdirektori yang tepat."""
        try:
            img_dir, label_dir = split_dir / 'images', split_dir / 'labels'
            ensure_dir(img_dir)
            ensure_dir(label_dir)
            
            # Pindahkan file gambar ke subdirektori images
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                for img_file in split_dir.glob(f'*{ext}'):
                    if img_file.parent == split_dir:  # Hanya file langsung di split_dir
                        shutil.move(str(img_file), str(img_dir / img_file.name))
            
            # Pindahkan file label ke subdirektori labels, kecuali README dll
            for label_file in split_dir.glob('*.txt'):
                if (label_file.parent == split_dir and  # Hanya file langsung di split_dir
                    label_file.stem.lower() not in {'readme', 'classes', 'data'}):
                    shutil.move(str(label_file), str(label_dir / label_file.name))
                    
            return True
        except Exception as e:
            self.logger.warning(f"⚠️ Error saat mengorganisir {split_dir}: {str(e)}")
            return False
    
    def _is_valid_yolo_structure(self, dataset_path: Path) -> bool:
        """Verifikasi apakah struktur dataset sudah sesuai format YOLO standard."""
        # Valid jika setidaknya satu split memiliki struktur benar
        for split in DEFAULT_SPLITS:
            split_path = dataset_path / split
            if not split_path.exists():
                continue
                
            if not ((split_path / 'images').exists() and (split_path / 'labels').exists()):
                return False
        
        # Setidaknya train harus ada
        return (dataset_path / 'train' / 'images').exists() and (dataset_path / 'train' / 'labels').exists()
    
    def _copy_dataset(
        self,
        source_dir: Path,
        target_dir: Path,
        clear_target: bool = True,
        show_progress: bool = True,
        num_workers: Optional[int] = None
    ) -> Dict[str, int]:
        """Copy dataset dengan pemrosesan pair gambar dan label."""
        num_workers = num_workers or self.num_workers
        stats = {'copied': 0, 'errors': 0}
        
        # Bersihkan target dir jika diminta
        if clear_target:
            self._clean_target_directory(target_dir)
            
        # Buat struktur direktori target
        for split in DEFAULT_SPLITS:
            ensure_dir(target_dir / split / 'images')
            ensure_dir(target_dir / split / 'labels')
        
        # Ambil pairs gambar dan label untuk disalin
        copy_tasks = []
        
        for split in DEFAULT_SPLITS:
            src_split = source_dir / split
            if not src_split.exists():
                src_split = source_dir  # Fallback ke source dir jika split tidak ada
                
            src_img_dir = src_split / 'images' if (src_split / 'images').exists() else src_split
            src_label_dir = src_split / 'labels' if (src_split / 'labels').exists() else src_split
            
            dst_img_dir = target_dir / split / 'images'
            dst_label_dir = target_dir / split / 'labels'
            
            # Buat tasks untuk copy
            self._gather_image_label_pairs(
                src_img_dir, src_label_dir, dst_img_dir, dst_label_dir, copy_tasks
            )
        
        # Progress bar
        pbar = tqdm(total=len(copy_tasks), desc="📤 Export Dataset", disable=not show_progress)
        
        # Process tasks in batches for better performance
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(self._copy_file_pair, task) for task in copy_tasks]
            
            for future in futures:
                if future.result():
                    stats['copied'] += 1
                else:
                    stats['errors'] += 1
                pbar.update(1)
                
        pbar.close()
        
        self.logger.success(f"✅ Export selesai: {stats['copied']} file berhasil disalin")
        return stats
        
    def _gather_image_label_pairs(
        self, 
        src_img_dir: Path, 
        src_label_dir: Path,
        dst_img_dir: Path,
        dst_label_dir: Path,
        tasks: List
    ) -> None:
        """Kumpulkan pasangan gambar dan label untuk diproses."""
        # Kumpulkan semua gambar dan periksa label yang bersesuaian
        for img_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG']:
            for img_path in src_img_dir.glob(f'*{img_ext}'):
                label_path = src_label_dir / f"{img_path.stem}.txt"
                if label_path.is_file():
                    tasks.append((img_path, label_path, dst_img_dir, dst_label_dir))
    
    def _copy_file_pair(self, task: tuple) -> bool:
        """Copy satu pasangan file gambar dan label."""
        src_img, src_label, dst_img_dir, dst_label_dir = task
        try:
            shutil.copy2(src_img, dst_img_dir)
            shutil.copy2(src_label, dst_label_dir)
            return True
        except Exception:
            return False
            
    def _copy_directory_contents(
        self,
        src_dir: Path,
        dst_dir: Path,
        show_progress: bool = True
    ) -> None:
        """Copy seluruh isi direktori dengan progress tracking."""
        items = list(src_dir.iterdir())
        with tqdm(total=len(items), desc="🔄 Memindahkan file", disable=not show_progress) as pbar:
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = []
                for item in items:
                    if item.is_file():
                        futures.append(executor.submit(shutil.copy2, item, dst_dir))
                    elif item.is_dir():
                        dst_item = dst_dir / item.name
                        ensure_dir(dst_item)
                        futures.append(executor.submit(
                            self._copy_directory_contents, item, dst_item, False
                        ))
                
                for future in futures:
                    try:
                        future.result()
                    except Exception:
                        pass
                    finally:
                        pbar.update(1)
    
    def _clean_target_directory(self, target_path: Path) -> None:
        """Bersihkan direktori target untuk dataset baru."""
        for split in DEFAULT_SPLITS:
            split_path = target_path / split
            if split_path.exists():
                for subdir in ['images', 'labels']:
                    subdir_path = split_path / subdir
                    if subdir_path.exists():
                        # Hapus file secara paralel
                        files = list(subdir_path.glob('*.*'))
                        if files:
                            self.logger.info(f"🧹 Membersihkan direktori {subdir}: {subdir_path}")
                            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                                list(executor.map(lambda f: f.unlink(missing_ok=True), files))
                    
                    # Pastikan direktori ada
                    ensure_dir(subdir_path)
    
    def _get_dataset_stats(self, dataset_dir: Path) -> Dict[str, Any]:
        """Dapatkan statistik dataset dengan pengukuran paralel."""
        stats = {'total_images': 0, 'total_labels': 0}
        
        # Hitung jumlah file paralel untuk setiap split
        with ThreadPoolExecutor(max_workers=len(DEFAULT_SPLITS)) as executor:
            futures = {}
            for split in DEFAULT_SPLITS:
                images_dir = dataset_dir / split / 'images'
                labels_dir = dataset_dir / split / 'labels'
                
                if images_dir.exists():
                    futures[split + '_images'] = executor.submit(len, list(images_dir.glob('*.*')))
                if labels_dir.exists():
                    futures[split + '_labels'] = executor.submit(len, list(labels_dir.glob('*.txt')))
            
            for key, future in futures.items():
                count = future.result()
                split, type_key = key.split('_')
                
                # Tambahkan ke total dan mapping
                stats[key] = count
                stats['total_' + type_key] += count
                
                if split not in stats:
                    stats[split] = {}
                stats[split][type_key] = count
        
        return stats