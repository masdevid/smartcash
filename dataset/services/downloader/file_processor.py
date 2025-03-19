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

from smartcash.common.logger import get_logger
from smartcash.dataset.utils.dataset_utils import DEFAULT_SPLITS
from smartcash.components.observer.manager_observer import ObserverManager
from smartcash.components.observer import notify, EventTopics

class FileProcessor:
    """Processor untuk operasi file dataset dengan dukungan ZIP dan restrukturisasi."""
    
    def __init__(self, logger=None, num_workers: int = 4, observer_manager=None):
        """
        Inisialisasi FileProcessor.
        
        Args:
            logger: Logger kustom (opsional)
            num_workers: Jumlah worker untuk proses paralel
            observer_manager: Observer manager untuk notifikasi progress
        """
        self.logger = logger or get_logger("file_processor")
        self.num_workers = num_workers
        self.observer_manager = observer_manager
        
        # Coba dapatkan observer_manager jika tidak disediakan
        if self.observer_manager is None:
            try:
                self.observer_manager = ObserverManager() 
            except (ImportError, AttributeError):
                pass
    
    # Sisanya tidak perlu diubah
    def process_zip_file(
        self,
        zip_path: Union[str, Path],
        output_dir: Union[str, Path],
        extract_only: bool = False,
        remove_zip: bool = False,
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """
        Proses lengkap file ZIP: ekstrak, restrukturisasi, dan validasi.
        
        Args:
            zip_path: Path ke file ZIP
            output_dir: Direktori output
            extract_only: Hanya ekstrak tanpa restrukturisasi
            remove_zip: Hapus file ZIP setelah proses
            show_progress: Tampilkan progress bar
            
        Returns:
            Dictionary berisi hasil proses
        """
        import time
        start_time = time.time()
        
        # Notifikasi mulai proses
        self._notify_event("ZIP_PROCESSING_START", message=f"Memulai proses file ZIP", 
                         zip_file=str(zip_path), status="info")
        
        self.logger.info(f"📦 Memproses file ZIP dataset: {zip_path}")
        
        zip_path, output_path = Path(zip_path), Path(output_dir)
        
        # Validasi file ZIP
        if not zip_path.exists() or not zipfile.is_zipfile(zip_path):
            self.logger.error(f"❌ File ZIP tidak valid: {zip_path}")
            self._notify_event("ZIP_PROCESSING_ERROR", message=f"File ZIP tidak valid", status="error")
            return {"status": "error", "message": "File ZIP tidak valid"}
        
        # Setup direktori output
        os.makedirs(output_path, exist_ok=True)
        
        # Direktori ekstraksi sementara
        tmp_extract_dir = output_path.with_name(f"{output_path.name}_extract_temp")
        if tmp_extract_dir.exists():
            shutil.rmtree(tmp_extract_dir)
        tmp_extract_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Ekstrak ZIP
            self._notify_event("ZIP_PROCESSING_PROGRESS", step="extract", message=f"Ekstraksi file ZIP",
                            progress=1, total_steps=3, current_step=1, status="info")
            
            self.logger.info(f"📂 Mengekstrak file ZIP ke {tmp_extract_dir}")
            self.extract_zip(zip_path, tmp_extract_dir, remove_zip, show_progress)
            
            # Periksa hasil ekstraksi
            extracted_files = list(tmp_extract_dir.glob('**/*'))
            if not extracted_files:
                raise ValueError(f"❌ Tidak ada file ditemukan setelah ekstraksi")
                
            self.logger.info(f"✅ Ekstraksi selesai: {len(extracted_files)} file")
            
            # Proses struktur dataset jika diperlukan
            if not extract_only:
                self._notify_event("ZIP_PROCESSING_PROGRESS", step="structure", 
                               message="Menyesuaikan struktur dataset",
                               progress=2, total_steps=3, current_step=2, status="info")
                
                self.logger.info(f"🔧 Menyesuaikan struktur dataset...")
                self.fix_dataset_structure(tmp_extract_dir)
                
                # Pindahkan hasil ke direktori final
                self._notify_event("ZIP_PROCESSING_PROGRESS", step="move", 
                               message=f"Memindahkan dataset ke {output_path}",
                               progress=3, total_steps=3, current_step=3, status="info")
                
                self.logger.info(f"🔄 Memindahkan dataset ke {output_path}")
                self.copy_dataset_to_data_dir(tmp_extract_dir, output_path)
            else:
                # Jika hanya ekstrak, pindahkan langsung semua file
                self._copy_directory_contents(tmp_extract_dir, output_path, show_progress)
            
            # Hapus direktori temporari
            shutil.rmtree(tmp_extract_dir)
            
            # Dapatkan statistik
            stats = self._get_dataset_stats(output_path)
            elapsed_time = time.time() - start_time
            
            # Notifikasi selesai
            self._notify_event("ZIP_PROCESSING_COMPLETE", 
                            message=f"Proses file ZIP selesai: {stats.get('total_images', 0)} gambar",
                            duration=elapsed_time, status="success")
            
            result = {
                "status": "success",
                "output_dir": str(output_path),
                "file_count": len(extracted_files),
                "stats": stats,
                "duration": elapsed_time
            }
            
            self.logger.success(
                f"✅ Proses file ZIP selesai ({elapsed_time:.1f}s)\n"
                f"   • Total file: {len(extracted_files)}\n"
                f"   • Gambar: {stats.get('total_images', 0)}\n"
                f"   • Label: {stats.get('total_labels', 0)}\n"
                f"   • Output: {output_path}"
            )
            
            return result
            
        except Exception as e:
            self._notify_event("ZIP_PROCESSING_ERROR", message=f"Error memproses file ZIP: {str(e)}",
                            status="error")
            
            # Hapus direktori temporari jika ada error
            if tmp_extract_dir.exists():
                shutil.rmtree(tmp_extract_dir)
                
            self.logger.error(f"❌ Error memproses file ZIP: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def extract_zip(
        self, 
        zip_path: Union[str, Path], 
        output_dir: Union[str, Path],
        remove_zip: bool = False,
        show_progress: bool = True
    ) -> bool:
        """
        Ekstrak file zip ke direktori output dengan progress tracking.
        
        Args:
            zip_path: Path ke file zip
            output_dir: Direktori output
            remove_zip: Apakah menghapus file zip setelah ekstraksi
            show_progress: Tampilkan progress bar
            
        Returns:
            Boolean menunjukkan keberhasilan
        """
        zip_path, output_dir = Path(zip_path), Path(output_dir)
        
        try:
            # Buka file zip dan periksa strukturnya
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                files = zip_ref.infolist()
                
                if not files:
                    self.logger.error(f"❌ File ZIP kosong: {zip_path}")
                    return False
                
                # Ekstrak dengan progress
                total_size = sum(f.file_size for f in files)
                extracted_size = 0
                
                # Buat progress bar
                progress = tqdm(total=total_size, unit='B', unit_scale=True, 
                               desc=f"📦 Extract {zip_path.name}", disable=not show_progress)
                
                # Ekstrak file
                for file in files:
                    zip_ref.extract(file, output_dir)
                    extracted_size += file.file_size
                    progress.update(file.file_size)
                    
                    # Update progress observer setiap 5% atau 5MB 
                    if (total_size > 0 and 
                        (extracted_size % max(int(total_size * 0.05), 5*1024*1024) < file.file_size or 
                         extracted_size >= total_size)):
                        self._notify_event("ZIP_EXTRACT_PROGRESS", progress=extracted_size, 
                                        total=total_size, percentage=int(100*extracted_size/total_size))
                
                progress.close()
            
            # Hapus zip jika diminta
            if remove_zip:
                zip_path.unlink()
                self.logger.info(f"🗑️ Zip dihapus: {zip_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Ekstraksi gagal: {str(e)}")
            return False
    
    def export_to_local(
        self,
        source_dir: Union[str, Path],
        target_dir: Union[str, Path],
        show_progress: bool = True,
        num_workers: Optional[int] = None,
        clear_target: bool = True
    ) -> Dict[str, int]:
        """
        Export dataset dari struktur Roboflow ke struktur lokal standar.
        
        Args:
            source_dir: Direktori sumber
            target_dir: Direktori target
            show_progress: Tampilkan progress bar
            num_workers: Jumlah worker untuk proses paralel
            clear_target: Hapus file yang sudah ada di direktori target
            
        Returns:
            Dictionary berisi statistik export
        """
        source_path, target_path = Path(source_dir), Path(target_dir)
        self.logger.info(f"📤 Export dataset dari {source_path} ke {target_path}")
        
        # Gunakan num_workers instance jika tidak ada parameter
        num_workers = num_workers or self.num_workers
        
        # Bersihkan direktori target jika diminta
        if clear_target:
            self._clean_target_directory(target_path)
            
        # Inisialisasi statistik
        stats = {'copied': 0, 'errors': 0}
        
        # Buat direktori target
        for split in DEFAULT_SPLITS:
            os.makedirs(target_path / split / 'images', exist_ok=True)
            os.makedirs(target_path / split / 'labels', exist_ok=True)
        
        # Daftar file untuk disalin
        copy_tasks = []
        
        # Temukan semua gambar dan label
        for split in DEFAULT_SPLITS:
            # Identifikasi direktori sumber
            src_split = source_path / split if (source_path / split).exists() else source_path
            src_img_dir = src_split / 'images' if (src_split / 'images').exists() else src_split
            src_label_dir = src_split / 'labels' if (src_split / 'labels').exists() else src_split
            
            # Target direktori
            dst_img_dir = target_path / split / 'images'
            dst_label_dir = target_path / split / 'labels'
            
            # Temukan pasangan file
            for img_path in src_img_dir.glob('*.jpg'):
                label_path = src_label_dir / f"{img_path.stem}.txt"
                if label_path.exists():
                    copy_tasks.append((img_path, label_path, dst_img_dir, dst_label_dir))
        
        # Progress bar
        pbar = tqdm(total=len(copy_tasks), desc="📤 Export Dataset") if show_progress else None
        
        # Copy file secara paralel
        def copy_file_pair(src_img, src_label, dst_img_dir, dst_label_dir):
            try:
                # Copy gambar dan label
                shutil.copy2(src_img, dst_img_dir / src_img.name)
                shutil.copy2(src_label, dst_label_dir / src_label.name)
                return True
            except Exception as e:
                self.logger.debug(f"⚠️ Error saat menyalin file {src_img.name}: {str(e)}")
                return False
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(copy_file_pair, *task) for task in copy_tasks]
            
            # Proses hasil
            for future in futures:
                stats['copied' if future.result() else 'errors'] += 1
                if pbar: pbar.update(1)
        
        if pbar: pbar.close()
        self.logger.success(f"✅ Export selesai: {stats['copied']} file berhasil disalin")
        return stats
    
    def fix_dataset_structure(self, dataset_dir: Union[str, Path]) -> bool:
        """
        Memperbaiki struktur dataset agar sesuai dengan format YOLOv5 standar.
        
        Args:
            dataset_dir: Direktori dataset
            
        Returns:
            Boolean menunjukkan keberhasilan perbaikan
        """
        dataset_path = Path(dataset_dir)
        splits = DEFAULT_SPLITS
        
        # Cek struktur yang sudah benar
        if all((dataset_path / split / 'images').exists() and 
              (dataset_path / split / 'labels').exists() for split in splits):
            self.logger.info("✅ Struktur dataset sudah sesuai standar YOLO")
            return True
            
        # Cek struktur alternatif 1: train/, valid/, test/ tanpa subdirektori images/labels
        if all((dataset_path / split).exists() for split in splits):
            self.logger.info("🔄 Menata ulang struktur dataset ke format standar YOLO")
            
            for split in splits:
                split_dir = dataset_path / split
                img_dir = split_dir / 'images'
                label_dir = split_dir / 'labels'
                
                img_dir.mkdir(exist_ok=True)
                label_dir.mkdir(exist_ok=True)
                
                # Pindahkan file
                for file in split_dir.glob('*.jpg'):
                    shutil.move(file, img_dir / file.name)
                    
                for file in split_dir.glob('*.txt'):
                    if file.stem.lower() not in ['readme', 'classes', 'data']:
                        shutil.move(file, label_dir / file.name)
                    
            self.logger.success("✅ Struktur dataset berhasil diperbaiki")
            return True
            
        # Cek struktur alternatif 2: images/ dan labels/ di root
        elif (dataset_path / 'images').exists() and (dataset_path / 'labels').exists():
            self.logger.info("🔄 Memigrasikan struktur flat ke struktur split")
            
            # Buat direktori split
            for split in splits:
                os.makedirs(dataset_path / split / 'images', exist_ok=True)
                os.makedirs(dataset_path / split / 'labels', exist_ok=True)
                
            # Untuk kasus ini, tempatkan semua file di 'train'
            for file in (dataset_path / 'images').glob('*.jpg'):
                shutil.copy2(file, dataset_path / 'train' / 'images' / file.name)
                
            for file in (dataset_path / 'labels').glob('*.txt'):
                if file.stem.lower() not in ['readme', 'classes', 'data']:
                    shutil.copy2(file, dataset_path / 'train' / 'labels' / file.name)
                
            self.logger.success("✅ Struktur dataset berhasil diperbaiki")
            return True
            
        else:
            self.logger.warning("⚠️ Tidak dapat mengenali struktur dataset")
            return False
    
    def copy_dataset_to_data_dir(
        self, 
        source_dir: Union[str, Path], 
        data_dir: Union[str, Path],
        clear_target: bool = True
    ) -> Dict[str, int]:
        """Alias untuk export_to_local."""
        return self.export_to_local(source_dir, data_dir, True, self.num_workers, clear_target)
    
    def _copy_directory_contents(
        self, 
        src_dir: Path, 
        dst_dir: Path, 
        show_progress: bool = True
    ) -> None:
        """Salin seluruh isi direktori sumber ke tujuan dengan progress."""
        total_items = len([f for f in src_dir.iterdir()])
        with tqdm(total=total_items, desc="🔄 Memindahkan file", disable=not show_progress) as pbar:
            for item in src_dir.iterdir():
                if item.is_file():
                    shutil.copy2(item, dst_dir)
                elif item.is_dir():
                    shutil.copytree(item, dst_dir / item.name, dirs_exist_ok=True)
                pbar.update(1)
    
    def _clean_target_directory(self, target_path: Path) -> None:
        """
        Bersihkan direktori target sebelum export, hanya menghapus file di folder yang diharapkan.
        
        Args:
            target_path: Path direktori target
        """
        # Hanya hapus direktori split yang tepat untuk keamanan
        for split in DEFAULT_SPLITS:
            split_path = target_path / split
            if split_path.exists():
                # Hapus gambar dan label
                for subdir in ['images', 'labels']:
                    subdir_path = split_path / subdir
                    if subdir_path.exists():
                        self.logger.info(f"🧹 Membersihkan direktori {subdir}: {subdir_path}")
                        for file in subdir_path.glob('*.*'):
                            try:
                                file.unlink()
                            except Exception as e:
                                self.logger.debug(f"⚠️ Gagal menghapus {file}: {str(e)}")
                
                # Buat direktori jika belum ada
                (split_path / 'images').mkdir(parents=True, exist_ok=True)
                (split_path / 'labels').mkdir(parents=True, exist_ok=True)
    
    def _get_dataset_stats(self, dataset_dir: Path) -> Dict[str, Any]:
        """
        Hitung statistik dataset (jumlah gambar dan label per split).
        
        Args:
            dataset_dir: Direktori dataset
            
        Returns:
            Dictionary berisi statistik
        """
        stats = {'total_images': 0, 'total_labels': 0}
        
        for split in DEFAULT_SPLITS:
            images_dir = dataset_dir / split / 'images'
            labels_dir = dataset_dir / split / 'labels'
            
            if images_dir.exists():
                image_count = len(list(images_dir.glob('*.*')))
                stats['total_images'] += image_count
                stats[f'{split}_images'] = image_count
                
            if labels_dir.exists():
                label_count = len(list(labels_dir.glob('*.txt')))
                stats['total_labels'] += label_count
                stats[f'{split}_labels'] = label_count
        
        return stats
    
    def _notify_event(self, event_type: str, **kwargs) -> None:
        """
        Helper untuk mengirim notifikasi event secara aman.
        
        Args:
            event_type: Tipe event
            **kwargs: Parameter tambahan untuk event
        """
        if not self.observer_manager: return
        try:
            event_const = getattr(EventTopics, event_type, None)
            if event_const: notify(event_const, self, **kwargs)
        except (ImportError, AttributeError):
            pass