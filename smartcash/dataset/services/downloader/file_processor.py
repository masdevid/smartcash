"""
File: smartcash/dataset/services/downloader/file_processor.py
Deskripsi: Perbaikan komponen untuk memproses file dataset dengan penanganan direktori target yang lebih baik
"""

import os
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List, Union, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
from tqdm.auto import tqdm

from smartcash.common.logger import get_logger
from smartcash.dataset.utils.dataset_utils import DEFAULT_SPLITS


class FileProcessor:
    """
    Processor untuk menangani operasi file dataset.
    Mendukung ekstraksi, konversi format, dan strukturisasi dataset.
    """
    
    def __init__(self, logger=None, num_workers: int = 4):
        """
        Inisialisasi FileProcessor.
        
        Args:
            logger: Logger kustom (opsional)
            num_workers: Jumlah worker untuk proses paralel
        """
        self.logger = logger or get_logger("file_processor")
        self.num_workers = num_workers
    
    def clean_existing_download(self, output_dir: Union[str, Path]) -> None:
        """
        Bersihkan download sebelumnya jika ada.
        
        Args:
            output_dir: Direktori output
        """
        output_path = Path(output_dir)
        
        if output_path.exists():
            if self.logger:
                self.logger.info(f"🧹 Membersihkan direktori lama: {output_path}")
            try:
                shutil.rmtree(output_path)
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"⚠️ Gagal menghapus direktori: {str(e)}")
    
    def extract_zip(
        self, 
        zip_path: Union[str, Path], 
        output_dir: Union[str, Path],
        remove_zip: bool = False,
        show_progress: bool = True
    ) -> None:
        """
        Ekstrak file zip ke direktori output.
        
        Args:
            zip_path: Path ke file zip
            output_dir: Direktori output
            remove_zip: Apakah menghapus file zip setelah ekstraksi
            show_progress: Tampilkan progress bar
        """
        zip_path = Path(zip_path)
        output_dir = Path(output_dir)
        
        # Cek apakah file zip valid
        if not zipfile.is_zipfile(zip_path):
            raise ValueError(f"File bukan zip yang valid: {zip_path}")
            
        # Ekstrak file dengan progress bar
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            files = zip_ref.infolist()
            
            # Ekstrak semua file
            if show_progress:
                progress = tqdm(files, desc=f"Ekstraksi {zip_path.name}")
                for file in progress:
                    zip_ref.extract(file, output_dir)
            else:
                zip_ref.extractall(output_dir)

        if remove_zip:
            try:
                zip_path.unlink()
            except Exception as e:
                self.logger.warning(f"⚠️ Gagal menghapus file zip: {str(e)}")
        
        if self.logger:
            self.logger.success(f"✅ Ekstraksi selesai: {output_dir}")
    
    def export_to_local(
        self,
        source_dir: Union[str, Path],
        target_dir: Union[str, Path],
        show_progress: bool = True,
        num_workers: int = None,
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
        source_path = Path(source_dir)
        target_path = Path(target_dir)
        
        if self.logger:
            self.logger.info(f"📤 Export dataset dari {source_path} ke {target_path}")
            
        # Gunakan num_workers instance jika tidak ada parameter
        if num_workers is None:
            num_workers = self.num_workers
        
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
        pbar = tqdm(total=len(copy_tasks), desc="Export Dataset") if show_progress else None
        
        # Copy file secara paralel
        def copy_file_pair(src_img, src_label, dst_img_dir, dst_label_dir):
            try:
                # Copy gambar dan label
                shutil.copy2(src_img, dst_img_dir / src_img.name)
                shutil.copy2(src_label, dst_label_dir / src_label.name)
                return True
            except Exception as e:
                if self.logger:
                    self.logger.debug(f"⚠️ Error saat menyalin file: {str(e)}")
                return False
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = []
            for task in copy_tasks:
                results.append(executor.submit(copy_file_pair, *task))
            
            # Proses hasil
            for future in results:
                if future.result():
                    stats['copied'] += 1
                else:
                    stats['errors'] += 1
                if pbar:
                    pbar.update(1)
        
        if pbar:
            pbar.close()
            
        if self.logger:
            self.logger.success(f"✅ Export selesai: {stats['copied']} file berhasil disalin")
        
        return stats
    
    def _clean_target_directory(self, target_path: Path) -> None:
        """
        Bersihkan direktori target sebelum export.
        
        Args:
            target_path: Path direktori target
        """
        # Hanya hapus direktori split yang tepat untuk keamanan
        for split in DEFAULT_SPLITS:
            split_path = target_path / split
            if split_path.exists():
                # Hapus gambar
                img_dir = split_path / 'images'
                if img_dir.exists():
                    if self.logger:
                        self.logger.info(f"🧹 Membersihkan direktori gambar: {img_dir}")
                    
                    # Hapus file-file gambar
                    for img_file in img_dir.glob('*.*'):
                        try:
                            img_file.unlink()
                        except Exception as e:
                            if self.logger:
                                self.logger.debug(f"⚠️ Gagal menghapus file {img_file}: {str(e)}")
                
                # Hapus label
                label_dir = split_path / 'labels'
                if label_dir.exists():
                    if self.logger:
                        self.logger.info(f"🧹 Membersihkan direktori label: {label_dir}")
                    
                    # Hapus file-file label
                    for label_file in label_dir.glob('*.*'):
                        try:
                            label_file.unlink()
                        except Exception as e:
                            if self.logger:
                                self.logger.debug(f"⚠️ Gagal menghapus file {label_file}: {str(e)}")
                
                # Buat direktori jika belum ada
                img_dir.mkdir(parents=True, exist_ok=True)
                label_dir.mkdir(parents=True, exist_ok=True)
                
                if self.logger:
                    self.logger.debug(f"✅ Direktori {split_path} dibersihkan")
    
    def fix_dataset_structure(self, dataset_dir: Union[str, Path]) -> bool:
        """
        Memperbaiki struktur dataset agar sesuai dengan format YOLOv5 yang standar.
        
        Args:
            dataset_dir: Direktori dataset
            
        Returns:
            Boolean menunjukkan keberhasilan perbaikan
        """
        dataset_path = Path(dataset_dir)
        
        # Cek dan perbaiki struktur dataset
        is_fixed = False
        
        # Struktur yang diharapkan: train/images, train/labels, valid/images, valid/labels, test/images, test/labels
        
        # Cek struktur yang sudah benar
        standard_structure = True
        for split in DEFAULT_SPLITS:
            if not (dataset_path / split / 'images').exists() or not (dataset_path / split / 'labels').exists():
                standard_structure = False
                break
                
        if standard_structure:
            if self.logger:
                self.logger.info("✅ Struktur dataset sudah sesuai standar YOLO")
            return True
            
        # Cek struktur alternatif 1: train/, valid/, test/ tanpa subdirektori images/labels
        alt_structure_1 = True
        for split in DEFAULT_SPLITS:
            if not (dataset_path / split).exists():
                alt_structure_1 = False
                break
                
        if alt_structure_1:
            if self.logger:
                self.logger.info("🔄 Menata ulang struktur dataset ke format standar YOLO")
                
            for split in DEFAULT_SPLITS:
                split_dir = dataset_path / split
                img_dir = split_dir / 'images'
                label_dir = split_dir / 'labels'
                
                # Buat direktori
                img_dir.mkdir(exist_ok=True)
                label_dir.mkdir(exist_ok=True)
                
                # Pindahkan file
                for file in split_dir.glob('*.jpg'):
                    shutil.move(file, img_dir / file.name)
                    
                for file in split_dir.glob('*.txt'):
                    # Skip file teks yang bukan label
                    if file.stem.lower() in ['readme', 'classes', 'data']:
                        continue
                    shutil.move(file, label_dir / file.name)
                    
            is_fixed = True
            
        # Cek struktur alternatif 2: images/ dan labels/ di root
        elif (dataset_path / 'images').exists() and (dataset_path / 'labels').exists():
            if self.logger:
                self.logger.info("🔄 Memigrasikan struktur flat ke struktur split")
                
            # Buat direktori split
            for split in DEFAULT_SPLITS:
                os.makedirs(dataset_path / split / 'images', exist_ok=True)
                os.makedirs(dataset_path / split / 'labels', exist_ok=True)
                
            # Untuk kasus ini, tempatkan semua file di 'train'
            for file in (dataset_path / 'images').glob('*.jpg'):
                shutil.copy2(file, dataset_path / 'train' / 'images' / file.name)
                
            for file in (dataset_path / 'labels').glob('*.txt'):
                # Skip file teks yang bukan label
                if file.stem.lower() in ['readme', 'classes', 'data']:
                    continue
                shutil.copy2(file, dataset_path / 'train' / 'labels' / file.name)
                
            is_fixed = True
            
        else:
            if self.logger:
                self.logger.warning("⚠️ Tidak dapat mengenali struktur dataset")
            return False
            
        if is_fixed and self.logger:
            self.logger.success("✅ Struktur dataset berhasil diperbaiki")
            
        return is_fixed
    
    def copy_dataset_to_data_dir(
        self, 
        source_dir: Union[str, Path], 
        data_dir: Union[str, Path],
        clear_target: bool = True
    ) -> Dict[str, int]:
        """
        Salin dataset dari direktori sumber ke direktori data.
        
        Args:
            source_dir: Direktori sumber
            data_dir: Direktori data
            clear_target: Hapus file yang sudah ada di direktori target
            
        Returns:
            Dictionary berisi statistik penyalinan
        """
        return self.export_to_local(source_dir, data_dir, True, self.num_workers, clear_target)