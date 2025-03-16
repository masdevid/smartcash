"""
File: smartcash/dataset/services/downloader/file_processor.py
Deskripsi: Komponen untuk memproses file dataset seperti ekstraksi, konversi dan strukturisasi
"""

import os
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List, Union, Tuple
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
        num_workers: int = None
    ) -> Dict[str, int]:
        """
        Export dataset dari struktur Roboflow ke struktur lokal standar.
        
        Args:
            source_dir: Direktori sumber
            target_dir: Direktori target
            show_progress: Tampilkan progress bar
            num_workers: Jumlah worker untuk proses paralel
            
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