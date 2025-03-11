"""
File: smartcash/handlers/dataset/core/file_processor.py
Author: Alfrida Sabar
Deskripsi: Komponen sederhana untuk memproses file dataset
"""

import os
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List, Union, Tuple
from concurrent.futures import ThreadPoolExecutor
from tqdm.auto import tqdm

from smartcash.utils.dataset.dataset_utils import DEFAULT_SPLITS


class FileProcessor:
    """Komponen untuk pemrosesan file dataset."""
    
    def __init__(self, logger=None, num_workers: int = 4):
        """Inisialisasi FileProcessor."""
        self.logger = logger
        self.num_workers = num_workers
    
    def clean_existing_download(self, output_dir: Union[str, Path]) -> None:
        """Bersihkan download sebelumnya jika ada."""
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
        """Ekstrak file zip ke direktori output."""
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
                for file in tqdm(files, desc=f"Ekstraksi {zip_path.name}"):
                    zip_ref.extract(file, output_dir)
            else:
                zip_ref.extractall(output_dir)

        if remove_zip:
            try:
                zip_path.unlink()
            except Exception:
                pass
        
        if self.logger:
            self.logger.success(f"✅ Ekstraksi selesai: {output_dir}")
    
    def export_to_local(
        self,
        source_dir: Union[str, Path],
        target_dir: Union[str, Path],
        show_progress: bool = True,
        num_workers: int = None
    ) -> Dict[str, int]:
        """Export dataset dari struktur Roboflow ke struktur lokal standar."""
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
            except Exception:
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
    
    def fix_dataset_structure(self, dataset_dir: Union[str, Path]) -> bool:
        """Perbaiki struktur dataset jika tidak sesuai format YOLOv5."""
        dataset_path = Path(dataset_dir)
        
        # Cek apakah sudah memiliki struktur yang benar
        is_valid = True
        for split in DEFAULT_SPLITS:
            if not (dataset_path / split / 'images').exists() or not (dataset_path / split / 'labels').exists():
                is_valid = False
                break
                
        if is_valid:
            return True
            
        # Coba perbaiki struktur
        if self.logger:
            self.logger.info(f"🔧 Memperbaiki struktur dataset di {dataset_path}")
            
        # Temukan struktur yang ada
        if (dataset_path / 'train').exists() and (dataset_path / 'valid').exists():
            # Ada direktori train/valid tapi mungkin tanpa subdir images/labels
            for split in DEFAULT_SPLITS:
                split_dir = dataset_path / split
                if split_dir.exists():
                    # Cek apakah ada direktori images/labels
                    if not (split_dir / 'images').exists():
                        os.makedirs(split_dir / 'images', exist_ok=True)
                        # Pindahkan file gambar ke direktori images
                        for ext in ['jpg', 'jpeg', 'png']:
                            for img_file in split_dir.glob(f'*.{ext}'):
                                shutil.move(img_file, split_dir / 'images' / img_file.name)
                    
                    if not (split_dir / 'labels').exists():
                        os.makedirs(split_dir / 'labels', exist_ok=True)
                        # Pindahkan file label ke direktori labels
                        for label_file in split_dir.glob('*.txt'):
                            shutil.move(label_file, split_dir / 'labels' / label_file.name)
        elif (dataset_path / 'images').exists() and (dataset_path / 'labels').exists():
            # Ada direktori root images/labels tanpa split
            # Buat struktur split
            train_dir = dataset_path / 'train'
            valid_dir = dataset_path / 'valid'
            test_dir = dataset_path / 'test'
            
            for split_dir in [train_dir, valid_dir, test_dir]:
                os.makedirs(split_dir / 'images', exist_ok=True)
                os.makedirs(split_dir / 'labels', exist_ok=True)
            
            # Pindahkan semua file ke train
            for img_file in (dataset_path / 'images').glob('*.*'):
                shutil.copy2(img_file, train_dir / 'images' / img_file.name)
                
            for label_file in (dataset_path / 'labels').glob('*.txt'):
                shutil.copy2(label_file, train_dir / 'labels' / label_file.name)
        else:
            # Struktur tidak dikenal
            if self.logger:
                self.logger.warning(f"⚠️ Struktur dataset tidak dikenali")
            return False
            
        if self.logger:
            self.logger.success(f"✅ Struktur dataset diperbaiki")
        return True
        
    def copy_dataset_to_data_dir(self, src_dir: Union[str, Path], dst_dir: Union[str, Path]) -> Dict[str, int]:
        """Salin dataset ke direktori data utama."""
        src_path = Path(src_dir)
        dst_path = Path(dst_dir)
        
        if src_path == dst_path:
            return {'copied': 0, 'errors': 0}
            
        if self.logger:
            self.logger.info(f"🔄 Menyalin dataset dari {src_path} ke {dst_path}")
            
        stats = {'copied': 0, 'errors': 0}
        
        # Salin untuk setiap split
        for split in DEFAULT_SPLITS:
            src_split = src_path / split
            dst_split = dst_path / split
            
            if not src_split.exists():
                continue
                
            # Buat direktori target
            os.makedirs(dst_split / 'images', exist_ok=True)
            os.makedirs(dst_split / 'labels', exist_ok=True)
            
            # Salin gambar
            for img_file in (src_split / 'images').glob('*.*'):
                try:
                    shutil.copy2(img_file, dst_split / 'images' / img_file.name)
                    stats['copied'] += 1
                except Exception:
                    stats['errors'] += 1
                    
            # Salin label
            for label_file in (src_split / 'labels').glob('*.txt'):
                try:
                    shutil.copy2(label_file, dst_split / 'labels' / label_file.name)
                    stats['copied'] += 1
                except Exception:
                    stats['errors'] += 1
                    
        if self.logger:
            self.logger.success(f"✅ Dataset disalin: {stats['copied']} file")
            
        return stats