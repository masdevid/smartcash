"""
File: smartcash/handlers/dataset/core/download_core/file_processor.py
Author: Alfrida Sabar
Deskripsi: Komponen untuk memproses file dataset
"""

import os
import shutil
import zipfile
from pathlib import Path
from typing import Dict, Union
from concurrent.futures import ThreadPoolExecutor
from tqdm.auto import tqdm

from smartcash.utils.dataset.dataset_utils import DEFAULT_SPLITS, IMG_EXTENSIONS


class FileProcessor:
    """Komponen untuk pemrosesan file dataset."""
    
    def __init__(self, logger=None, num_workers: int = 4):
        self.logger = logger
        self.num_workers = num_workers
    
    def clean_existing_download(self, output_dir: Union[str, Path]) -> None:
        output_path = Path(output_dir)
        if output_path.exists():
            self.logger and self.logger.info(f"🧹 Membersihkan direktori lama: {output_path}")
            try:
                shutil.rmtree(output_path)
                self.logger and self.logger.info(f"✅ Direktori lama berhasil dihapus")
            except Exception as e:
                self.logger and self.logger.warning(f"⚠️ Gagal menghapus direktori: {str(e)}")
    
    def extract_zip(self, zip_path: Union[str, Path], output_dir: Union[str, Path], remove_zip: bool = False, show_progress: bool = True) -> None:
        zip_path, output_dir = Path(zip_path), Path(output_dir)
        if not zipfile.is_zipfile(zip_path):
            self.logger and self.logger.error(f"❌ File bukan zip yang valid: {zip_path}")
            raise ValueError(f"File bukan zip yang valid: {zip_path}")
            
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            if show_progress:
                files = zip_ref.infolist()
                with tqdm(total=sum(f.file_size for f in files), unit='B', unit_scale=True, desc=f"Ekstraksi {zip_path.name}") as pbar:
                    for file in files:
                        zip_ref.extract(file, output_dir)
                        pbar.update(file.file_size)
            else:
                zip_ref.extractall(output_dir)
        
        if remove_zip and zip_path.exists():
            try:
                zip_path.unlink()
                self.logger and self.logger.info(f"🗑️ File zip dihapus: {zip_path}")
            except Exception as e:
                self.logger and self.logger.warning(f"⚠️ Gagal menghapus file zip: {str(e)}")
        self.logger and self.logger.success(f"✅ Ekstraksi selesai: {output_dir}")
    
    def export_to_local(self, source_dir: Union[str, Path], target_dir: Union[str, Path], show_progress: bool = True, num_workers: int = None) -> Dict[str, int]:
        source_path, target_path, workers = Path(source_dir), Path(target_dir), num_workers or self.num_workers
        self.logger and self.logger.info(f"📤 Mengexport dataset dari {source_path} ke {target_path}")
        stats = {'copied': 0, 'errors': 0}
        
        for split in DEFAULT_SPLITS:
            for subdir in ('images', 'labels'):
                (target_path / split / subdir).mkdir(parents=True, exist_ok=True)
        
        copy_tasks = []
        for split in DEFAULT_SPLITS:
            src_split = source_path if not (source_path / split).exists() else source_path / split
            src_img_dir = next((d for d in [src_split / 'images', src_split / 'train'] if d.exists()), src_split)
            src_label_dir = next((d for d in [src_split / 'labels', src_split / 'train'] if d.exists()), src_split)
            dst_img_dir, dst_label_dir = target_path / split / 'images', target_path / split / 'labels'
            
            for img_path in src_img_dir.glob('*'):
                if img_path.suffix.lower() in IMG_EXTENSIONS and (label_path := src_label_dir / f"{img_path.stem}.txt").exists():
                    copy_tasks.append((img_path, label_path, dst_img_dir / img_path.name, dst_label_dir / label_path.name))
        
        pbar = tqdm(total=len(copy_tasks), desc="Export Dataset") if show_progress else None
        
        def copy_file_worker(src_img, src_label, dst_img, dst_label):
            try:
                dst_img.exists() or shutil.copy2(src_img, dst_img)
                dst_label.exists() or shutil.copy2(src_label, dst_label)
                return True
            except Exception:
                return False
        
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(copy_file_worker, *task) for task in copy_tasks]
            for future in futures:
                stats['copied' if future.result() else 'errors'] += 1
                pbar and pbar.update(1)
        
        pbar and pbar.close()
        self.logger and self.logger.success(f"✅ Export selesai: {stats['copied']} file disalin, {stats['errors']} error")
        return stats