
"""
File: smartcash/dataset/services/downloader/file_processor.py
Deskripsi: Updated file processor dengan Drive storage dan progress callback
"""

import os, shutil, zipfile
from pathlib import Path
from typing import Dict, Union, Optional, Callable
from concurrent.futures import ThreadPoolExecutor
import time

from smartcash.common.logger import get_logger
from smartcash.common.environment import get_environment_manager
from smartcash.dataset.utils.dataset_constants import DEFAULT_SPLITS
from smartcash.common.io import ensure_dir, copy_files, extract_zip as wrapper_extract_zip

class DownloadFileProcessor:
    """File processor dengan Drive storage dan progress callback."""
    
    def __init__(self, output_dir: str, config: Dict, logger=None, num_workers: int = 4, observer_manager=None):
        self.config = config
        self.logger = logger or get_logger()
        self.num_workers = num_workers
        self.observer_manager = observer_manager
        self._progress_callback: Optional[Callable] = None
        
        # Environment manager
        self.env_manager = get_environment_manager()
        
        # Setup paths dengan Drive priority
        self.output_dir = self._setup_drive_path(output_dir)
        self.data_dir = self.output_dir.parent
    
    def set_progress_callback(self, callback: Callable[[str, int, int, str], None]) -> None:
        """Set progress callback."""
        self._progress_callback = callback
    
    def _notify_progress(self, step: str, current: int, total: int, message: str) -> None:
        """Notify progress via callback."""
        if self._progress_callback:
            try:
                self._progress_callback(step, current, total, message)
            except Exception:
                pass
    
    def _setup_drive_path(self, output_dir: str) -> Path:
        """Setup output path dengan Drive priority."""
        if self.env_manager.is_colab and self.env_manager.is_drive_mounted:
            # Drive path
            drive_path = self.env_manager.drive_path / 'downloads' / Path(output_dir).name
            drive_path.mkdir(parents=True, exist_ok=True)
            
            # Setup symlink ke Colab
            colab_path = Path('/content') / Path(output_dir).name
            if colab_path.exists():
                if colab_path.is_symlink():
                    if colab_path.resolve() != drive_path.resolve():
                        colab_path.unlink()
                        colab_path.symlink_to(drive_path)
                else:
                    shutil.rmtree(colab_path, ignore_errors=True)
                    colab_path.symlink_to(drive_path)
            else:
                colab_path.symlink_to(drive_path)
            
            return drive_path
        else:
            # Local path
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            return output_path
    
    def process_zip_file(self, zip_path: Union[str, Path], output_dir: Union[str, Path], 
                        extract_only: bool = False, remove_zip: bool = False, 
                        show_progress: bool = True) -> Dict[str, Any]:
        """Process ZIP dengan Drive storage dan progress callback."""
        start_time = time.time()
        zip_path, output_path = Path(zip_path), self._setup_drive_path(str(output_dir))
        
        self.logger.info(f"📦 Memproses ZIP: {zip_path} → {output_path}")
        self._notify_progress("zip_process", 0, 100, "Memulai proses ZIP")
        
        if not self._validate_zip_file(zip_path):
            msg = f"ZIP tidak valid: {zip_path}"
            self.logger.error(f"❌ {msg}")
            return {"status": "error", "message": msg}
        
        # Setup temp directory
        tmp_extract_dir = output_path.with_name(f"{output_path.name}_extract_temp")
        self._safe_remove_dir(tmp_extract_dir)
        ensure_dir(tmp_extract_dir)
        
        try:
            # Extract
            self._notify_progress("zip_process", 20, 100, "Mengekstrak ZIP")
            extraction_result = self._extract_zip_with_callback(zip_path, tmp_extract_dir, remove_zip)
            
            if extraction_result.get("errors", 0) > 0:
                raise ValueError("Error saat ekstraksi")
            
            # Structure fix jika diperlukan
            if not extract_only:
                self._notify_progress("zip_process", 60, 100, "Menyesuaikan struktur")
                self._fix_dataset_structure(tmp_extract_dir)
            
            # Move ke final location (Drive)
            self._notify_progress("zip_process", 80, 100, f"Memindahkan ke {output_path}")
            self._safe_remove_dir(output_path)
            shutil.move(str(tmp_extract_dir), str(output_path))
            
            # Stats
            stats = self._get_dataset_stats(output_path)
            elapsed_time = time.time() - start_time
            
            self._notify_progress("zip_process", 100, 100, f"Proses ZIP selesai: {stats.get('total_images', 0)} gambar")
            
            self.logger.success(
                f"✅ ZIP proses selesai ({elapsed_time:.1f}s)\n"
                f"   • Gambar: {stats.get('total_images', 0)}\n"
                f"   • Output: {output_path}"
            )
            
            return {
                "status": "success", "output_dir": str(output_path),
                "file_count": extraction_result.get('extracted', 0),
                "stats": stats, "duration": elapsed_time
            }
            
        except Exception as e:
            self.logger.error(f"❌ ZIP proses gagal: {str(e)}")
            return {"status": "error", "message": str(e)}
        finally:
            self._safe_remove_dir(tmp_extract_dir)
    
    def _extract_zip_with_callback(self, zip_path: Path, extract_dir: Path, remove_zip: bool) -> Dict[str, int]:
        """Extract ZIP dengan progress callback."""
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                files = zip_ref.infolist()
                total_files = len(files)
                extracted = 0
                
                for i, file in enumerate(files):
                    zip_ref.extract(file, extract_dir)
                    extracted += 1
                    
                    # Progress callback
                    if i % max(1, total_files // 10) == 0:
                        progress = int((i / total_files) * 100)
                        self._notify_progress("extract", progress, 100, f"Ekstrak: {i}/{total_files}")
                
                if remove_zip:
                    zip_path.unlink()
                
                return {"extracted": extracted, "errors": 0}
                
        except Exception as e:
            self.logger.error(f"❌ Extract error: {str(e)}")
            return {"extracted": 0, "errors": 1}
    
    def _validate_zip_file(self, zip_path: Path) -> bool:
        """Validate ZIP file."""
        try:
            return zip_path.is_file() and zipfile.is_zipfile(zip_path)
        except Exception:
            return False
    
    def _fix_dataset_structure(self, dataset_dir: Path) -> bool:
        """Fix dataset structure untuk YOLO format."""
        if self._is_valid_yolo_structure(dataset_dir):
            return True
        
        # Fix flat structure ke split structure
        if (dataset_dir / 'images').exists() and (dataset_dir / 'labels').exists():
            train_img_dir = dataset_dir / 'train' / 'images'
            train_label_dir = dataset_dir / 'train' / 'labels'
            ensure_dir(train_img_dir)
            ensure_dir(train_label_dir)
            
            # Move files
            for img_file in (dataset_dir / 'images').glob('*.*'):
                shutil.move(str(img_file), str(train_img_dir))
            
            for label_file in (dataset_dir / 'labels').glob('*.txt'):
                if label_file.stem.lower() not in {'readme', 'classes', 'data'}:
                    shutil.move(str(label_file), str(train_label_dir))
            
            # Remove empty dirs
            shutil.rmtree(dataset_dir / 'images', ignore_errors=True)
            shutil.rmtree(dataset_dir / 'labels', ignore_errors=True)
            
            return True
        
        return False
    
    def _is_valid_yolo_structure(self, dataset_path: Path) -> bool:
        """Check valid YOLO structure."""
        return (dataset_path / 'train' / 'images').exists() and (dataset_path / 'train' / 'labels').exists()
    
    def _safe_remove_dir(self, dir_path: Path) -> None:
        """Safe directory removal."""
        if not dir_path.exists():
            return
        
        try:
            if dir_path.is_symlink():
                dir_path.unlink()
            else:
                shutil.rmtree(dir_path, ignore_errors=True)
        except Exception as e:
            self.logger.warning(f"⚠️ Gagal hapus {dir_path}: {str(e)}")
    
    def _get_dataset_stats(self, dataset_dir: Path) -> Dict[str, Any]:
        """Get dataset statistics."""
        stats = {'total_images': 0, 'total_labels': 0}
        
        for split in DEFAULT_SPLITS:
            images_dir = dataset_dir / split / 'images'
            if images_dir.exists():
                img_count = len(list(images_dir.glob('*.*')))
                stats['total_images'] += img_count
                stats[f'{split}_images'] = img_count
        
        return stats