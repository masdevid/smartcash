"""
File: smartcash/detection/handlers/batch_handler.py
Deskripsi: Handler untuk deteksi objek pada batch/kumpulan gambar.
"""

import os
import glob
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from smartcash.common.utils import generate_unique_id
from smartcash.common.logger import SmartCashLogger, get_logger
from smartcash.common.constants import (IMAGE_EXTENSIONS, MAX_BATCH_SIZE)
from smartcash.common.types import Detection
from smartcash.dataset.utils import (
    ensure_dir
)


class BatchHandler:
    """Handler untuk mengelola proses deteksi pada batch gambar"""
    
    def __init__(self, 
                 detection_handler, 
                 num_workers: int = 4,
                 batch_size: int = 16,
                 max_batch_size: int = MAX_BATCH_SIZE,
                 logger: Optional[SmartCashLogger] = None):
        """
        Inisialisasi Batch Handler
        
        Args:
            detection_handler: Handler untuk deteksi gambar tunggal
            num_workers: Jumlah worker untuk pemrosesan paralel
            batch_size: Ukuran batch untuk batch processing
            max_batch_size: Ukuran batch maksimum
            logger: Logger untuk mencatat aktivitas (opsional)
        """
        self.detection_handler = detection_handler
        self.num_workers = num_workers
        self.batch_size = min(batch_size, max_batch_size)
        self.logger = logger or get_logger("BatchHandler")
    
    def detect_directory(self, 
                        input_dir: str, 
                        output_dir: Optional[str] = None,
                        extensions: List[str] = None,
                        recursive: bool = False,
                        conf_threshold: float = 0.25,
                        iou_threshold: float = 0.45,
                        save_results: bool = True,
                        save_visualizations: bool = True,
                        result_format: str = 'json') -> Dict:
        """
        Deteksi objek pada semua gambar di direktori
        
        Args:
            input_dir: Direktori input gambar
            output_dir: Direktori output hasil (opsional)
            extensions: List ekstensi file gambar yang akan diproses
            recursive: Flag untuk mencari gambar secara rekursif
            conf_threshold: Threshold confidence minimum
            iou_threshold: Threshold IoU untuk NMS
            save_results: Flag untuk menyimpan hasil deteksi
            save_visualizations: Flag untuk menyimpan visualisasi
            result_format: Format hasil deteksi ('json', 'txt', 'csv')
            
        Returns:
            Dictionary berisi hasil deteksi per gambar
        """
        # Default ekstensi file yang didukung jika tidak ditentukan
        if extensions is None:
            extensions = IMAGE_EXTENSIONS
        
        # Buat pola pencarian berdasarkan recursive flag
        search_pattern = os.path.join(input_dir, '**', f'*.{{' + ','.join(extensions) + '}') if recursive else \
                         os.path.join(input_dir, f'*.{{' + ','.join(extensions) + '}')
                         
        # Dapatkan semua file gambar
        image_files = []
        for ext in extensions:
            if recursive:
                image_files.extend(glob.glob(os.path.join(input_dir, f'**/*.{ext}'), recursive=True))
            else:
                image_files.extend(glob.glob(os.path.join(input_dir, f'*.{ext}')))
        
        # Periksa jika ditemukan file gambar
        if not image_files:
            self.logger.warning(f"⚠️ Tidak ditemukan file gambar di {input_dir} dengan ekstensi {extensions}")
            return {"status": "error", "message": "Tidak ditemukan file gambar"}
        
        # Proses batch gambar
        self.logger.info(f"🔄 Memproses {len(image_files)} gambar dari {input_dir}")
        
        # Buat output_dir jika tidak ada dan diperlukan
        if save_results and output_dir:
            ensure_dir(output_dir)
        
        # Lakukan deteksi per file dengan progress bar
        results = {}
        
        with tqdm(total=len(image_files), desc="📊 Memproses batch gambar", unit="gambar") as progress:
            # Fungsi worker untuk ThreadPoolExecutor
            def process_single_image(img_path):
                try:
                    # Generate nama file output jika perlu menyimpan hasil
                    if save_results and output_dir:
                        # Buat nama file relatif terhadap input_dir
                        rel_path = os.path.relpath(img_path, input_dir)
                        # Hapus ekstensi dan gunakan sebagai basis nama file output
                        base_name = os.path.splitext(rel_path)[0]
                        # Buat path output lengkap
                        output_path = os.path.join(output_dir, base_name)
                        # Pastikan direktori output ada
                        ensure_dir(os.path.dirname(output_path))
                    else:
                        output_path = None
                    
                    # Detect objek pada gambar
                    detections = self.detection_handler.detect(
                        image=img_path,
                        conf_threshold=conf_threshold,
                        iou_threshold=iou_threshold,
                        apply_postprocessing=True,
                        return_visualization=False
                    )
                    
                    # Simpan hasil jika diperlukan
                    result_info = {
                        "detections": detections,
                        "count": len(detections)
                    }
                    
                    if save_results and output_path:
                        # Simpan hasil deteksi
                        result_paths = self.detection_handler.save_result(
                            detections=detections,
                            output_path=output_path,
                            image=img_path,
                            save_visualization=save_visualizations,
                            format=result_format
                        )
                        result_info["file_paths"] = result_paths
                    
                    progress.update(1)
                    return img_path, result_info
                    
                except Exception as e:
                    self.logger.error(f"❌ Error pada {img_path}: {str(e)}")
                    progress.update(1)
                    return img_path, {"status": "error", "message": str(e)}
            
            # Gunakan ThreadPoolExecutor untuk memproses gambar secara paralel
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = {executor.submit(process_single_image, img_path): img_path for img_path in image_files}
                
                for future in futures:
                    img_path, result = future.result()
                    results[img_path] = result
        
        # Hitung ringkasan hasil
        success_count = sum(1 for r in results.values() if "status" not in r or r["status"] != "error")
        error_count = len(results) - success_count
        detection_count = sum(r.get("count", 0) for r in results.values() if "count" in r)
        
        summary = {
            "processed_files": len(image_files),
            "success_count": success_count,
            "error_count": error_count,
            "total_detections": detection_count,
            "details": results
        }
        
        self.logger.info(f"✅ Selesai memproses batch: {success_count} berhasil, {error_count} gagal, {detection_count} objek terdeteksi")
        return summary
    
    def detect_zip(self, 
                  zip_path: str, 
                  output_dir: str,
                  conf_threshold: float = 0.25,
                  iou_threshold: float = 0.45,
                  extensions: List[str] = None,
                  save_extracted: bool = False) -> Dict:
        """
        Deteksi objek pada gambar di dalam file ZIP
        
        Args:
            zip_path: Path ke file ZIP
            output_dir: Direktori output hasil
            conf_threshold: Threshold confidence minimum
            iou_threshold: Threshold IoU untuk NMS
            extensions: List ekstensi file gambar yang akan diproses
            save_extracted: Flag untuk menyimpan file yang diekstrak
            
        Returns:
            Dictionary berisi hasil deteksi
        """
        import zipfile
        import tempfile
        import shutil
        
        if extensions is None:
            extensions = IMAGE_EXTENSIONS
            
        # Buat direktori sementara untuk ekstraksi
        with tempfile.TemporaryDirectory() as temp_dir:
            self.logger.info(f"📦 Mengekstrak {zip_path} ke direktori sementara")
            
            # Ekstrak file zip
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                for file in zip_ref.namelist():
                    # Hanya ekstrak file dengan ekstensi yang sesuai
                    if any(file.lower().endswith(f".{ext}") for ext in extensions):
                        zip_ref.extract(file, temp_dir)
            
            # Proses direktori hasil ekstraksi
            results = self.detect_directory(
                input_dir=temp_dir,
                output_dir=output_dir,
                extensions=extensions,
                recursive=True,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold
            )
            
            # Simpan file yang diekstrak jika diminta
            if save_extracted:
                extracted_dir = os.path.join(output_dir, "extracted")
                ensure_dir(extracted_dir)
                for root, _, files in os.walk(temp_dir):
                    for file in files:
                        if any(file.lower().endswith(f".{ext}") for ext in extensions):
                            rel_path = os.path.relpath(os.path.join(root, file), temp_dir)
                            dest_path = os.path.join(extracted_dir, rel_path)
                            ensure_dir(os.path.dirname(dest_path))
                            shutil.copy2(os.path.join(root, file), dest_path)
            
            # Tambahkan info zip ke hasil
            results["zip_info"] = {
                "filename": os.path.basename(zip_path),
                "extracted_files": sum(1 for _, r in results.get("details", {}).items() if "status" not in r)
            }
            
            return results