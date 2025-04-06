"""
File: smartcash/detection/services/inference/batch_processor.py
Deskripsi: Processor untuk inferensi batch gambar.
"""

import os
import time
import glob
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from smartcash.common.logger import SmartCashLogger, get_logger
from smartcash.common import Detection, ImageType, format_time, get_timestamp, IMAGE_EXTENSIONS, ensure_dir  

class BatchProcessor:
    """Processor untuk inferensi batch gambar"""
    
    def __init__(self, 
                inference_service,
                output_dir: Optional[str] = None,
                num_workers: int = 4,
                batch_size: int = 16,
                logger: Optional[SmartCashLogger] = None):
        """
        Inisialisasi Batch Processor
        
        Args:
            inference_service: Layanan inferensi untuk memproses gambar
            output_dir: Direktori untuk output hasil (opsional)
            num_workers: Jumlah worker untuk pemrosesan paralel
            batch_size: Ukuran batch untuk pemrosesan gambar
            logger: Logger untuk mencatat aktivitas (opsional)
        """
        self.inference_service = inference_service
        self.output_dir = output_dir
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.logger = logger or get_logger("BatchProcessor")
    
    def process_directory(self, 
                         input_dir: str, 
                         output_dir: Optional[str] = None,
                         extensions: List[str] = None,
                         recursive: bool = False,
                         conf_threshold: float = 0.25,
                         iou_threshold: float = 0.45,
                         save_results: bool = True,
                         save_visualizations: bool = True,
                         result_format: str = 'json',
                         callback: Optional[Callable] = None) -> Dict:
        """
        Memproses direktori gambar
        
        Args:
            input_dir: Direktori input gambar
            output_dir: Direktori output hasil (override atribut kelas jika ditentukan)
            extensions: List ekstensi file gambar yang akan diproses
            recursive: Flag untuk mencari gambar secara rekursif
            conf_threshold: Threshold confidence minimum
            iou_threshold: Threshold IoU untuk NMS
            save_results: Flag untuk menyimpan hasil deteksi
            save_visualizations: Flag untuk menyimpan visualisasi
            result_format: Format hasil deteksi ('json', 'txt', 'csv')
            callback: Callback function untuk setiap gambar (opsional)
            
        Returns:
            Dictionary berisi hasil deteksi
        """
        # Setup direktori output
        final_output_dir = output_dir or self.output_dir
        if save_results and final_output_dir:
            ensure_dir(final_output_dir)
            
        # Default ekstensi file yang didukung jika tidak ditentukan
        if extensions is None:
            extensions = IMAGE_EXTENSIONS
            
        # Dapatkan daftar file gambar
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
            
        # Sortir file untuk konsistensi
        image_files.sort()
            
        # Log jumlah file yang ditemukan
        self.logger.info(f"🔍 Ditemukan {len(image_files)} file gambar di {input_dir}")
            
        # Proses gambar dalam batch
        results = {}
        start_time = time.time()
        processed_count = 0
        
        # Fungsi untuk memproses satu file
        def process_single_file(img_path):
            try:
                # Lakukan inferensi
                detections = self.inference_service.infer(
                    image=img_path,
                    conf_threshold=conf_threshold,
                    iou_threshold=iou_threshold
                )
                
                # Format hasil
                result = {
                    "status": "success",
                    "detections": detections,
                    "count": len(detections)
                }
                
                # Simpan hasil jika diperlukan
                if save_results and final_output_dir:
                    rel_path = os.path.relpath(img_path, input_dir)
                    base_name = os.path.splitext(rel_path)[0]
                    output_path = os.path.join(final_output_dir, base_name)
                    ensure_dir(os.path.dirname(output_path))
                    
                    result["output_files"] = self._save_result(
                        img_path=img_path,
                        detections=detections,
                        output_path=output_path,
                        save_visualization=save_visualizations,
                        format=result_format
                    )
                
                # Panggil callback jika ada
                if callback:
                    callback(img_path, result)
                    
                return img_path, result
                
            except Exception as e:
                self.logger.error(f"❌ Error saat memproses {img_path}: {str(e)}")
                return img_path, {"status": "error", "message": str(e)}
        
        # Gunakan ThreadPoolExecutor untuk memproses gambar secara paralel
        with tqdm(total=len(image_files), desc="🖼️ Memproses gambar", unit="gambar") as progress:
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                for i in range(0, len(image_files), self.batch_size):
                    batch = image_files[i:i+self.batch_size]
                    futures = {executor.submit(process_single_file, img_path): img_path for img_path in batch}
                    
                    for future in futures:
                        img_path, result = future.result()
                        results[img_path] = result
                        processed_count += 1
                        progress.update(1)
        
        # Hitung statistik
        total_time = time.time() - start_time
        success_count = sum(1 for r in results.values() if r.get("status") == "success")
        error_count = processed_count - success_count
        detection_count = sum(r.get("count", 0) for r in results.values() if r.get("status") == "success")
        
        avg_detections = detection_count / max(success_count, 1)
        avg_time_per_image = total_time / max(processed_count, 1)
        
        # Kompilasi statistik
        stats = {
            "total_files": len(image_files),
            "processed_files": processed_count,
            "success_count": success_count,
            "error_count": error_count,
            "total_detections": detection_count,
            "avg_detections_per_image": avg_detections,
            "total_time": total_time,
            "avg_time_per_image": avg_time_per_image,
            "throughput": processed_count / max(total_time, 0.001)
        }
        
        self.logger.info(f"✅ Selesai memproses {processed_count} file dalam {format_time(total_time)}: "
                       f"{success_count} berhasil, {error_count} error, {detection_count} objek terdeteksi "
                       f"({avg_detections:.1f} per gambar)")
        
        return {
            "status": "success",
            "stats": stats,
            "results": results
        }
    
    def process_batch(self, 
                     images: List[ImageType],
                     output_dir: Optional[str] = None,
                     filenames: Optional[List[str]] = None,
                     conf_threshold: float = 0.25,
                     iou_threshold: float = 0.45,
                     save_results: bool = True,
                     save_visualizations: bool = True,
                     result_format: str = 'json') -> Dict:
        """
        Memproses batch gambar yang sudah dimuat
        
        Args:
            images: List gambar (ndarray, path, atau PIL Image)
            output_dir: Direktori output hasil (override atribut kelas jika ditentukan)
            filenames: List nama file untuk output (opsional)
            conf_threshold: Threshold confidence minimum
            iou_threshold: Threshold IoU untuk NMS
            save_results: Flag untuk menyimpan hasil deteksi
            save_visualizations: Flag untuk menyimpan visualisasi
            result_format: Format hasil deteksi ('json', 'txt', 'csv')
            
        Returns:
            Dictionary berisi hasil deteksi
        """
        # Setup direktori output
        final_output_dir = output_dir or self.output_dir
        if save_results and final_output_dir:
            ensure_dir(final_output_dir)
        
        # Generate nama file jika tidak disediakan
        if filenames is None:
            timestamp = get_timestamp()
            filenames = [f"image_{timestamp}_{i}" for i in range(len(images))]
        
        # Pastikan panjang filenames sama dengan images
        if len(filenames) != len(images):
            filenames = [f"image_{i}" for i in range(len(images))]
            self.logger.warning(f"⚠️ Jumlah filenames tidak sama dengan jumlah gambar, menggunakan nama default")
        
        # Lakukan inferensi batch
        self.logger.info(f"🔄 Memproses batch {len(images)} gambar")
        start_time = time.time()
        
        # Gunakan batch inference jika tersedia
        try:
            batch_detections = self.inference_service.batch_infer(
                images=images,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold
            )
        except (AttributeError, NotImplementedError):
            # Fallback ke inferensi satu per satu
            self.logger.warning("⚠️ Batch inferensi tidak tersedia, menggunakan inferensi sekuensial")
            batch_detections = []
            for img in tqdm(images, desc="🖼️ Memproses gambar", unit="gambar"):
                detections = self.inference_service.infer(
                    image=img,
                    conf_threshold=conf_threshold,
                    iou_threshold=iou_threshold
                )
                batch_detections.append(detections)
        
        # Kompilasi hasil
        results = {}
        for i, (img, filename, detections) in enumerate(zip(images, filenames, batch_detections)):
            result = {
                "status": "success",
                "detections": detections,
                "count": len(detections)
            }
            
            # Simpan hasil jika diperlukan
            if save_results and final_output_dir:
                output_path = os.path.join(final_output_dir, filename)
                
                result["output_files"] = self._save_result(
                    img_path=img,
                    detections=detections,
                    output_path=output_path,
                    save_visualization=save_visualizations,
                    format=result_format
                )
            
            results[filename] = result
        
        # Hitung statistik
        total_time = time.time() - start_time
        detection_count = sum(len(dets) for dets in batch_detections)
        avg_detections = detection_count / max(len(images), 1)
        avg_time_per_image = total_time / max(len(images), 1)
        
        # Kompilasi statistik
        stats = {
            "total_images": len(images),
            "total_detections": detection_count,
            "avg_detections_per_image": avg_detections,
            "total_time": total_time,
            "avg_time_per_image": avg_time_per_image,
            "throughput": len(images) / max(total_time, 0.001)
        }
        
        self.logger.info(f"✅ Selesai memproses {len(images)} gambar dalam {format_time(total_time)}: "
                       f"{detection_count} objek terdeteksi ({avg_detections:.1f} per gambar)")
        
        return {
            "status": "success",
            "stats": stats,
            "results": results
        }
    
    def _save_result(self, 
                    img_path: ImageType, 
                    detections: List[Detection],
                    output_path: str,
                    save_visualization: bool = True,
                    format: str = 'json') -> Dict[str, str]:
        """
        Menyimpan hasil deteksi ke file
        
        Args:
            img_path: Path atau gambar
            detections: List hasil deteksi
            output_path: Path dasar untuk file output
            save_visualization: Flag untuk menyimpan visualisasi
            format: Format hasil deteksi ('json', 'txt', 'csv')
            
        Returns:
            Dictionary berisi path file hasil
        """
        result_files = {}
        
        # Simpan hasil deteksi dalam format yang diminta
        if format.lower() == 'json':
            import json
            data_path = f"{output_path}.json"
            with open(data_path, 'w') as f:
                json.dump([d.__dict__ for d in detections], f, indent=2)
            result_files['data'] = data_path
            
        elif format.lower() == 'txt':
            data_path = f"{output_path}.txt"
            with open(data_path, 'w') as f:
                for detection in detections:
                    f.write(f"{detection.class_id} {detection.confidence:.4f} "
                           f"{detection.bbox[0]:.4f} {detection.bbox[1]:.4f} "
                           f"{detection.bbox[2]:.4f} {detection.bbox[3]:.4f}\n")
            result_files['data'] = data_path
            
        elif format.lower() == 'csv':
            import csv
            data_path = f"{output_path}.csv"
            with open(data_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['class_id', 'class_name', 'confidence', 'x1', 'y1', 'x2', 'y2'])
                for d in detections:
                    writer.writerow([
                        d.class_id, 
                        d.class_name, 
                        f"{d.confidence:.4f}",
                        f"{d.bbox[0]:.4f}", 
                        f"{d.bbox[1]:.4f}", 
                        f"{d.bbox[2]:.4f}", 
                        f"{d.bbox[3]:.4f}"
                    ])
            result_files['data'] = data_path
        
        # Simpan visualisasi jika diminta
        if save_visualization:
            from PIL import Image
            
            viz_path = f"{output_path}_viz.jpg"
            
            # Visualisasikan deteksi
            visualization = self.inference_service.visualize(img_path, detections)
            
            # Simpan hasil visualisasi
            if isinstance(visualization, np.ndarray):
                Image.fromarray(visualization).save(viz_path)
                result_files['visualization'] = viz_path
        
        return result_files