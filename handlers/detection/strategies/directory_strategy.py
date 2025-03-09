# File: smartcash/handlers/detection/strategies/directory_strategy.py
# Author: Alfrida Sabar
# Deskripsi: Strategi deteksi untuk direktori berisi gambar

import os
import time
from pathlib import Path
from typing import Dict, Any, List, Union, Optional
import torch
from tqdm import tqdm

from smartcash.handlers.detection.strategies.base_strategy import BaseDetectionStrategy
from smartcash.exceptions.base import DataError

class DirectoryDetectionStrategy(BaseDetectionStrategy):
    """
    Strategi untuk deteksi objek pada direktori berisi gambar.
    Mendukung proses batch dan penanganan error per gambar.
    """
    
    def detect(
        self,
        source: Union[str, Path],
        conf_threshold: Optional[float] = None,
        batch_size: int = 1,
        recursive: bool = False,
        visualize: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Deteksi objek dari direktori berisi gambar.
        
        Args:
            source: Path ke direktori
            conf_threshold: Threshold konfidiensi (opsional)
            batch_size: Ukuran batch untuk proses paralel
            recursive: Flag untuk mencari gambar secara rekursif
            visualize: Flag untuk visualisasi hasil (default: True)
            **kwargs: Parameter tambahan
            
        Returns:
            Dictionary hasil deteksi
        """
        # Tentukan path direktori
        if isinstance(source, str):
            source = Path(source)
            
        # Validasi
        if not source.exists() or not source.is_dir():
            raise DataError(f"Direktori tidak ditemukan: {source}")
            
        # Cari semua file gambar dalam direktori
        image_files = self._find_image_files(source, recursive)
        
        if not image_files:
            self.logger.warning(f"⚠️ Tidak ada file gambar ditemukan di direktori: {source}")
            return {
                'source': str(source),
                'total_images': 0,
                'processed_images': 0,
                'results': [],
                'errors': []
            }
            
        # Inisialisasi hasil
        results = {
            'source': str(source),
            'total_images': len(image_files),
            'processed_images': 0,
            'results': [],
            'errors': [],
            'detections_by_layer': {}
        }
        
        # Notifikasi start
        self.notify_observers('start', {
            'source': str(source),
            'total_images': len(image_files)
        })
        
        start_time = time.time()
        self.logger.info(f"🔍 Mendeteksi objek dari {len(image_files)} gambar di direktori: {source}")
        
        # Proses setiap gambar
        with tqdm(image_files, desc="📷 Memproses gambar") as pbar:
            for img_path in pbar:
                try:
                    # Proses satu gambar dengan ImageDetectionStrategy
                    img_result = self._process_single_image(
                        img_path, 
                        conf_threshold=conf_threshold, 
                        visualize=visualize,
                        **kwargs
                    )
                    
                    # Tambahkan ke hasil
                    results['results'].append(img_result)
                    results['processed_images'] += 1
                    
                    # Update detections_by_layer
                    for layer, detections in img_result.get('detections_by_layer', {}).items():
                        if layer not in results['detections_by_layer']:
                            results['detections_by_layer'][layer] = []
                        results['detections_by_layer'][layer].extend(detections)
                    
                    # Update progress
                    self.notify_observers('progress', {
                        'current': results['processed_images'],
                        'total': results['total_images'],
                        'latest_result': img_result
                    })
                    
                except Exception as e:
                    # Log error untuk gambar ini, tapi lanjutkan proses
                    self.logger.error(f"❌ Error saat memproses {img_path}: {str(e)}")
                    results['errors'].append({
                        'source': str(img_path),
                        'error': str(e)
                    })
                
                # Update tqdm description
                pbar.set_description(
                    f"📷 Memproses gambar ({results['processed_images']}/{results['total_images']})"
                )
        
        # Hitung statistik akhir
        results['execution_time'] = time.time() - start_time
        results['success_rate'] = results['processed_images'] / results['total_images'] if results['total_images'] > 0 else 0
        results['num_errors'] = len(results['errors'])
        
        total_detections = sum(len(r.get('detections', [])) for r in results['results'])
        results['total_detections'] = total_detections
        
        # Log hasil
        self.logger.info(
            f"✅ Deteksi selesai: {results['processed_images']}/{results['total_images']} gambar sukses, "
            f"{total_detections} objek terdeteksi ({results['execution_time']:.3f} detik)"
        )
        
        # Notifikasi complete
        self.notify_observers('complete', results)
        
        return results
    
    def _find_image_files(self, directory: Path, recursive: bool = False) -> List[Path]:
        """
        Cari semua file gambar dalam direktori.
        
        Args:
            directory: Path direktori
            recursive: Flag untuk pencarian rekursif
            
        Returns:
            List path file gambar
        """
        # Extensions gambar yang didukung
        extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        
        image_files = []
        
        # Tentukan fungsi glob berdasarkan recursive
        glob_func = directory.rglob if recursive else directory.glob
        
        # Cari file untuk setiap extension
        for ext in extensions:
            image_files.extend(list(glob_func(f"*{ext}")))
            image_files.extend(list(glob_func(f"*{ext.upper()}")))
        
        # Sort agar urutan konsisten
        image_files.sort()
        
        return image_files
    
    def _process_single_image(
        self, 
        img_path: Path, 
        conf_threshold: Optional[float] = None,
        visualize: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Proses satu gambar.
        
        Args:
            img_path: Path gambar
            conf_threshold: Threshold konfidiensi
            visualize: Flag untuk visualisasi
            **kwargs: Parameter tambahan
            
        Returns:
            Dictionary hasil deteksi
        """
        # Preprocessing
        preprocess_result = self.preprocessor.process(img_path)
        img_tensor = preprocess_result['tensor']
        original_shape = preprocess_result['original_shape']
        original_image = preprocess_result.get('original_image')
        
        # Deteksi
        detection_result = self.detector.detect(
            img_tensor, 
            conf_thres=conf_threshold,
            **kwargs
        )
        
        # Postprocessing
        result = self.postprocessor.process(
            detection_result,
            original_shape=original_shape
        )
        
        # Tambahkan informasi source
        result['source'] = str(img_path)
        
        # Visualisasi jika diminta
        if visualize and original_image is not None:
            output_path = self.output_manager.save_visualization(
                source=img_path,
                image=original_image,
                detections=result['detections'],
                **kwargs
            )
            result['visualization_path'] = output_path
        
        # Simpan hasil ke JSON
        output_paths = self.output_manager.save_results(img_path, result, **kwargs)
        result['output_paths'] = output_paths
        
        return result