"""
File: smartcash/model/services/prediction/batch_processor_prediction_service.py
Deskripsi: Modul untuk layanan batch processing prediksi
"""

import time
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Union, Tuple, Optional, Any
from concurrent.futures import ThreadPoolExecutor
from tqdm.auto import tqdm

from smartcash.common.logger import get_logger
from smartcash.model.services.prediction.core_prediction_service import PredictionService


class BatchPredictionProcessor:
    """
    Processor untuk menjalankan prediksi pada dataset besar dengan optimasi throughput.
    Mendukung pemrosesan paralel dan tracking progres.
    """
    
    def __init__(
        self,
        prediction_service: PredictionService,
        output_dir: Optional[str] = None,
        num_workers: int = 4,
        batch_size: int = 16,
        logger = None
    ):
        """
        Inisialisasi batch prediction processor.
        
        Args:
            prediction_service: Instance PredictionService untuk inferensi
            output_dir: Direktori output untuk menyimpan hasil (opsional)
            num_workers: Jumlah worker thread untuk I/O
            batch_size: Ukuran batch untuk inferensi
            logger: Logger untuk pencatatan
        """
        self.prediction_service = prediction_service
        self.output_dir = Path(output_dir) if output_dir else Path("predictions")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.logger = logger or get_logger()
        
        self.logger.info(
            f"🔄 Batch prediction processor diinisialisasi:\n"
            f"   • Output dir: {self.output_dir}\n"
            f"   • Workers: {self.num_workers}\n"
            f"   • Batch size: {self.batch_size}"
        )
    
    def process_directory(
        self,
        input_dir: str,
        save_results: bool = True,
        save_annotated: bool = False,
        file_ext: Union[str, List[str]] = ['.jpg', '.jpeg', '.png'],
        recursive: bool = False
    ) -> Dict[str, Any]:
        """
        Proses semua gambar dalam direktori.
        
        Args:
            input_dir: Direktori berisi gambar
            save_results: Flag untuk menyimpan hasil
            save_annotated: Flag untuk menyimpan gambar dengan anotasi
            file_ext: Ekstensi file yang diproses
            recursive: Flag untuk rekursi subdirektori
            
        Returns:
            Dict hasil batch prediksi
        """
        # Normalize file_ext ke list
        if isinstance(file_ext, str):
            file_ext = [file_ext]
            
        # Temukan file yang sesuai
        all_files = []
        input_path = Path(input_dir)
        
        if recursive:
            for ext in file_ext:
                all_files.extend(list(input_path.glob(f"**/*{ext}")))
        else:
            for ext in file_ext:
                all_files.extend(list(input_path.glob(f"*{ext}")))
        
        if not all_files:
            self.logger.warning(f"⚠️ Tidak ada file gambar ditemukan di {input_dir}")
            return {'processed': 0, 'time_taken': 0, 'files': []}
        
        self.logger.info(f"🔍 Menemukan {len(all_files)} file di {input_dir}")
        
        # Jalankan batch processing
        return self.process_files(
            all_files, 
            save_results=save_results,
            save_annotated=save_annotated
        )
    
    def process_files(
        self,
        files: List[Union[str, Path]],
        save_results: bool = True,
        save_annotated: bool = False
    ) -> Dict[str, Any]:
        """
        Proses list file gambar.
        
        Args:
            files: List path file
            save_results: Flag untuk menyimpan hasil
            save_annotated: Flag untuk menyimpan gambar dengan anotasi
            
        Returns:
            Dict hasil batch prediksi
        """
        # Convert semua path ke string absolut
        file_paths = [str(Path(f).absolute()) for f in files]
        
        if not file_paths:
            return {'processed': 0, 'time_taken': 0, 'files': []}
        
        self.logger.info(f"🔄 Memproses {len(file_paths)} file...")
        
        # Tracking statistik
        start_time = time.time()
        processed = 0
        all_results = []
        
        # Bagi files menjadi batch untuk memory efficiency
        for batch_idx in range(0, len(file_paths), self.batch_size):
            batch_files = file_paths[batch_idx:batch_idx + self.batch_size]
            
            try:
                # Proses batch
                batch_results = self.prediction_service.predict_from_files(
                    batch_files, 
                    return_annotated=save_annotated
                )
                
                # Tambahkan ke hasil
                if 'predictions' in batch_results:
                    processed += len(batch_results['predictions'])
                    all_results.append(batch_results)
                    
                    # Simpan hasil jika diminta
                    if save_results:
                        self._save_batch_results(batch_results, batch_idx)
            except Exception as e:
                self.logger.error(f"❌ Error pada batch {batch_idx}: {str(e)}")
        
        # Kompilasi hasil
        total_time = time.time() - start_time
        
        self.logger.success(
            f"✅ Batch prediksi selesai:\n"
            f"   • File diproses: {processed}/{len(file_paths)}\n"
            f"   • Waktu: {total_time:.2f} detik\n"
            f"   • Throughput: {processed/max(1, total_time):.2f} gambar/detik"
        )
        
        return {
            'processed': processed,
            'time_taken': total_time,
            'throughput': processed / max(1, total_time),
            'files': file_paths,
            'results': all_results
        }
    
    def _save_batch_results(
        self, 
        batch_results: Dict[str, Any],
        batch_idx: int
    ) -> None:
        """
        Simpan hasil batch.
        
        Args:
            batch_results: Hasil prediksi
            batch_idx: Index batch
        """
        import json
        import cv2
        
        # Simpan prediksi ke JSON
        if 'predictions' in batch_results and 'paths' in batch_results:
            # Grupkan prediksi berdasarkan path
            for i, (path, preds) in enumerate(zip(batch_results['paths'], batch_results['predictions'])):
                try:
                    # Buat nama file
                    file_stem = Path(path).stem
                    json_path = self.output_dir / f"{file_stem}.json"
                    
                    # Simpan prediksi
                    with open(json_path, 'w') as f:
                        json.dump(preds, f, indent=2)
                    
                    # Simpan gambar dengan anotasi jika ada
                    if 'annotated_images' in batch_results and i < len(batch_results['annotated_images']):
                        annotated_img = batch_results['annotated_images'][i]
                        img_path = self.output_dir / f"{file_stem}_annotated.jpg"
                        
                        # Konversi dari RGB ke BGR untuk cv2
                        cv2.imwrite(str(img_path), cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))
                except Exception as e:
                    self.logger.warning(f"⚠️ Error menyimpan hasil {path}: {str(e)}")
    
    def run_and_save(
        self,
        input_source: Union[str, List[str]],
        output_filename: Optional[str] = None,
        save_annotated: bool = True
    ) -> str:
        """
        Jalankan prediksi dan simpan ke file hasil terkonsolidasi.
        
        Args:
            input_source: Path direktori atau list file
            output_filename: Nama file output (tanpa ekstensi)
            save_annotated: Flag untuk menyimpan gambar dengan anotasi
            
        Returns:
            Path file hasil
        """
        # Tentukan input files
        files = []
        if isinstance(input_source, str) and Path(input_source).is_dir():
            # Proses direktori
            results = self.process_directory(
                input_source,
                save_results=False,  # Simpan hasil konsolidasi di akhir
                save_annotated=save_annotated,
                recursive=True
            )
            files = results.get('files', [])
        else:
            # List file
            if isinstance(input_source, str):
                input_source = [input_source]
            files = [str(Path(f).absolute()) for f in input_source]
        
        if not files:
            self.logger.warning("⚠️ Tidak ada file untuk diproses")
            return ""
        
        # Jalankan prediksi
        results = self.process_files(
            files,
            save_results=False,  # Simpan hasil konsolidasi di akhir
            save_annotated=save_annotated
        )
        
        # Tentukan nama file output
        if output_filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_filename = f"predictions_{timestamp}"
        
        # Pastikan ekstensi .json
        if not output_filename.endswith('.json'):
            output_filename += '.json'
        
        output_path = self.output_dir / output_filename
        
        # Simpan hasil konsolidasi
        try:
            import json
            
            # Gabungkan semua prediksi
            consolidated = {
                'metadata': {
                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                    'file_count': len(files),
                    'processed_count': results['processed'],
                    'time_taken': results['time_taken']
                },
                'results': {}
            }
            
            # Gabungkan prediksi dari semua batch
            for batch_result in results.get('results', []):
                for i, (path, preds) in enumerate(zip(batch_result.get('paths', []), 
                                                    batch_result.get('predictions', []))):
                    # Gunakan nama file sebagai key
                    file_name = Path(path).name
                    consolidated['results'][file_name] = preds
            
            # Simpan ke file
            with open(output_path, 'w') as f:
                json.dump(consolidated, f, indent=2)
                
            self.logger.success(f"✅ Hasil disimpan ke {output_path}")
            return str(output_path)
        
        except Exception as e:
            self.logger.error(f"❌ Error menyimpan hasil: {str(e)}")
            return ""