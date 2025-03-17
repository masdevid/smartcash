"""
File: smartcash/model/services/prediction/interface_prediction_service.py
Deskripsi: Modul interface untuk layanan prediksi
"""

import os
import time
import torch
import numpy as np
from pathlib import Path
import base64
from io import BytesIO
from typing import Dict, List, Union, Tuple, Optional, Any
from PIL import Image

from smartcash.common.logger import get_logger
from smartcash.model.services.prediction.core_prediction_service import PredictionService
from smartcash.model.services.prediction.batch_processor_prediction_service import BatchPredictionProcessor
from smartcash.common.exceptions import ModelInferenceError


class PredictionInterface:
    """
    Interface untuk layanan prediksi yang menyediakan metode sederhana dan user-friendly
    untuk berbagai kebutuhan prediksi, termasuk dari file, URL, dan base64.
    """
    
    def __init__(
        self,
        prediction_service: PredictionService,
        output_dir: Optional[str] = None,
        batch_size: int = 16,
        num_workers: int = 4,
        cache_results: bool = True,
        logger = None
    ):
        """
        Inisialisasi prediction interface.
        
        Args:
            prediction_service: Instance PredictionService
            output_dir: Direktori output untuk hasil
            batch_size: Ukuran batch untuk inferensi
            num_workers: Jumlah worker thread
            cache_results: Flag untuk caching hasil
            logger: Logger untuk pencatatan
        """
        self.logger = logger or get_logger("prediction_interface")
        self.prediction_service = prediction_service
        self.output_dir = Path(output_dir) if output_dir else Path("predictions")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Inisialisasi batch processor
        self.batch_processor = BatchPredictionProcessor(
            prediction_service=prediction_service,
            output_dir=str(self.output_dir),
            num_workers=num_workers,
            batch_size=batch_size,
            logger=self.logger
        )
        
        # Flag dan cache
        self.cache_results = cache_results
        self._result_cache = {}
        
        self.logger.info(
            f"🔮 Prediction interface siap digunakan:\n"
            f"   • Output dir: {self.output_dir}\n"
            f"   • Caching: {'Aktif' if cache_results else 'Nonaktif'}"
        )
    
    def predict_image(
        self,
        image_path: str,
        return_annotated: bool = True,
        save_result: bool = False
    ) -> Dict[str, Any]:
        """
        Prediksi pada satu gambar.
        
        Args:
            image_path: Path file gambar
            return_annotated: Flag untuk return gambar dengan anotasi
            save_result: Flag untuk menyimpan hasil
            
        Returns:
            Dict hasil prediksi
        """
        # Cek cache
        if self.cache_results and image_path in self._result_cache:
            return self._result_cache[image_path]
        
        # Buat prediksi
        result = self.prediction_service.predict_from_files(
            image_path, 
            return_annotated=return_annotated
        )
        
        # Tambahkan path gambar ke result
        result['image_path'] = image_path
        
        # Simpan jika diminta
        if save_result:
            self._save_result(result)
        
        # Cache jika aktif
        if self.cache_results:
            self._result_cache[image_path] = result
        
        return result
    
    def predict_from_url(
        self,
        image_url: str,
        return_annotated: bool = True,
        save_result: bool = False
    ) -> Dict[str, Any]:
        """
        Prediksi pada gambar dari URL.
        
        Args:
            image_url: URL gambar
            return_annotated: Flag untuk return gambar dengan anotasi
            save_result: Flag untuk menyimpan hasil
            
        Returns:
            Dict hasil prediksi
        """
        # Cek cache
        if self.cache_results and image_url in self._result_cache:
            return self._result_cache[image_url]
        
        try:
            import requests
            from io import BytesIO
            
            # Download gambar
            self.logger.info(f"📥 Mendownload gambar dari {image_url}")
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()
            
            # Load gambar sebagai array
            img_array = np.array(Image.open(BytesIO(response.content)))
            
            # Buat prediksi
            result = self.prediction_service.predict(
                img_array, 
                return_annotated=return_annotated
            )
            
            # Tambahkan URL gambar ke result
            result['image_url'] = image_url
            
            # Simpan jika diminta
            if save_result:
                self._save_url_result(image_url, result)
            
            # Cache jika aktif
            if self.cache_results:
                self._result_cache[image_url] = result
            
            return result
        
        except Exception as e:
            error_msg = f"❌ Error mendownload/memproses URL: {str(e)}"
            self.logger.error(error_msg)
            raise ModelInferenceError(error_msg) from e
    
    def predict_from_base64(
        self,
        base64_string: str,
        return_annotated: bool = True,
        save_result: bool = False
    ) -> Dict[str, Any]:
        """
        Prediksi pada gambar dari string base64.
        
        Args:
            base64_string: String base64 gambar
            return_annotated: Flag untuk return gambar dengan anotasi
            save_result: Flag untuk menyimpan hasil
            
        Returns:
            Dict hasil prediksi
        """
        try:
            # Decode base64
            img_data = base64.b64decode(base64_string)
            
            # Load gambar sebagai array
            img_array = np.array(Image.open(BytesIO(img_data)))
            
            # Buat prediksi
            result = self.prediction_service.predict(
                img_array, 
                return_annotated=return_annotated
            )
            
            # Generate hash untuk base64 string (untuk caching/referensi)
            import hashlib
            img_hash = hashlib.md5(base64_string.encode()).hexdigest()
            result['image_hash'] = img_hash
            
            # Simpan jika diminta
            if save_result:
                self._save_base64_result(img_hash, result)
            
            # Cache jika aktif
            if self.cache_results:
                self._result_cache[img_hash] = result
            
            return result
        
        except Exception as e:
            error_msg = f"❌ Error memproses gambar base64: {str(e)}"
            self.logger.error(error_msg)
            raise ModelInferenceError(error_msg) from e
    
    def run_batch_prediction(
        self,
        input_dir: str,
        save_results: bool = True,
        save_annotated: bool = True,
        recursive: bool = False,
        file_extensions: List[str] = ['.jpg', '.jpeg', '.png']
    ) -> Dict[str, Any]:
        """
        Jalankan prediksi batch pada direktori.
        
        Args:
            input_dir: Direktori berisi gambar
            save_results: Flag untuk menyimpan hasil individu
            save_annotated: Flag untuk menyimpan gambar dengan anotasi
            recursive: Flag untuk rekursi subdirektori
            file_extensions: List ekstensi file yang diproses
            
        Returns:
            Dict hasil batch prediksi
        """
        return self.batch_processor.process_directory(
            input_dir=input_dir,
            save_results=save_results,
            save_annotated=save_annotated,
            file_ext=file_extensions,
            recursive=recursive
        )
    
    def generate_report(
        self,
        input_source: Union[str, List[str]],
        output_filename: Optional[str] = None,
        include_annotated: bool = True
    ) -> str:
        """
        Generate laporan prediksi terkonsolidasi.
        
        Args:
            input_source: Direktori atau list file
            output_filename: Nama file output
            include_annotated: Flag untuk menyertakan gambar dengan anotasi
            
        Returns:
            Path file hasil
        """
        return self.batch_processor.run_and_save(
            input_source=input_source,
            output_filename=output_filename,
            save_annotated=include_annotated
        )
    
    def _save_result(self, result: Dict[str, Any]) -> None:
        """
        Simpan hasil prediksi satu gambar.
        
        Args:
            result: Hasil prediksi
        """
        import json
        import cv2
        
        try:
            # Extract path dan filename
            if 'image_path' not in result:
                return
                
            image_path = Path(result['image_path'])
            file_stem = image_path.stem
            
            # Simpan prediksi ke JSON
            json_path = self.output_dir / f"{file_stem}.json"
            
            # Extract prediksi
            if 'predictions' in result and len(result['predictions']) > 0:
                predictions = result['predictions'][0]
                
                with open(json_path, 'w') as f:
                    json.dump(predictions, f, indent=2)
                
                # Simpan gambar dengan anotasi jika ada
                if 'annotated_images' in result and len(result['annotated_images']) > 0:
                    annotated_img = result['annotated_images'][0]
                    img_path = self.output_dir / f"{file_stem}_annotated.jpg"
                    
                    # Konversi dari RGB ke BGR untuk cv2
                    cv2.imwrite(
                        str(img_path), 
                        cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR)
                    )
        
        except Exception as e:
            self.logger.warning(f"⚠️ Error menyimpan hasil: {str(e)}")
    
    def _save_url_result(self, url: str, result: Dict[str, Any]) -> None:
        """
        Simpan hasil prediksi dari URL.
        
        Args:
            url: URL gambar
            result: Hasil prediksi
        """
        import json
        import cv2
        import hashlib
        
        try:
            # Generate filename dari hash URL
            url_hash = hashlib.md5(url.encode()).hexdigest()
            
            # Simpan prediksi ke JSON
            json_path = self.output_dir / f"url_{url_hash}.json"
            
            # Tambahkan URL ke hasil
            result_copy = result.copy()
            result_copy['source_url'] = url
            
            with open(json_path, 'w') as f:
                json.dump(result_copy, f, indent=2)
            
            # Simpan gambar dengan anotasi jika ada
            if 'annotated_images' in result and len(result['annotated_images']) > 0:
                annotated_img = result['annotated_images'][0]
                img_path = self.output_dir / f"url_{url_hash}_annotated.jpg"
                
                # Konversi dari RGB ke BGR untuk cv2
                cv2.imwrite(
                    str(img_path), 
                    cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR)
                )
        
        except Exception as e:
            self.logger.warning(f"⚠️ Error menyimpan hasil URL: {str(e)}")
    
    def _save_base64_result(self, img_hash: str, result: Dict[str, Any]) -> None:
        """
        Simpan hasil prediksi dari base64.
        
        Args:
            img_hash: Hash gambar base64
            result: Hasil prediksi
        """
        import json
        import cv2
        
        try:
            # Simpan prediksi ke JSON
            json_path = self.output_dir / f"b64_{img_hash}.json"
            
            with open(json_path, 'w') as f:
                json.dump(result, f, indent=2)
            
            # Simpan gambar dengan anotasi jika ada
            if 'annotated_images' in result and len(result['annotated_images']) > 0:
                annotated_img = result['annotated_images'][0]
                img_path = self.output_dir / f"b64_{img_hash}_annotated.jpg"
                
                # Konversi dari RGB ke BGR untuk cv2
                cv2.imwrite(
                    str(img_path), 
                    cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR)
                )
        
        except Exception as e:
            self.logger.warning(f"⚠️ Error menyimpan hasil base64: {str(e)}")
    
    def clear_cache(self) -> None:
        """Reset cache."""
        self._result_cache = {}
        self.prediction_service.clear_cache()
        self.logger.info("🧹 Prediction cache dibersihkan")
    
    def get_results_as_base64(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Konversi gambar hasil anotasi ke base64 untuk web/API.
        
        Args:
            result: Hasil prediksi
            
        Returns:
            Dict dengan base64 gambar
        """
        if 'annotated_images' not in result or not result['annotated_images']:
            return result
            
        # Clone result
        new_result = result.copy()
        
        # Konversi annotated_images ke base64
        base64_images = []
        
        for img in result['annotated_images']:
            # Konversi numpy array ke PIL image
            pil_img = Image.fromarray(img)
            
            # Save ke buffer
            buffer = BytesIO()
            pil_img.save(buffer, format="JPEG")
            
            # Encode ke base64
            img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
            base64_images.append(f"data:image/jpeg;base64,{img_str}")
        
        # Replace annotated_images dengan base64
        new_result['annotated_images_base64'] = base64_images
        
        return new_result