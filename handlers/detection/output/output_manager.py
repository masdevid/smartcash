# File: smartcash/handlers/detection/output/output_manager.py
# Author: Alfrida Sabar
# Deskripsi: Manager untuk pengelolaan output hasil deteksi

import os
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Union, Optional, Tuple
import numpy as np

from smartcash.utils.logger import get_logger
from smartcash.exceptions.base import DataError

class OutputManager:
    """
    Manager untuk pengelolaan output hasil deteksi.
    Mengelola penyimpanan hasil deteksi dalam berbagai format.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        output_dir: Union[str, Path],
        visualizer,
        logger = None,
        colab_mode: bool = False
    ):
        """
        Inisialisasi output manager.
        
        Args:
            config: Konfigurasi
            output_dir: Direktori output
            visualizer: Adapter visualizer
            logger: Logger kustom (opsional)
            colab_mode: Flag untuk mode Google Colab
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.visualizer = visualizer
        self.logger = logger or get_logger("output_manager")
        self.colab_mode = colab_mode
        
        # Buat direktori output jika belum ada
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Buat subdir untuk jenis output berbeda
        self.viz_dir = self.output_dir / "visualizations"
        self.json_dir = self.output_dir / "json"
        self.viz_dir.mkdir(exist_ok=True)
        self.json_dir.mkdir(exist_ok=True)
        
        self.logger.info(f"📂 OutputManager diinisialisasi dengan output_dir: {self.output_dir}")
    
    def save_visualization(
        self,
        source: Union[str, Path],
        image: np.ndarray,
        detections: List[Dict[str, Any]],
        custom_filename: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Simpan visualisasi hasil deteksi.
        
        Args:
            source: Path gambar sumber
            image: Gambar sebagai array numpy
            detections: List deteksi
            custom_filename: Nama file kustom (opsional)
            **kwargs: Parameter tambahan untuk visualizer
            
        Returns:
            Path file output
        """
        # Tentukan output path
        if custom_filename:
            output_filename = custom_filename
        else:
            source_path = Path(source)
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            output_filename = f"{source_path.stem}_detected_{timestamp}{source_path.suffix}"
            
        output_path = self.viz_dir / output_filename
        
        # Visualisasikan dan simpan
        try:
            # Gunakan visualizer adapter
            result = self.visualizer.visualize_detections(
                image=image,
                detections=detections,
                output_path=str(output_path),
                **kwargs
            )
            
            self.logger.info(f"✅ Visualisasi disimpan: {output_path}")
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"❌ Error saat menyimpan visualisasi: {str(e)}")
            raise
    
    def save_results(
        self,
        source: Union[str, Path],
        results: Dict[str, Any],
        custom_filename: Optional[str] = None,
        **kwargs
    ) -> Dict[str, str]:
        """
        Simpan hasil deteksi ke file JSON.
        
        Args:
            source: Path gambar sumber
            results: Dictionary hasil deteksi
            custom_filename: Nama file kustom (opsional)
            **kwargs: Parameter tambahan
            
        Returns:
            Dictionary path output untuk setiap format
        """
        # Tentukan output path
        if custom_filename:
            if custom_filename.endswith('.json'):
                json_filename = custom_filename
            else:
                json_filename = f"{custom_filename}.json"
        else:
            source_path = Path(source)
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            json_filename = f"{source_path.stem}_results_{timestamp}.json"
            
        json_path = self.json_dir / json_filename
        
        # Bersihkan hasil untuk JSON
        clean_results = self._prepare_results_for_json(results)
        
        # Simpan ke JSON
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(clean_results, f, indent=2, ensure_ascii=False)
                
            self.logger.info(f"✅ Hasil deteksi disimpan: {json_path}")
            
            return {
                'json': str(json_path)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error saat menyimpan hasil: {str(e)}")
            raise
    
    def _prepare_results_for_json(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Persiapkan hasil deteksi untuk disimpan ke JSON.
        Mengkonversi tipe data yang tidak compatible dengan JSON.
        
        Args:
            results: Dictionary hasil deteksi
            
        Returns:
            Dictionary hasil yang siap untuk disimpan ke JSON
        """
        # Copy untuk menghindari modifikasi dict asli
        clean_results = {}
        
        # Tambahkan metadata
        clean_results['timestamp'] = datetime.now().isoformat()
        clean_results['source'] = results.get('source', '')
        
        # Cleanup detections
        if 'detections' in results:
            clean_detections = []
            for det in results['detections']:
                clean_det = {}
                for k, v in det.items():
                    # Konversi numpy array dan tensor ke list
                    if isinstance(v, (np.ndarray, list)):
                        clean_det[k] = [float(x) for x in v]
                    elif hasattr(v, 'item') and callable(getattr(v, 'item')):
                        # Handle torch.Tensor scalar atau numpy scalar
                        clean_det[k] = v.item()
                    else:
                        clean_det[k] = v
                clean_detections.append(clean_det)
            clean_results['detections'] = clean_detections
        
        # Tambahkan info per layer
        if 'detections_by_layer' in results:
            clean_by_layer = {}
            for layer, dets in results['detections_by_layer'].items():
                clean_dets = []
                for det in dets:
                    clean_det = {}
                    for k, v in det.items():
                        # Konversi numpy array dan tensor ke list
                        if isinstance(v, (np.ndarray, list)):
                            clean_det[k] = [float(x) for x in v]
                        elif hasattr(v, 'item') and callable(getattr(v, 'item')):
                            clean_det[k] = v.item()
                        else:
                            clean_det[k] = v
                    clean_dets.append(clean_det)
                clean_by_layer[layer] = clean_dets
            clean_results['detections_by_layer'] = clean_by_layer
        
        # Tambahkan statistik
        clean_results['num_detections'] = results.get('num_detections', 0)
        
        # Tambahkan execution time
        if 'inference_time' in results:
            clean_results['inference_time'] = float(results['inference_time'])
        if 'execution_time' in results:
            clean_results['execution_time'] = float(results['execution_time'])
        
        # Output path visualisasi
        if 'visualization_path' in results:
            clean_results['visualization_path'] = results['visualization_path']
        
        return clean_results
    
    def get_colab_path(self, path: Union[str, Path]) -> str:
        """
        Dapatkan path yang user-friendly untuk Google Colab.
        
        Args:
            path: Path file
            
        Returns:
            Path file untuk ditampilkan ke user
        """
        if not self.colab_mode:
            return str(path)
            
        # Konversi ke path absolut
        abs_path = Path(path).absolute()
        
        # Cek apakah di directory Google Drive
        if '/content/drive/' in str(abs_path):
            # Format path untuk menampilkan di Colab
            return f"📂 Google Drive: {str(abs_path).replace('/content/drive/MyDrive/', '')}"
        else:
            # Path lokal di Colab
            return f"📂 Colab: {str(abs_path).replace('/content/', '')}"