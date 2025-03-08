"""
File: smartcash/utils/training/training_metrics.py
Author: Alfrida Sabar
Deskripsi: Pengelolaan dan pelacakan metrik training dengan dukungan persistensi dan visualisasi
"""

import os
import csv
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import time

class TrainingMetrics:
    """
    Pengelolaan metrik training dengan dukungan:
    - Pelacakan history metrik
    - Persistensi ke CSV dan JSON
    - Restore metrik dari checkpoint
    """
    
    def __init__(self, logger=None):
        """
        Inisialisasi pengelola metrik.
        
        Args:
            logger: Logger untuk mencatat aktivitas
        """
        self.logger = logger
        
        # Dictionary untuk menyimpan history metrik
        self.history = {}
        
        # State untuk tracking CSV headers
        self.csv_headers_written = False
    
    def update_history(self, key: str, value: Any) -> None:
        """
        Update history untuk metrik tertentu.
        
        Args:
            key: Nama metrik
            value: Nilai metrik
        """
        if key not in self.history:
            self.history[key] = []
        
        self.history[key].append(value)
    
    def get_history(self, key: str) -> List:
        """
        Dapatkan history untuk metrik tertentu.
        
        Args:
            key: Nama metrik
            
        Returns:
            List nilai metrik atau list kosong jika tidak ada
        """
        return self.history.get(key, [])
    
    def get_all_history(self) -> Dict[str, List]:
        """
        Dapatkan semua history metrik.
        
        Returns:
            Dict berisi semua history metrik
        """
        return self.history
    
    def restore_history(self, history: Dict[str, List]) -> None:
        """
        Restore history metrik dari dict (mis. dari checkpoint).
        
        Args:
            history: Dict berisi history metrik
        """
        if not history:
            return
            
        self.history = history
    
    def log_to_csv(self, epoch: int, metrics: Dict[str, float], output_dir: Path) -> bool:
        """
        Catat metrics ke file CSV.
        
        Args:
            epoch: Epoch saat ini
            metrics: Metrics yang akan dicatat
            output_dir: Direktori untuk menyimpan file CSV
            
        Returns:
            Bool yang menunjukkan keberhasilan logging
        """
        try:
            # Persiapkan data
            log_dir = output_dir / 'logs'
            log_dir.mkdir(parents=True, exist_ok=True)
            
            metrics_file = log_dir / 'metrics.csv'
            
            # Persiapkan data untuk tulis ke CSV
            data = {'epoch': epoch}
            data.update(metrics)
            
            # Periksa apakah file sudah ada
            file_exists = metrics_file.exists()
            
            try:
                with open(metrics_file, 'a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=data.keys())
                    
                    # Tulis header jika file baru
                    if not file_exists:
                        writer.writeheader()
                        self.csv_headers_written = True
                    
                    # Tulis data
                    writer.writerow(data)
                    
                return True
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"⚠️ Gagal menulis metrics ke CSV: {str(e)}")
                return False
                
        except Exception as e:
            if self.logger:
                self.logger.warning(f"⚠️ Gagal mencatat metrics: {str(e)}")
            return False
    
    def save_to_json(self, output_path: str) -> bool:
        """
        Simpan semua history metrik ke file JSON.
        
        Args:
            output_path: Path untuk menyimpan file JSON
            
        Returns:
            Bool yang menunjukkan keberhasilan penyimpanan
        """
        try:
            # Tambahkan metadata
            data_to_save = {
                'history': self.history,
                'timestamp': time.time(),
                'metrics_count': {k: len(v) for k, v in self.history.items()}
            }
            
            # Pastikan direktori ada
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Simpan ke file
            with open(output_path, 'w') as f:
                json.dump(data_to_save, f, indent=2)
                
            if self.logger:
                self.logger.info(f"✅ History metrik disimpan ke {output_path}")
                
            return True
        except Exception as e:
            if self.logger:
                self.logger.warning(f"⚠️ Gagal menyimpan history metrik: {str(e)}")
            return False
    
    def load_from_json(self, input_path: str) -> bool:
        """
        Muat history metrik dari file JSON.
        
        Args:
            input_path: Path file JSON
            
        Returns:
            Bool yang menunjukkan keberhasilan loading
        """
        try:
            # Check if file exists
            if not os.path.exists(input_path):
                if self.logger:
                    self.logger.warning(f"⚠️ File {input_path} tidak ditemukan")
                return False
                
            # Load from file
            with open(input_path, 'r') as f:
                data = json.load(f)
                
            # Update history
            if 'history' in data:
                self.history = data['history']
                
                if self.logger:
                    self.logger.info(f"✅ History metrik dimuat dari {input_path}")
                    
                return True
            else:
                if self.logger:
                    self.logger.warning(f"⚠️ Format file tidak valid: {input_path}")
                return False
                
        except Exception as e:
            if self.logger:
                self.logger.warning(f"⚠️ Gagal memuat history metrik: {str(e)}")
            return False
    
    def get_best_metrics(self) -> Dict[str, float]:
        """
        Dapatkan metrik terbaik (loss terendah atau metrik tertinggi).
        
        Returns:
            Dict berisi metrik terbaik
        """
        best_metrics = {}
        
        # Best validation loss
        val_losses = self.get_history('val_loss')
        if val_losses:
            best_idx = val_losses.index(min(val_losses))
            best_metrics['best_val_loss'] = val_losses[best_idx]
            best_metrics['best_epoch'] = best_idx
            
            # Add other metrics from the same epoch
            for key, values in self.history.items():
                if key != 'val_loss' and len(values) > best_idx:
                    best_metrics[f'best_{key}'] = values[best_idx]
        
        return best_metrics