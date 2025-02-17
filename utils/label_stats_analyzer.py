# File: utils/label_coordinate_counter.py
# Author: Alfrida Sabar
# Deskripsi: Menghitung statistik jumlah koordinat pada label dataset deteksi mata uang

import os
import numpy as np
from typing import Dict, List
from pathlib import Path
import yaml
from tqdm.auto import tqdm

class LabelCoordinateCounter:
    """
    Penghitung koordinat untuk label dataset deteksi mata uang
    Mendukung berbagai format label objek
    """
    
    def __init__(
        self, 
        labels_dir: str, 
        config_path: str = 'configs/base_config.yaml'
    ):
        """
        Inisialisasi counter koordinat
        
        Args:
            labels_dir (str): Path direktori label
            config_path (str): Path konfigurasi dataset
        """
        self.labels_dir = Path(labels_dir)
        
        # Load konfigurasi kelas dari config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            self.class_names = config['dataset']['classes']
        
        # Validasi direktori
        if not self.labels_dir.exists():
            raise ValueError(f"Direktori label tidak ditemukan: {labels_dir}")
        
        # Struktur untuk menyimpan statistik koordinat
        self.coordinate_stats = {
            'total_files': 0,
            'total_labels': 0,
            'coordinate_counts': {
                'min': float('inf'),
                'max': float('-inf'),
                'mean': 0,
                'median': None,
                'values': []
            }
        }
    
    def count_coordinates(self) -> Dict:
        """
        Menghitung jumlah koordinat pada label
        
        Returns:
            Dict: Statistik jumlah koordinat label
        """
        # Cari semua file label
        label_files = list(self.labels_dir.glob('*.txt'))
        
        # Progress bar
        for label_file in tqdm(label_files, desc="📊 Menghitung Koordinat Label"):
            self._process_label_file(label_file)
        
        # Hitung statistik akhir
        self._compute_final_stats()
        
        return self.coordinate_stats
    
    def _process_label_file(self, label_file: Path) -> None:
        """
        Proses satu file label
        
        Args:
            label_file (Path): Path file label
        """
        try:
            with open(label_file, 'r') as f:
                # Hitung label dalam file ini
                labels_in_file = len(f.readlines())
                
                # Update statistik
                self.coordinate_stats['total_files'] += 1
                self.coordinate_stats['total_labels'] += labels_in_file
                self.coordinate_stats['coordinate_counts']['values'].append(labels_in_file)
                
                # Update min/max
                self.coordinate_stats['coordinate_counts']['min'] = min(
                    self.coordinate_stats['coordinate_counts']['min'], 
                    labels_in_file
                )
                self.coordinate_stats['coordinate_counts']['max'] = max(
                    self.coordinate_stats['coordinate_counts']['max'], 
                    labels_in_file
                )
        
        except Exception as e:
            print(f"❌ Gagal memproses {label_file}: {e}")
    
    def _compute_final_stats(self) -> None:
        """
        Hitung statistik akhir dari koordinat
        """
        counts = self.coordinate_stats['coordinate_counts']['values']
        
        # Hitung rata-rata
        self.coordinate_stats['coordinate_counts']['mean'] = (
            np.mean(counts) if counts else 0
        )
        
        # Hitung median
        self.coordinate_stats['coordinate_counts']['median'] = (
            np.median(counts) if counts else 0
        )
    
    def generate_report(self) -> str:
        """
        Generate laporan statistik koordinat
        
        Returns:
            str: Laporan statistik koordinat label
        """
        stats = self.coordinate_stats['coordinate_counts']
        
        report = "📊 Laporan Statistik Jumlah Koordinat Label\n"
        report += "=" * 40 + "\n\n"
        report += f"📁 Total File: {self.coordinate_stats['total_files']}\n"
        report += f"🏷️ Total Label: {self.coordinate_stats['total_labels']}\n\n"
        report += "🔢 Statistik Koordinat:\n"
        report += f"  Minimum: {stats['min']}\n"
        report += f"  Maksimum: {stats['max']}\n"
        report += f"  Rata-rata: {stats['mean']:.2f}\n"
        report += f"  Median: {stats['median']:.2f}\n"
        
        return report

# Contoh penggunaan
def main():
    """
    Fungsi utama untuk menjalankan penghitungan koordinat
    """
    try:
        # Sesuaikan path dengan struktur project anda
        counter = LabelCoordinateCounter(
            labels_dir='data/train/labels',  # Path sesuaikan dengan struktur project
            config_path='configs/base_config.yaml'
        )
        
        # Hitung koordinat
        stats = counter.count_coordinates()
        
        # Cetak laporan
        print(counter.generate_report())
        
    except Exception as e:
        print(f"❌ Gagal menghitung koordinat: {e}")

if __name__ == "__main__":
    main()