# File: src/data/data_manager.py
# Author: Alfrida Sabar
# Deskripsi: Manajer pusat untuk operasi dan manajemen dataset SmartCash Detector

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

from config.manager import ConfigManager
from interfaces.handlers.preparation_handler import DataPreparationHandler
from interfaces.handlers.verification_handler import DataVerificationHandler
from interfaces.handlers.statistics_handler import DataStatisticsHandler
from interfaces.handlers.cleaning_handler import DataCleaningHandler
from interfaces.handlers.augmentation_handler import DataAugmentationHandler
from utils.logging import ColoredLogger

class DataManager:
    """
    Manajer pusat untuk operasi manajemen dataset dengan pendekatan modular
    
    Tanggung Jawab Utama:
    - Koordinasi antar handler data
    - Penyediaan antarmuka terpadu untuk operasi dataset
    - Manajemen konteks dan status dataset
    """
    
    def __init__(self, config: ConfigManager):
        """
        Inisialisasi DataManager dengan konfigurasi dan handler spesifik
        
        Args:
            config (ConfigManager): Konfigurasi proyek untuk manajemen dataset
        """
        self.logger = ColoredLogger('DataManager')
        self.cfg = config
        
        # Inisialisasi handler-handler operasi dataset
        self.handlers = {
            'preparation': DataPreparationHandler(config),
            'verification': DataVerificationHandler(config),
            'statistics': DataStatisticsHandler(config),
            'cleaning': DataCleaningHandler(config),
            'augmentation': DataAugmentationHandler(config)
        }
        
        # Direktori utama dataset
        self.data_dir = Path(config.data.data_dir)
        self.rupiah_dir = self.data_dir / 'rupiah'
        
    def prepare_dataset(self) -> bool:
        """
        Persiapan struktur dan konfigurasi dataset
        
        Returns:
            bool: Status keberhasilan persiapan dataset
        """
        self.logger.info("🔧 Mempersiapkan struktur dataset...")
        return self.handlers['preparation'].prepare_dataset()
    
    def verify_dataset(self, plot: bool = False) -> bool:
        """
        Verifikasi integritas dan kualitas dataset
        
        Args:
            plot (bool): Apakah akan membuat plot visualisasi
        
        Returns:
            bool: Status validitas dataset
        """
        self.logger.info("🔍 Memverifikasi dataset...")
        
        # Jalankan verifikasi
        verification_results = self.handlers['verification'].verify_dataset()
        
        # Optionally plot verification results
        if plot:
            self._plot_verification_results(verification_results)
        
        # Periksa apakah dataset siap digunakan
        readiness, issues = self.handlers['verification'].check_dataset_readiness()
        
        if not readiness:
            self.logger.warning("⚠️ Dataset belum siap digunakan!")
            for issue in issues:
                self.logger.warning(f"  • {issue}")
        
        return readiness
    
    def analyze_dataset(self) -> Dict:
        """
        Analisis komprehensif dataset
        
        Returns:
            Dict: Statistik dan informasi dataset
        """
        self.logger.info("📊 Menganalisis dataset...")
        
        # Dapatkan statistik dari handler statistik
        stats_handler = self.handlers['statistics']
        dataset_stats = stats_handler.analyze_dataset()
        
        # Dapatkan rekomendasi augmentasi
        augmentation_recs = stats_handler.get_augmentation_recommendations()
        
        # Analisis keseimbangan kelas
        class_balance = stats_handler.analyze_class_balance()
        
        # Analisis kualitas
        quality_issues = stats_handler.analyze_quality_issues()
        
        return {
            'stats': dataset_stats,
            'augmentation_recommendations': augmentation_recs,
            'class_balance': class_balance,
            'quality_issues': quality_issues
        }
    
    def clean_dataset(self, mode: str = 'all') -> Dict:
        """
        Pembersihan dataset berdasarkan mode tertentu
        
        Args:
            mode (str): Mode pembersihan 
                - 'all': Hapus semua data
                - 'augmented': Hapus data augmentasi
                - 'training': Hapus data training
                - 'corrupt': Hapus data korup
        
        Returns:
            Dict: Statistik pembersihan
        """
        self.logger.info(f"🧹 Membersihkan dataset (mode: {mode})...")
        return self.handlers['cleaning'].clean_dataset(mode)
    
    def augment_dataset(self, 
                       factor: int = 2, 
                       modes: Optional[List[str]] = None) -> Dict:
        """
        Augmentasi dataset untuk meningkatkan variasi
        
        Args:
            factor (int): Faktor pengulangan augmentasi
            modes (Optional[List[str]]): Mode augmentasi spesifik
        
        Returns:
            Dict: Statistik augmentasi
        """
        self.logger.info(f"🎨 Melakukan augmentasi dataset (faktor: {factor})...")
        return self.handlers['augmentation'].augment_dataset(
            factor=factor, 
            modes=modes
        )
    
    def print_dataset_info(self):
        """
        Cetak informasi dataset dalam format yang mudah dibaca
        """
        stats = self.analyze_dataset()
        
        print("\n🔍 Ringkasan Dataset:")
        print(f"📁 Total Gambar: {stats['stats'].total_images}")
        print(f"📊 Distribusi Kelas: ")
        for cls, count in stats['stats'].class_counts.items():
            print(f"   • {cls}: {count}")
        
        print("\n💡 Rekomendasi Augmentasi:")
        for rec in stats['augmentation_recommendations']['general']:
            print(f"   • {rec['description']}")
            for action in rec['actions']:
                print(f"     - {action}")
    
    def _plot_verification_results(self, results: Dict):
        """
        Buat visualisasi hasil verifikasi dataset
        
        Args:
            results (Dict): Hasil verifikasi dari handler
        """
        try:
            plt.figure(figsize=(15, 5))
            
            # Plot distribusi gambar per split
            splits = list(results.keys())
            images = [results[split]['images'] for split in splits]
            labels = [results[split]['labels'] for split in splits]
            
            plt.subplot(1, 3, 1)
            plt.bar(splits, images, color='skyblue', label='Gambar')
            plt.bar(splits, labels, bottom=images, color='lightgreen', label='Label')
            plt.title('Distribusi Dataset')
            plt.xlabel('Split')
            plt.ylabel('Jumlah')
            plt.legend()
            
            # Plot isu ukuran gambar
            size_issues = [len(results[split]['size_issues']) for split in splits]
            plt.subplot(1, 3, 2)
            plt.bar(splits, size_issues, color='coral')
            plt.title('Isu Ukuran Gambar')
            plt.xlabel('Split')
            plt.ylabel('Jumlah Isu')
            
            # Plot isu label
            label_issues = [len(results[split]['label_issues']) for split in splits]
            plt.subplot(1, 3, 3)
            plt.bar(splits, label_issues, color='violet')
            plt.title('Isu Label')
            plt.xlabel('Split')
            plt.ylabel('Jumlah Isu')
            
            plt.tight_layout()
            plt.savefig(self.rupiah_dir / 'verification_results.png')
            plt.close()
            
            self.logger.info("📊 Plot hasil verifikasi tersimpan!")
        except Exception as e:
            self.logger.error(f"❌ Gagal membuat plot verifikasi: {str(e)}")
    
    def validate_directory(self, path: Path) -> bool:
        """
        Validasi direktori dengan opsi pembuatan
        
        Args:
            path (Path): Path direktori yang akan divalidasi
        
        Returns:
            bool: Status validitas direktori
        """
        try:
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
            return True
        except Exception as e:
            self.logger.error(f"❌ Gagal memvalidasi direktori {path}: {str(e)}")
            return False