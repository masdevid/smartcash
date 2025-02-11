# File: src/interfaces/data_interface.py
# Author: Alfrida Sabar
# Deskripsi: Antarmuka modular untuk manajemen data SmartCash Detector

from pathlib import Path
from typing import Dict, Optional
from termcolor import colored
import cv2
from .base_interface import BaseInterface
from handlers.dataset_handlers import (
    DatasetCopyHandler,
    DatasetCleanHandler,
    DatasetVerifyHandler
)
from handlers.augmentation_handlers import AugmentationHandler

class DataInterface(BaseInterface):
    def __init__(self, config):
        super().__init__()
        self.cfg = config
        
        # Define directory structure
        self.project_root = Path(__file__).parent.parent.parent
        self.data_dir = self.project_root / "data"
        self.rupiah_dir = self.data_dir / "rupiah"
        
        # Initialize handlers
        self._init_handlers()

    def _init_handlers(self):
        """Initialize operation handlers"""
        self.copy_handler = DatasetCopyHandler(self.rupiah_dir)
        self.clean_handler = DatasetCleanHandler(self.rupiah_dir)
        self.verify_handler = DatasetVerifyHandler(self.rupiah_dir)
        self.aug_handler = AugmentationHandler(self.rupiah_dir)

    def tampilkan_menu(self):
        menu = """
📁 Menu Manajemen Data:

1. Persiapan Dataset
2. Terapkan Augmentasi
3. Verifikasi Dataset
4. Statistik Dataset
5. Salin Dataset Eksternal 📥
6. Bersihkan Dataset 🗑️

0. Kembali ke Menu Utama
"""
        return self.prompt(menu, color='cyan')

    def handle_menu(self):
        while True:
            pilihan = self.tampilkan_menu()
            
            try:
                if pilihan == '0':
                    break
                elif pilihan == '1':
                    self.persiapkan_dataset()
                elif pilihan == '2':
                    self.terapkan_augmentasi()
                elif pilihan == '3':
                    self.verifikasi_dataset()
                elif pilihan == '4':
                    self.tampilkan_statistik()
                elif pilihan == '5':
                    self.salin_dataset_eksternal()
                elif pilihan == '6':
                    self.bersihkan_dataset()
                else:
                    self.show_error("❌ Pilihan menu tidak valid!")
            except Exception as e:
                self.show_error(f"❌ Terjadi kesalahan: {str(e)}")

    def persiapkan_dataset(self):
        """Proses persiapan dataset"""
        self.logger.info("\n🔄 Persiapan Dataset")
        
        # Verifikasi dataset jika sudah ada
        if self.rupiah_dir.exists():
            stats = self.verify_handler.verify_dataset()
            self._display_verification_results(stats)
            
            if not self.confirm("\nDataset sudah ada. Ingin mempersiapkan ulang?"):
                return
        
        # Buat struktur direktori
        try:
            self.copy_handler._create_dest_structure()
            self.copy_handler._create_config()
            self.show_success("✨ Persiapan dataset selesai!")
        except Exception as e:
            self.show_error(f"❌ Gagal mempersiapkan dataset: {str(e)}")

    def terapkan_augmentasi(self):
        """Proses augmentasi data"""
        # Get augmentation descriptions
        aug_types = self.aug_handler.get_augmentation_descriptions()
        
        # Display augmentation info
        self.logger.info("\n🎨 Konfigurasi Augmentasi Data:\n")
        self._display_augmentation_info()
        
        # Get user choice
        for i, (key, info) in enumerate(aug_types.items(), 1):
            print(colored(f"\n{i}. {info['name']}", 'yellow'))
            print(f"   {info['description']}")
            for feature in info['features']:
                print(f"   • {feature}")
            print(f"   Cocok untuk: {info['use_case']}")
        
        choice = self.prompt("\nPilih jenis augmentasi", default="3")
        mode = list(aug_types.keys())[int(choice)-1]
        
        # Get augmentation parameters if needed
        params = None
        if mode == 'currency':
            params = self._get_currency_augmentation_params()
        
        # Get multiplication factor
        aug_info = self.aug_handler.get_augmentation_info()
        self._display_augmentation_factor_info(aug_info)
        factor = int(self.prompt("Faktor penggandaan", 
                               default=str(aug_info['recommended_factor'])))
        
        # Apply augmentation
        if self.confirm("\nMulai proses augmentasi?"):
            try:
                stats = self.aug_handler.apply_augmentation(mode, factor, params)
                self._display_augmentation_results(stats)
            except Exception as e:
                self.show_error(f"❌ Gagal melakukan augmentasi: {str(e)}")

    def verifikasi_dataset(self):
        """Verifikasi integritas dataset"""
        self.logger.info("\n🔍 Memulai verifikasi dataset...")
        
        try:
            stats = self.verify_handler.verify_dataset()
            self._display_verification_results(stats)
        except Exception as e:
            self.show_error(f"❌ Gagal memverifikasi dataset: {str(e)}")

    def tampilkan_statistik(self):
        """Tampilkan statistik dataset"""
        self.logger.info("\n📊 Menganalisis dataset...")
        
        try:
            # Get verification stats
            verify_stats = self.verify_handler.verify_dataset()
            
            # Get augmentation stats
            aug_stats = self.aug_handler.get_augmentation_info()
            
            self._display_dataset_statistics(verify_stats, aug_stats)
        except Exception as e:
            self.show_error(f"❌ Gagal mengambil statistik: {str(e)}")

    def salin_dataset_eksternal(self):
        """Proses penyalinan dataset dari sumber eksternal"""
        self.logger.info("\n📥 Salin Dataset Eksternal")
        
        # Get source location
        source_dir = self._get_source_location()
        if not source_dir:
            return
            
        # Validate and copy
        if self.confirm("\nMulai penyalinan dataset?"):
            try:
                stats = self.copy_handler.copy_dataset(source_dir)
                self._display_copy_results(stats)
            except Exception as e:
                self.show_error(f"❌ Gagal menyalin dataset: {str(e)}")

    def bersihkan_dataset(self):
        """Proses pembersihan dataset"""
        self.logger.info("\n🗑️ Pembersihan Dataset")
        
        # Get cleaning mode
        mode = self._get_cleaning_mode()
        if not mode:
            return
            
        # Confirm and clean
        if self.confirm("Anda yakin ingin melanjutkan?"):
            try:
                stats = self.clean_handler.clean_dataset(mode)
                self._display_cleaning_results(stats)
            except Exception as e:
                self.show_error(f"❌ Gagal membersihkan dataset: {str(e)}")

    # Helper methods for displaying information
    def _display_verification_results(self, stats: Dict):
        """Display dataset verification results"""
        self.logger.info("\n📊 Hasil Verifikasi:")
        for split, info in stats.items():
            print(colored(f"\n{split.upper()}:", 'cyan'))
            print(f"  • Total Gambar: {info['images']}")
            print(f"  • Total Label: {info['labels']}")
            print(f"  • File Rusak: {info['corrupt']}")
            print(f"  • Label Invalid: {info['invalid']}")
            print(f"  • Gambar Original: {info['original']}")
            print(f"  • Gambar Augmentasi: {info['augmented']}")

    def _display_augmentation_info(self):
        """Display general augmentation information"""
        self.logger.info("📝 Tujuan Augmentasi:")
        self.logger.info("- Meningkatkan variasi data training")
        self.logger.info("- Mencegah overfitting")
        self.logger.info("- Meningkatkan ketahanan model terhadap variasi kondisi nyata\n")

    def _display_augmentation_factor_info(self, aug_info: Dict):
        """Display multiplication factor guidance"""
        self.logger.info("\n📊 Panduan Faktor Penggandaan:")
        self.logger.info("1-2x  : Penambahan minimal, untuk dataset besar (>1000 gambar)")
        self.logger.info("3-4x  : Penambahan moderat, untuk dataset sedang (500-1000 gambar)")
        self.logger.info("5-8x  : Penambahan besar, untuk dataset kecil (<500 gambar)")
        self.logger.info(f"\nℹ️  Jumlah gambar saat ini: {aug_info['original']}")
        self.logger.info(f"💡 Rekomendasi faktor: {aug_info['recommended_factor']}x")

    def _display_augmentation_results(self, stats: Dict):
        """Display augmentation results"""
        self.show_success(
            f"✨ Augmentasi selesai!\n"
            f"   • Gambar diproses: {stats['processed']}\n"
            f"   • Augmentasi dibuat: {stats['augmented']}\n"
            f"   • Error: {stats['errors']}"
        )

    def _display_copy_results(self, stats: Dict):
        """Display dataset copy results"""
        self.show_success(
            f"✨ Dataset berhasil disalin!\n"
            f"   • File disalin: {stats['copied']}\n"
            f"   • File dilewati: {stats['skipped']}\n"
            f"   • Error: {stats['errors']}"
        )

    def _display_cleaning_results(self, stats: Dict):
        """Display dataset cleaning results"""
        self.show_success(
            f"✨ Pembersihan selesai!\n"
            f"   • File dihapus: {stats['removed']}\n"
            f"   • Error: {stats['errors']}"
        )

    def _display_dataset_statistics(self, verify_stats: Dict, aug_stats: Dict):
        """Display comprehensive dataset statistics"""
        self.logger.info("\n📈 Statistik Dataset:")
        
        # Overall stats
        total_images = sum(s['images'] for s in verify_stats.values())
        total_original = sum(s['original'] for s in verify_stats.values())
        total_augmented = sum(s['augmented'] for s in verify_stats.values())
        
        print(colored("\nRingkasan:", 'cyan'))
        print(f"  • Total Gambar: {total_images}")
        print(f"  • Gambar Original: {total_original}")
        print(f"  • Gambar Augmentasi: {total_augmented}")
        
        # Per-split stats
        for split, stats in verify_stats.items():
            print(colored(f"\n{split.upper()}:", 'cyan'))
            print(f"  • Gambar: {stats['images']}")
            print(f"  • Original: {stats['original']}")
            print(f"  • Augmentasi: {stats['augmented']}")
            print(f"  • File Rusak: {stats['corrupt']}")
            
        # Augmentation recommendation
        print(colored("\nRekomendasi Augmentasi:", 'cyan'))
        print(f"  • Faktor yang disarankan: {aug_stats['recommended_factor']}x")

    # Helper methods for user input
    def _get_source_location(self) -> Optional[Path]:
        """Get and validate source dataset location"""
        self.logger.info("\nPilih lokasi sumber dataset:")
        print(colored("1. Dari direktori proyek saat ini", 'yellow'))
        print(colored("2. Dari direktori data", 'yellow'))
        print(colored("3. Dari lokasi eksternal", 'yellow'))

        choice = self.prompt("Pilih lokasi sumber", default="1")
        
        # Determine base directory
        if choice == "1":
            base_dir = self.project_root
            self.logger.info(f"📂 Direktori proyek: {base_dir}")
        elif choice == "2":
            base_dir = self.data_dir
            self.logger.info(f"📂 Direktori data: {base_dir}")
        else:
            base_dir = Path(self.prompt("Masukkan path lengkap direktori sumber"))
            if not base_dir.exists():
                self.show_error("❌ Direktori tidak ditemukan!")
                return None

        # Get subdirectory
        subdir = self.prompt("Nama subdirektori sumber (e.g., rupiah_baru)")
        source_dir = base_dir / subdir
        
        if not source_dir.exists():
            self.show_error("❌ Subdirektori sumber tidak ditemukan!")
            return None
            
        return source_dir

    def _get_cleaning_mode(self) -> Optional[str]:
        """Get dataset cleaning mode from user"""
        self.logger.info("\nPilih operasi pembersihan:")
        print(colored("1. Bersihkan semua data", 'red'))
        print(colored("2. Bersihkan hanya data augmentasi", 'yellow'))
        print(colored("3. Bersihkan data training saja", 'yellow'))
        print(colored("4. Bersihkan file rusak", 'yellow'))

        choice = self.prompt("Pilih operasi", default="2")
        
        # Map choice to mode
        mode_map = {
            "1": "all",
            "2": "augmented",
            "3": "training",
            "4": "corrupt"
        }
        
        if choice not in mode_map:
            self.show_error("❌ Pilihan tidak valid!")
            return None
            
        # Show warning for destructive operations
        if choice == "1":
            self.logger.warning("\n⚠️ PERINGATAN: SEMUA DATA akan dihapus!")
        else:
            self.logger.warning("\n⚠️ PERINGATAN: Data yang dihapus tidak dapat dikembalikan!")
            
        return mode_map[choice]

    def _get_currency_augmentation_params(self) -> Dict:
        """Get parameters for currency-specific augmentation"""
        print(colored("\nParameter Augmentasi Mata Uang:", 'cyan'))
        print("1. Variasi Pencahayaan (simulasi kondisi pencahayaan berbeda)")
        print("2. Variasi Sudut (simulasi posisi kamera berbeda)")
        print("3. Variasi Kondisi (simulasi uang lusuh/terlipat)")
        print("4. Semua variasi di atas")
        
        choice = self.prompt("Pilih parameter (1-4)", default="4")
        
        return {
            'lighting': choice in ['1', '4'],
            'geometric': choice in ['2', '4'],
            'condition': choice in ['3', '4']
        }

    def _validate_dataset_structure(self, path: Path) -> bool:
        """Validate dataset directory structure"""
        required_structure = {
            'train': ['images', 'labels'],
            'val': ['images', 'labels'],
            'test': ['images', 'labels']
        }
        
        missing = []
        for split, subdirs in required_structure.items():
            for subdir in subdirs:
                if not (path / split / subdir).exists():
                    missing.append(f"{split}/{subdir}")
        
        if missing:
            self.logger.warning("⚠️ Struktur direktori tidak lengkap!")
            self.logger.warning("Direktori yang tidak ditemukan:")
            for dir_path in missing:
                print(colored(f"  • {dir_path}", 'yellow'))
            return False
            
        return True

    def _validate_image_sizes(self, path: Path) -> bool:
        """Validate image dimensions and formats"""
        min_size = self.cfg.data.min_size
        max_size = self.cfg.data.max_size
        invalid = []
        
        for img_path in path.rglob('*.jpg'):
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    invalid.append((img_path, "Tidak dapat dibaca"))
                    continue
                    
                h, w = img.shape[:2]
                if h < min_size[0] or w < min_size[1]:
                    invalid.append((img_path, f"Terlalu kecil ({w}x{h})"))
                elif h > max_size[0] or w > max_size[1]:
                    invalid.append((img_path, f"Terlalu besar ({w}x{h})"))
            except Exception as e:
                invalid.append((img_path, f"Error: {str(e)}"))
        
        if invalid:
            self.logger.warning("\n⚠️ Ditemukan gambar yang tidak valid:")
            for path, reason in invalid:
                print(colored(f"  • {path.name}: {reason}", 'yellow'))
            return False
            
        return True

    def _validate_labels(self, path: Path) -> bool:
        """Validate label format and content"""
        invalid = []
        
        for label_path in path.rglob('*.txt'):
            try:
                with open(label_path) as f:
                    for i, line in enumerate(f, 1):
                        try:
                            values = list(map(float, line.strip().split()))
                            if len(values) != 5:
                                invalid.append((label_path, f"Baris {i}: Format tidak valid"))
                            elif not (0 <= values[0] <= 6):
                                invalid.append((label_path, f"Baris {i}: Kelas tidak valid"))
                            elif not all(0 <= v <= 1 for v in values[1:]):
                                invalid.append((label_path, f"Baris {i}: Koordinat tidak valid"))
                        except ValueError:
                            invalid.append((label_path, f"Baris {i}: Tidak dapat diparse"))
            except Exception as e:
                invalid.append((label_path, f"Error: {str(e)}"))
        
        if invalid:
            self.logger.warning("\n⚠️ Ditemukan label yang tidak valid:")
            for path, reason in invalid:
                print(colored(f"  • {path.name}: {reason}", 'yellow'))
            return False
            
        return True