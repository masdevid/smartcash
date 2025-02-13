# File: src/interfaces/data_management_interface.py
# Author: Alfrida Sabar
# Deskripsi: Antarmuka untuk manajemen dataset dengan pendekatan context-driven



# File: src/interfaces/data_interface.py
# Author: Alfrida Sabar
# Deskripsi: Antarmuka untuk manajemen dataset dengan pendekatan context-driven

from pathlib import Path
from typing import Dict, Optional

from .base_interface import BaseInterface
from interfaces.data_manager import DataManager
from config.manager import ConfigManager

class DataInterface(BaseInterface):
    """Antarmuka untuk operasi manajemen dataset"""
    def __init__(self, config: ConfigManager):
        super().__init__()
        self.cfg = config
        self.data_manager = DataManager(config)
        
    def tampilkan_menu(self) -> str:
        """Tampilkan menu manajemen dataset"""
        menu = """
📦 Menu Manajemen Dataset:

1. Persiapan Dataset
   └─ Buat struktur direktori dan konfigurasi
2. Verifikasi Dataset
   └─ Periksa integritas dan konsistensi data
3. Informasi Dataset
   └─ Tampilkan statistik dan detail dataset
4. Pembersihan Dataset
   └─ Hapus data korup atau tidak valid
5. Augmentasi Data
   └─ Perluas dan variasikan dataset

0. Kembali ke Menu Utama
"""
        return self.prompt(menu, color='cyan')

    def handle_menu(self):
        """Tangani pilihan menu manajemen dataset"""
        while True:
            try:
                pilihan = self.tampilkan_menu()
                
                menu_actions = {
                    '0': lambda: True,
                    '1': self._handle_preparation,
                    '2': self._handle_verification,
                    '3': self._handle_dataset_info,
                    '4': self._handle_cleaning,
                    '5': self._handle_augmentation
                }
                
                # Eksekusi aksi atau minta ulang
                action = menu_actions.get(pilihan)
                if action is None:
                    self.show_error("❌ Pilihan menu tidak valid!")
                    continue
                
                # Jalankan aksi
                should_exit = action()
                if should_exit:
                    break
                
                # Konfirmasi kembali ke menu
                if not self.confirm("\nKembali ke menu Manajemen Dataset?"):
                    break
                    
            except KeyboardInterrupt:
                self.logger.warning("\n⚠️ Operasi dibatalkan oleh pengguna")
            except Exception as e:
                self.show_error(f"❌ Terjadi kesalahan: {str(e)}")

    def _handle_preparation(self):
        """Tangani persiapan dataset"""
        # Validasi dan persiapan direktori
        if (self.data_manager.validate_directory(self.data_manager.data_dir) and 
            self.data_manager.prepare_dataset()):
            self.show_success("✅ Persiapan dataset berhasil!")
        else:
            self.show_error("❌ Gagal mempersiapkan dataset")
        return False

    def _handle_verification(self):
        """Tangani verifikasi dataset"""
        self.logger.info("\n🔍 Memverifikasi Dataset...")
        
        is_verified = self.data_manager.verify_dataset(plot=True)
        
        if is_verified:
            self.show_success("✅ Dataset valid dan siap digunakan!")
        else:
            self.show_error("❌ Ditemukan masalah dalam dataset")
        return False

    def _handle_dataset_info(self):
        """Tampilkan informasi dan statistik dataset"""
        self.logger.info("\n📊 Informasi Dataset:")
        
        try:
            # Tampilkan informasi dataset
            self.data_manager.print_dataset_info()
            
            # Analisis komprehensif
            analysis = self.data_manager.analyze_dataset()
            
            # Generate rekomendasi
            recs = analysis.get('augmentation_recommendations', {})
            if recs.get('general'):
                print("\n💡 Rekomendasi:")
                for rec in recs['general']:
                    print(f"  • {rec.get('description', 'Rekomendasi umum')}")
                    for action in rec.get('actions', []):
                        print(f"    - {action}")
        except Exception as e:
            self.logger.error(f"Gagal mendapatkan analisis: {str(e)}")
        return False

    def _handle_cleaning(self):
        """Tangani pembersihan dataset"""
        self.logger.info("\n🧹 Membersihkan Dataset...")
        
        modes = {
            '1': 'all',
            '2': 'augmented',
            '3': 'training',
            '4': 'corrupt'
        }
        
        print("\nPilih Mode Pembersihan:")
        print("1. Hapus Semua Data")
        print("2. Hapus Data Augmentasi")
        print("3. Hapus Data Training")
        print("4. Hapus Data Korup")
        print("0. Batal")
        
        mode_choice = self.prompt("Pilih mode", default='0')
        
        # Keluar jika dibatalkan
        if mode_choice == '0':
            return False
        
        # Validasi pilihan
        if mode_choice not in modes:
            self.show_error("❌ Pilihan tidak valid!")
            return False
        
        # Konfirmasi pembersihan
        if not self.confirm(f"Yakin hapus data '{modes[mode_choice]}'?"):
            return False
        
        # Jalankan pembersihan
        try:
            stats = self.data_manager.clean_dataset(modes[mode_choice])
            
            print(f"  • Total file dihapus: {stats.get('removed', 0)}")
            if stats.get('errors', 0) > 0:
                self.show_error(f"⚠️ Terjadi {stats['errors']} kesalahan")
            else:
                self.show_success("✅ Pembersihan dataset berhasil!")
        except Exception as e:
            self.show_error(f"❌ Gagal membersihkan dataset: {str(e)}")
        
        return False

    def _handle_augmentation(self):
        """Tangani augmentasi dataset"""
        self.logger.info("\n🔄 Memulai Augmentasi Dataset...")
        
        augmentation_handler = self.data_manager.handlers['augmentation']
        
        # Dapatkan rekomendasi faktor
        recommended_factor = augmentation_handler.get_recommended_factor()
        
        # Tampilkan deskripsi augmentasi
        descriptions = augmentation_handler.get_augmentation_descriptions()
        
        print("\nTipe Augmentasi Tersedia:")
        for key, desc in descriptions.items():
            print(f"\n{desc['name']}:")
            print(f"  Deskripsi: {desc['description']}")
            print("  Fitur:")
            for feature in desc.get('features', []):
                print(f"    • {feature}")
        
        # Pilih faktor dan mode
        factor = int(self.prompt("Faktor augmentasi", default=str(recommended_factor)))
        
        modes_options = {
            '1': ['lighting'],
            '2': ['geometric'],
            '3': ['condition'],
            '4': ['lighting', 'geometric', 'condition']
        }
        
        print("\nPilih Mode Augmentasi:")
        print("1. Pencahayaan")
        print("2. Geometrik")
        print("3. Kondisi")
        print("4. Semua Mode")
        
        mode_choice = self.prompt("Pilih mode", default='4')
        
        # Validasi pilihan
        if mode_choice not in modes_options:
            self.show_error("❌ Pilihan tidak valid!")
            return False
        
        # Jalankan augmentasi
        try:
            stats = self.data_manager.augment_dataset(
                factor=factor, 
                modes=modes_options[mode_choice]
            )
            
            print(f"  • Total diproses: {stats.get('processed', 0)}")
            print(f"  • Data ditambahkan: {stats.get('augmented', 0)}")
            if stats.get('errors', 0) > 0:
                self.show_error(f"⚠️ Terjadi {stats['errors']} kesalahan")
            else:
                self.show_success("✅ Augmentasi dataset berhasil!")
        except Exception as e:
            self.show_error(f"❌ Gagal melakukan augmentasi: {str(e)}")
        
        return False