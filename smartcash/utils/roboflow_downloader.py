# File: smartcash/utils/roboflow_downloader.py
# Author: Alfrida Sabar
# Deskripsi: Utility untuk mengelola download dan pemindahan dataset dari Roboflow API

import os
import shutil
import yaml
from pathlib import Path
from typing import Dict, Optional, Union
from tqdm.notebook import tqdm

class RoboflowDownloader:
    """
    Downloader untuk dataset Roboflow dengan fungsi otomatis untuk:
    - Mengunduh dataset dari Roboflow API
    - Memindahkan data ke struktur direktori yang sesuai
    - Membersihkan file-file sementara
    - Memperbarui konfigurasi secara otomatis
    """
    
    def __init__(
        self, 
        config: Dict,
        output_dir: Union[str, Path] = 'data',
        config_path: str = 'configs/experiment_config.yaml'
    ):
        """
        Inisialisasi downloader Roboflow
        
        Args:
            config: Konfigurasi berisi kredensial Roboflow
            output_dir: Direktori keluaran (default: 'data')
            config_path: Path file konfigurasi untuk update
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.config_path = config_path
        
        # Ekstrak informasi dari config
        self.roboflow_config = config.get('data', {}).get('roboflow', {})
        self.api_key = self.roboflow_config.get('api_key') or os.environ.get("ROBOFLOW_API_KEY")
        self.workspace = self.roboflow_config.get('workspace', 'smartcash-wo2us')
        self.project = self.roboflow_config.get('project', 'rupiah-emisi-2022')
        self.version = self.roboflow_config.get('version', '3')
        
    def check_existing_dataset(self) -> Dict[str, int]:
        """
        Periksa keberadaan dataset di direktori output
        
        Returns:
            Dict statistik file yang ditemukan
        """
        stats = {'train': 0, 'valid': 0, 'test': 0, 'total': 0}
        
        for split in ['train', 'valid', 'test']:
            images_dir = self.output_dir / split / 'images'
            if images_dir.exists():
                file_count = len(list(images_dir.glob('*.*')))
                stats[split] = file_count
                stats['total'] += file_count
        
        return stats
    
    def clean_existing_dataset(self) -> None:
        """
        Bersihkan dataset yang ada di direktori output
        """
        print("🧹 Membersihkan dataset yang ada...")
        
        for split in ['train', 'valid', 'test']:
            for subdir in ['images', 'labels']:
                dir_path = self.output_dir / split / subdir
                if dir_path.exists():
                    # Hapus semua file dalam direktori
                    for file in dir_path.glob('*.*'):
                        try:
                            file.unlink()
                        except Exception as e:
                            print(f"⚠️ Gagal menghapus {file}: {str(e)}")
        
        print("✅ Pembersihan selesai.")
    
    def clean_temp_files(self) -> None:
        """
        Bersihkan file-file sementara hasil download
        """
        print("🧹 Mencari dan membersihkan file sementara...")
        
        # Bersihkan file zip yang mungkin tersisa
        for zip_file in Path('.').glob('*.zip'):
            print(f"  → Menghapus file zip: {zip_file}")
            try:
                zip_file.unlink()
            except Exception as e:
                print(f"⚠️ Gagal menghapus file zip {zip_file}: {str(e)}")
                
        # Bersihkan direktori sementara download
        temp_dirs = [d for d in Path('.').glob('*') if d.is_dir() and d.name.startswith('rupiah') and d.name.endswith('-yolov5')]
        for temp_dir in temp_dirs:
            print(f"  → Menghapus direktori sementara: {temp_dir}")
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"⚠️ Gagal menghapus direktori {temp_dir}: {str(e)}")
        
        print("✅ Pembersihan file sementara selesai.")
    
    def download_dataset(self) -> Optional[Path]:
        """
        Download dataset dari Roboflow API
        
        Returns:
            Path direktori dataset yang diunduh atau None jika gagal
        """
        if not self.api_key:
            print("❌ API key tidak ditemukan dalam konfigurasi")
            return None
        
        print(f"🔄 Menyiapkan download dari Roboflow...")
        print(f"   • Workspace: {self.workspace}")
        print(f"   • Project: {self.project}")
        print(f"   • Version: {self.version}")
        
        try:
            from roboflow import Roboflow
            
            print(f"🔌 Menghubungkan ke Roboflow API...")
            rf = Roboflow(api_key=self.api_key)
            
            try:
                workspace_obj = rf.workspace(self.workspace)
                project_obj = workspace_obj.project(self.project)
                version_obj = project_obj.version(self.version)
            except Exception as e:
                print(f"❌ Gagal mengakses project: {str(e)}")
                print(f"   Pastikan workspace, project, dan version sudah benar.")
                return None
            
            print(f"📥 Mengunduh dataset melalui API Roboflow...")
            dataset = version_obj.download("yolov5")
            
            # Validasi hasil download
            dataset_location = Path(dataset.location)
            if not dataset_location.exists():
                print(f"❌ Direktori dataset tidak ditemukan setelah download")
                return None
                
            # Validasi ukuran dataset
            dataset_size = sum(f.stat().st_size for f in dataset_location.glob('**/*') if f.is_file())
            if dataset_size < 10000:  # Kurang dari 10KB
                print(f"❌ Dataset yang diunduh terlalu kecil: {dataset_size} bytes")
                return None
                
            # Validasi isi dataset
            train_images = list((dataset_location / 'train' / 'images').glob('*.*'))
            if not train_images:
                print(f"❌ Tidak ada gambar training yang ditemukan dalam dataset")
                return None
                
            print(f"✅ Dataset berhasil diunduh ke {dataset_location}")
            print(f"   Ukuran: {dataset_size/1024/1024:.2f} MB")
            
            return dataset_location
            
        except Exception as e:
            print(f"❌ Gagal mengunduh dataset: {str(e)}")
            return None
    
    def transfer_dataset(self, source_dir: Path) -> bool:
        """
        Pindahkan dataset dari lokasi sumber ke direktori output
        
        Args:
            source_dir: Direktori sumber dataset
            
        Returns:
            bool: True jika berhasil, False jika gagal
        """
        print(f"🔄 Memindahkan dataset dari {source_dir} ke {self.output_dir}...")
        
        # Validasi direktori sumber
        if not source_dir.exists():
            print(f"❌ Direktori sumber {source_dir} tidak ditemukan")
            return False
            
        # Validasi struktur dataset
        valid_structure = True
        for split in ['train', 'valid', 'test']:
            if not (source_dir / split / 'images').exists():
                print(f"⚠️ Direktori {split}/images tidak ditemukan di sumber")
                valid_structure = False
        
        if not valid_structure:
            print("❌ Struktur dataset tidak valid, membatalkan pemindahan")
            return False
        
        try:
            # Pastikan direktori tujuan ada
            for split in ['train', 'valid', 'test']:
                (self.output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
                (self.output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
            
            # Counter untuk file yang dipindahkan
            files_moved = 0
            
            # Untuk setiap split dataset
            for split in ['train', 'valid', 'test']:
                split_source = source_dir / split
                
                if not split_source.exists():
                    print(f"⚠️ Direktori {split} tidak ditemukan di {source_dir}")
                    continue
                    
                print(f"\n✨ Memproses {split}...")
                
                # Pindahkan gambar
                images_source = split_source / 'images'
                images_target = self.output_dir / split / 'images'
                
                if images_source.exists():
                    image_files = list(images_source.glob('*.*'))
                    
                    for img_file in tqdm(image_files, desc=f"Gambar {split}"):
                        target_file = images_target / img_file.name
                        if not target_file.exists():
                            shutil.copy2(img_file, target_file)
                            files_moved += 1
                
                # Pindahkan label
                labels_source = split_source / 'labels'
                labels_target = self.output_dir / split / 'labels'
                
                if labels_source.exists():
                    label_files = list(labels_source.glob('*.*'))
                    
                    for lbl_file in tqdm(label_files, desc=f"Label {split}"):
                        target_file = labels_target / lbl_file.name
                        if not target_file.exists():
                            shutil.copy2(lbl_file, target_file)
                            files_moved += 1
            
            # Verifikasi pemindahan
            if files_moved == 0:
                print("⚠️ Tidak ada file yang dipindahkan, mungkin dataset kosong")
                return False
            
            print(f"\n✅ Total {files_moved} file dipindahkan ke {self.output_dir}")
            return True
            
        except Exception as e:
            print(f"❌ Gagal memindahkan dataset: {str(e)}")
            return False
    
    def update_config_to_local(self) -> bool:
        """
        Perbarui konfigurasi untuk menggunakan data lokal
        
        Returns:
            bool: True jika berhasil, False jika gagal
        """
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                # Ubah sumber data ke lokal
                config['data']['source'] = 'local'
                
                with open(self.config_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False)
                
                print("✅ Konfigurasi diperbarui: data_source = local")
                return True
        except Exception as e:
            print(f"⚠️ Gagal update konfigurasi: {str(e)}")
            return False
        
        return False
    
    def run_download_pipeline(self, force_download: bool = False) -> Dict:
        """
        Jalankan pipeline lengkap: periksa, bersihkan, unduh, pindahkan, dan perbarui konfigurasi
        
        Args:
            force_download: Paksa download meskipun dataset sudah ada
            
        Returns:
            Dict statistik hasil download
        """
        results = {'success': False, 'stats': {}}
        
        # 1. Periksa dataset yang ada
        existing_stats = self.check_existing_dataset()
        results['stats']['existing'] = existing_stats
        
        if existing_stats['total'] > 0 and not force_download:
            print(f"🔍 Dataset sudah ada: {existing_stats['total']} gambar total")
            print(f"   Train: {existing_stats['train']}, Valid: {existing_stats['valid']}, Test: {existing_stats['test']} gambar")
            print(f"💡 Gunakan force_download=True untuk mengunduh ulang")
            results['success'] = True
            return results
        
        # 2. Bersihkan dataset yang ada jika perlu
        if existing_stats['total'] > 0:
            self.clean_existing_dataset()
        
        # 3. Download dataset baru
        dataset_location = self.download_dataset()
        if not dataset_location:
            print("❌ Gagal mengunduh dataset baru")
            return results
        
        # 4. Pindahkan dataset
        transfer_success = self.transfer_dataset(dataset_location)
        if not transfer_success:
            print("❌ Gagal memindahkan dataset")
            return results
        
        # 5. Perbarui konfigurasi
        self.update_config_to_local()
        
        # 6. Bersihkan file-file sementara
        self.clean_temp_files()
        
        # 7. Periksa hasil akhir
        final_stats = self.check_existing_dataset()
        results['stats']['final'] = final_stats
        results['success'] = True
        
        print(f"\n✅ Pipeline selesai: {final_stats['total']} gambar dataset siap")
        print(f"   Train: {final_stats['train']}, Valid: {final_stats['valid']}, Test: {final_stats['test']} gambar")
        
        return results

# Fungsi bantuan untuk digunakan di cell Jupyter
def download_dataset_from_roboflow(config, force=False):
    """
    Fungsi bantuan untuk mengunduh dataset dari Roboflow di notebook
    
    Args:
        config: Konfigurasi berisi kredensial Roboflow
        force: Paksa download meskipun dataset sudah ada
    
    Returns:
        Dict hasil download
    """
    downloader = RoboflowDownloader(config)
    return downloader.run_download_pipeline(force_download=force)