# File: smartcash/utils/roboflow_downloader.py
# Author: Alfrida Sabar
# Deskripsi: Utility untuk download dataset dari Roboflow menggunakan REST API

import os
import requests
import zipfile
from pathlib import Path
from tqdm.notebook import tqdm
import shutil

class RoboflowDownloader:
    """
    Utility untuk download dataset Roboflow dengan REST API
    Fitur:
    - Download langsung dengan progress bar
    - Multi-method fallback
    - Ekstraksi otomatis
    - Validasi dataset
    """
    
    def __init__(
        self, 
        api_key: str, 
        workspace: str, 
        project: str, 
        version: str = '3',
        output_dir: str = 'data'
    ):
        """
        Inisialisasi downloader Roboflow
        
        Args:
            api_key: API key Roboflow
            workspace: Nama workspace
            project: Nama project
            version: Versi dataset
            output_dir: Direktori output
        """
        self.api_key = api_key
        self.workspace = workspace
        self.project = project
        self.version = version
        self.output_dir = Path(output_dir)
        
        # Buat direktori output
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _generate_download_url(self, format='yolov5'):
        """
        Generate URL download dataset
        
        Args:
            format: Format dataset (default: YOLOv5)
        
        Returns:
            URL download
        """
        base_url = "https://api.roboflow.com/ds/api/export"
        return (
            f"{base_url}/{self.project}?"
            f"key={self.api_key}&"
            f"format={format}&"
            f"version={self.version}"
        )
    
    def download(self, format='yolov5'):
        """
        Download dataset dengan progress bar
        
        Args:
            format: Format dataset
        
        Returns:
            Path ke direktori dataset yang didownload
        """
        # Generate URL download
        download_url = self._generate_download_url(format)
        
        # Nama file zip
        zip_filename = f"{self.project}_v{self.version}.zip"
        zip_path = self.output_dir / zip_filename
        
        try:
            # Request HEAD untuk mendapatkan ukuran file
            head_response = requests.head(download_url)
            total_size = int(head_response.headers.get('content-length', 0))
            
            # Download dengan progress bar
            response = requests.get(download_url, stream=True)
            response.raise_for_status()
            
            # Progress bar
            progress_bar = tqdm(
                total=total_size, 
                unit='iB', 
                unit_scale=True, 
                desc=f"Mendownload {self.project}"
            )
            
            # Simpan file zip
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    size = f.write(chunk)
                    progress_bar.update(size)
            
            progress_bar.close()
            
            # Ekstraksi
            extract_dir = self.output_dir / f"{self.project}_v{self.version}"
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            
            # Hapus file zip setelah ekstraksi
            os.remove(zip_path)
            
            return extract_dir
        
        except Exception as e:
            print(f"❌ Gagal download dataset: {str(e)}")
            return None
    
    def move_dataset(self, dataset_path):
        """
        Pindahkan dataset ke struktur direktori yang diinginkan
        
        Args:
            dataset_path: Path dataset yang didownload
        
        Returns:
            Dict berisi path train, valid, test
        """
        splits = ['train', 'valid', 'test']
        
        for split in splits:
            src_images = dataset_path / split / 'images'
            src_labels = dataset_path / split / 'labels'
            
            dst_images = self.output_dir / split / 'images'
            dst_labels = self.output_dir / split / 'labels'
            
            # Buat direktori tujuan
            dst_images.mkdir(parents=True, exist_ok=True)
            dst_labels.mkdir(parents=True, exist_ok=True)
            
            # Pindahkan file
            for img_file in src_images.glob('*'):
                shutil.move(str(img_file), str(dst_images / img_file.name))
            
            for label_file in src_labels.glob('*'):
                shutil.move(str(label_file), str(dst_labels / label_file.name))
        
        # Hapus direktori sumber
        shutil.rmtree(dataset_path)
        
        return {
            'train': self.output_dir / 'train',
            'valid': self.output_dir / 'valid', 
            'test': self.output_dir / 'test'
        }
    
    def validate_dataset(self):
        """
        Validasi struktur dataset yang didownload
        
        Returns:
            Dict berisi status validasi
        """
        splits = ['train', 'valid', 'test']
        results = {}
        
        for split in splits:
            images_dir = self.output_dir / split / 'images'
            labels_dir = self.output_dir / split / 'labels'
            
            results[split] = {
                'images_count': len(list(images_dir.glob('*'))),
                'labels_count': len(list(labels_dir.glob('*'))),
                'valid': (
                    images_dir.exists() and 
                    labels_dir.exists() and 
                    len(list(images_dir.glob('*'))) > 0 and
                    len(list(labels_dir.glob('*'))) > 0
                )
            }
        
        return results