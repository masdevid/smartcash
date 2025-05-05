"""
File: smartcash/dataset/services/preprocessor/storage.py
Deskripsi: Komponen pengelolaan penyimpanan untuk hasil preprocessing dataset
"""

import os
import json
import shutil
import time
from pathlib import Path
from typing import Dict, Any, Optional
import logging

class PreprocessedStorage:
    """Manager penyimpanan untuk hasil preprocessing dataset dengan fitur metadata."""
    
    def __init__(self, output_dir: str, logger=None):
        """
        Inisialisasi storage manager.
        
        Args:
            output_dir: Direktori output untuk hasil preprocessing
            logger: Logger untuk logging (opsional)
        """
        self.output_dir = Path(output_dir)
        self.logger = logger or logging.getLogger(__name__)
        self.metadata_dir = self.output_dir / 'metadata'
        
        # Buat direktori utama jika belum ada
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
    
    def get_split_path(self, split: str) -> Path:
        """
        Dapatkan path direktori untuk split dataset tertentu.
        
        Args:
            split: Nama split ('train', 'valid', 'test')
            
        Returns:
            Path direktori split
        """
        return self.output_dir / split
    
    def update_stats(self, split: str, stats: Dict[str, Any]) -> None:
        """
        Update statistik preprocessing untuk split tertentu.
        
        Args:
            split: Nama split
            stats: Statistik preprocessing
        """
        try:
            # Buat atau update file statistik
            stats_file = self.metadata_dir / f"{split}_stats.json"
            
            # Convert values yang tidak JSON serializable
            serializable_stats = self._make_serializable(stats)
            
            with open(stats_file, 'w') as f:
                json.dump(serializable_stats, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"❌ Gagal menyimpan statistik: {str(e)}")
    
    def save_metadata(self, split: str, image_id: str, metadata: Dict[str, Any]) -> None:
        """
        Simpan metadata untuk gambar yang dipreprocess.
        
        Args:
            split: Nama split dataset
            image_id: ID gambar (filename tanpa ekstensi)
            metadata: Metadata untuk disimpan
        """
        try:
            # Buat direktori metadata untuk split jika belum ada
            split_metadata_dir = self.metadata_dir / split
            split_metadata_dir.mkdir(parents=True, exist_ok=True)
            
            # Truncate ID gambar jika terlalu panjang (maksimal 50 karakter)
            if len(image_id) > 50:
                truncated_id = image_id[-50:]
            else:
                truncated_id = image_id
                
            # Sanitasi filename
            safe_id = truncated_id.replace('/', '_').replace('\\', '_')
            
            # Simpan metadata
            metadata_file = split_metadata_dir / f"{safe_id}.json"
            
            # Convert values yang tidak JSON serializable
            serializable_metadata = self._make_serializable(metadata)
            
            with open(metadata_file, 'w') as f:
                json.dump(serializable_metadata, f, indent=2)
                
        except Exception as e:
            # Jangan break proses preprocessing, hanya log error
            self.logger.warning(f"⚠️ Gagal menyimpan metadata untuk {image_id}: {str(e)}")
    
    def get_stats(self, split: str) -> Dict[str, Any]:
        """
        Dapatkan statistik preprocessing untuk split tertentu.
        
        Args:
            split: Nama split
            
        Returns:
            Dictionary statistik atau empty dict jika tidak ditemukan
        """
        stats_file = self.metadata_dir / f"{split}_stats.json"
        
        if not stats_file.exists():
            return {}
            
        try:
            with open(stats_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"❌ Gagal membaca statistik: {str(e)}")
            return {}
    
    def clean_storage(self, split: Optional[str] = None) -> None:
        """
        Bersihkan storage untuk split tertentu atau semua split.
        
        Args:
            split: Nama split yang akan dibersihkan, atau None untuk semua direktori
        """
        try:
            if split:
                # Bersihkan split tertentu
                split_path = self.get_split_path(split)
                if split_path.exists():
                    shutil.rmtree(split_path)
                    self.logger.info(f"🧹 Direktori {split_path} berhasil dibersihkan")
                
                # Bersihkan metadata untuk split
                split_metadata_path = self.metadata_dir / split
                if split_metadata_path.exists():
                    shutil.rmtree(split_metadata_path)
                    self.logger.info(f"🧹 Metadata untuk {split} berhasil dibersihkan")
                
                # Hapus file statistik untuk split
                stats_file = self.metadata_dir / f"{split}_stats.json"
                if stats_file.exists():
                    os.remove(stats_file)
            else:
                # Bersihkan semua direktori kecuali metadata
                for item in self.output_dir.iterdir():
                    if item.is_dir() and item.name != 'metadata':
                        shutil.rmtree(item)
                        self.logger.info(f"🧹 Direktori {item} berhasil dibersihkan")
                
                # Bersihkan semua file metadata
                if self.metadata_dir.exists():
                    for item in self.metadata_dir.iterdir():
                        if item.is_file() or (item.is_dir() and item.name != '.gitkeep'):
                            if item.is_dir():
                                shutil.rmtree(item)
                            else:
                                os.remove(item)
                            
                self.logger.info(f"🧹 Metadata berhasil dibersihkan")
                
                # Buat ulang direktori metadata
                self.metadata_dir.mkdir(parents=True, exist_ok=True)
                
        except Exception as e:
            self.logger.error(f"❌ Error saat membersihkan storage: {str(e)}")
            
    def _make_serializable(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert dictionary values ke format yang dapat di-serialisasi JSON.
        
        Args:
            data: Dictionary yang akan dikonversi
            
        Returns:
            Dictionary dengan values yang dapat di-serialisasi
        """
        result = {}
        
        for key, value in data.items():
            if isinstance(value, (str, int, float, bool, type(None))):
                result[key] = value
            elif isinstance(value, (list, tuple)):
                result[key] = [self._make_serializable_value(v) for v in value]
            elif isinstance(value, dict):
                result[key] = self._make_serializable(value)
            else:
                # Convert types yang tidak didukung ke string
                result[key] = str(value)
                
        return result
    
    def _make_serializable_value(self, value: Any) -> Any:
        """
        Convert nilai tunggal ke format yang dapat di-serialisasi JSON.
        
        Args:
            value: Nilai yang akan dikonversi
            
        Returns:
            Nilai yang dapat di-serialisasi
        """
        if isinstance(value, (str, int, float, bool, type(None))):
            return value
        elif isinstance(value, (list, tuple)):
            return [self._make_serializable_value(v) for v in value]
        elif isinstance(value, dict):
            return self._make_serializable(value)
        else:
            return str(value)