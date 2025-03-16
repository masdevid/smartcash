"""
File: smartcash/dataset/services/preprocessor/storage.py
Deskripsi: Pengelola penyimpanan untuk dataset yang telah dipreprocessing
"""

import os
import json
import shutil
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any

from smartcash.common.logger import get_logger


class PreprocessedStorage:
    """
    Pengelola penyimpanan untuk hasil preprocessing dataset.
    Menyediakan fungsi untuk menyimpan, memuat, dan mengelola metadata.
    """
    
    def __init__(
        self,
        base_dir: Union[str, Path],
        logger: Optional[Any] = None
    ):
        """
        Inisialisasi pengelola penyimpanan preprocessed.
        
        Args:
            base_dir: Direktori dasar untuk penyimpanan
            logger: Logger untuk mencatat aktivitas
        """
        self.logger = logger or get_logger("preprocessed_storage")
        self.base_dir = Path(base_dir)
        
        # Buat direktori jika belum ada
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # File metadata
        self.metadata_file = self.base_dir / 'metadata.json'
        
        # Load metadata jika ada
        self.metadata = self._load_metadata()
        
        self.logger.info(f"💾 PreprocessedStorage diinisialisasi (base_dir: {self.base_dir})")
    
    def _load_metadata(self) -> Dict[str, Any]:
        """
        Load metadata dari file.
        
        Returns:
            Dictionary metadata
        """
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    metadata = json.load(f)
                return metadata
            except Exception as e:
                self.logger.warning(f"⚠️ Gagal memuat metadata: {str(e)}")
        
        # Default metadata jika gagal atau tidak ada
        return {
            'splits': {},
            'config': {},
            'stats': {},
            'version': '1.0'
        }
    
    def _save_metadata(self) -> bool:
        """
        Simpan metadata ke file.
        
        Returns:
            True jika berhasil, False jika gagal
        """
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
            return True
        except Exception as e:
            self.logger.error(f"❌ Gagal menyimpan metadata: {str(e)}")
            return False
    
    def get_split_path(self, split: str) -> Path:
        """
        Dapatkan path untuk split tertentu.
        
        Args:
            split: Nama split ('train', 'val', 'test')
            
        Returns:
            Path direktori untuk split
        """
        split_dir = self.base_dir / split
        split_dir.mkdir(exist_ok=True)
        return split_dir
    
    def get_split_metadata(self, split: str) -> Dict[str, Any]:
        """
        Dapatkan metadata untuk split tertentu.
        
        Args:
            split: Nama split
            
        Returns:
            Dictionary metadata split
        """
        return self.metadata.get('splits', {}).get(split, {})
    
    def update_split_metadata(self, split: str, metadata: Dict[str, Any]) -> bool:
        """
        Update metadata untuk split tertentu.
        
        Args:
            split: Nama split
            metadata: Metadata yang akan diupdate
            
        Returns:
            True jika berhasil, False jika gagal
        """
        if 'splits' not in self.metadata:
            self.metadata['splits'] = {}
            
        # Update metadata
        if split not in self.metadata['splits']:
            self.metadata['splits'][split] = {}
            
        self.metadata['splits'][split].update(metadata)
        
        # Simpan metadata
        return self._save_metadata()
    
    def save_preprocessed_image(
        self,
        split: str,
        image_id: str,
        image_data: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, str]:
        """
        Simpan gambar hasil preprocessing.
        
        Args:
            split: Nama split
            image_id: ID gambar
            image_data: Data gambar hasil preprocessing
            metadata: Metadata tambahan (opsional)
            
        Returns:
            Tuple (success, path)
        """
        split_dir = self.get_split_path(split)
        images_dir = split_dir / 'images'
        images_dir.mkdir(exist_ok=True)
        
        # Buat filename
        img_path = images_dir / f"{image_id}.npy"
        meta_path = images_dir / f"{image_id}_meta.npy"
        
        try:
            # Simpan gambar sebagai numpy array
            np.save(str(img_path), image_data)
            
            # Simpan metadata jika ada
            if metadata:
                np.save(str(meta_path), np.array([metadata], dtype=object))
            
            return True, str(img_path)
        except Exception as e:
            self.logger.error(f"❌ Gagal menyimpan gambar preprocessed: {str(e)}")
            return False, ""
    
    def load_preprocessed_image(
        self,
        split: str,
        image_id: str,
        with_metadata: bool = False
    ) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]:
        """
        Load gambar hasil preprocessing.
        
        Args:
            split: Nama split
            image_id: ID gambar
            with_metadata: Load metadata juga
            
        Returns:
            Tuple (image_data, metadata)
        """
        split_dir = self.get_split_path(split)
        images_dir = split_dir / 'images'
        
        # Path file
        img_path = images_dir / f"{image_id}.npy"
        meta_path = images_dir / f"{image_id}_meta.npy"
        
        try:
            # Load gambar
            if not img_path.exists():
                return None, None
                
            image_data = np.load(str(img_path))
            
            # Load metadata jika diminta dan tersedia
            metadata = None
            if with_metadata and meta_path.exists():
                metadata_array = np.load(str(meta_path), allow_pickle=True)
                if len(metadata_array) > 0:
                    metadata = metadata_array[0]
            
            return image_data, metadata
        except Exception as e:
            self.logger.error(f"❌ Gagal memuat gambar preprocessed: {str(e)}")
            return None, None
    
    def copy_label_file(
        self,
        source_path: Union[str, Path],
        split: str,
        label_id: str
    ) -> Tuple[bool, str]:
        """
        Salin file label ke direktori preprocessed.
        
        Args:
            source_path: Path sumber file label
            split: Nama split
            label_id: ID label
            
        Returns:
            Tuple (success, path)
        """
        split_dir = self.get_split_path(split)
        labels_dir = split_dir / 'labels'
        labels_dir.mkdir(exist_ok=True)
        
        # Path target
        target_path = labels_dir / f"{label_id}.txt"
        
        try:
            # Salin file
            shutil.copy2(source_path, target_path)
            return True, str(target_path)
        except Exception as e:
            self.logger.error(f"❌ Gagal menyalin file label: {str(e)}")
            return False, ""
    
    def list_preprocessed_images(self, split: str) -> List[str]:
        """
        Daftar semua gambar hasil preprocessing untuk split tertentu.
        
        Args:
            split: Nama split
            
        Returns:
            List ID gambar
        """
        split_dir = self.get_split_path(split)
        images_dir = split_dir / 'images'
        
        if not images_dir.exists():
            return []
            
        # List semua file npy dan filter metadata
        image_files = [p.stem for p in images_dir.glob('*.npy') 
                      if not p.stem.endswith('_meta')]
        
        return image_files
    
    def get_stats(self, split: Optional[str] = None) -> Dict[str, Any]:
        """
        Dapatkan statistik penyimpanan.
        
        Args:
            split: Nama split (opsional)
            
        Returns:
            Dictionary statistik
        """
        if split:
            return self.metadata.get('stats', {}).get(split, {})
        else:
            return self.metadata.get('stats', {})
    
    def update_stats(self, split: str, stats: Dict[str, Any]) -> bool:
        """
        Update statistik untuk split tertentu.
        
        Args:
            split: Nama split
            stats: Statistik yang akan diupdate
            
        Returns:
            True jika berhasil, False jika gagal
        """
        if 'stats' not in self.metadata:
            self.metadata['stats'] = {}
            
        # Update stats
        if split not in self.metadata['stats']:
            self.metadata['stats'][split] = {}
            
        self.metadata['stats'][split].update(stats)
        
        # Simpan metadata
        return self._save_metadata()
    
    def clean_storage(self, split: Optional[str] = None) -> bool:
        """
        Bersihkan penyimpanan hasil preprocessing.
        
        Args:
            split: Nama split (jika None, bersihkan semua)
            
        Returns:
            True jika berhasil, False jika gagal
        """
        try:
            if split:
                # Bersihkan satu split
                split_dir = self.get_split_path(split)
                if split_dir.exists():
                    shutil.rmtree(split_dir)
                    split_dir.mkdir(parents=True)
                    
                # Update metadata
                if 'splits' in self.metadata and split in self.metadata['splits']:
                    del self.metadata['splits'][split]
                    
                if 'stats' in self.metadata and split in self.metadata['stats']:
                    del self.metadata['stats'][split]
            else:
                # Bersihkan semua kecuali metadata
                for item in self.base_dir.iterdir():
                    if item.is_dir():
                        shutil.rmtree(item)
                        
                # Reset metadata
                self.metadata = {
                    'splits': {},
                    'config': self.metadata.get('config', {}),
                    'stats': {},
                    'version': self.metadata.get('version', '1.0')
                }
            
            # Simpan metadata
            self._save_metadata()
            
            self.logger.info(
                f"🧹 Berhasil membersihkan penyimpanan: "
                f"{split if split else 'semua split'}"
            )
            
            return True
        except Exception as e:
            self.logger.error(f"❌ Gagal membersihkan penyimpanan: {str(e)}")
            return False