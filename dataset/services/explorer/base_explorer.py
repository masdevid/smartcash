"""
File: smartcash/dataset/services/explorer/base_explorer.py
Deskripsi: Fixed base explorer dengan constructor yang konsisten
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
from smartcash.common.logger import get_logger
from smartcash.dataset.utils.dataset_utils import DatasetUtils


class BaseExplorer:
    """Base class untuk dataset explorers dengan fixed constructor."""
    
    def __init__(self, config: Dict[str, Any], data_dir: str, logger=None, num_workers: int = 4):
        """
        Inisialisasi BaseExplorer dengan parameter yang konsisten.
        
        Args:
            config: Konfigurasi aplikasi
            data_dir: Direktori data
            logger: Logger (optional)
            num_workers: Jumlah worker untuk operasi paralel
        """
        self.config = config
        self.data_dir = Path(data_dir)
        self.logger = logger or get_logger()
        self.num_workers = num_workers
        
        # Setup utils tanpa layer_config untuk avoid dependency issues
        self.utils = DatasetUtils(config, data_dir, logger)
    
    def _validate_directories(self, split: str) -> tuple:
        """Validasi direktori split dengan return tuple konsisten."""
        split_path = self.utils.get_split_path(split)
        images_dir = split_path / 'images'
        labels_dir = split_path / 'labels'
        
        valid = (split_path.exists() and 
                images_dir.exists() and 
                labels_dir.exists())
        
        return split_path, images_dir, labels_dir, valid
    
    def _get_valid_files(self, images_dir: Path, labels_dir: Path, sample_size: int = 0) -> List[Path]:
        """Dapatkan file gambar yang memiliki label corresponding."""
        image_files = []
        
        # Get all image files
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(images_dir.glob(ext))
        
        # Filter hanya yang memiliki label
        valid_files = []
        for img_path in image_files:
            label_path = labels_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                valid_files.append(img_path)
        
        # Apply sampling jika diminta
        if sample_size > 0 and len(valid_files) > sample_size:
            import random
            valid_files = random.sample(valid_files, sample_size)
        
        return valid_files