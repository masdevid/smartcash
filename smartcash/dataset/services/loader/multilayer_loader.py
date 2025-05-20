"""
File: smartcash/dataset/services/loader/multilayer_loader.py
Deskripsi: Loader khusus untuk dataset multilayer dengan dukungan caching dan optimasi
"""

import torch
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Union, Any

from smartcash.common.logger import get_logger
from smartcash.common.layer_config import get_layer_config
from smartcash.dataset.components.datasets.multilayer_dataset import MultilayerDataset
from smartcash.dataset.utils.transform.image_transform import ImageTransformer


class MultilayerLoader:
    """
    Loader khusus untuk dataset multilayer dengan fitur caching dan optimasi.
    Berfungsi sebagai lapisan antara DatasetLoader dan MultilayerDataset.
    """
    
    def __init__(
        self,
        config: Dict,
        data_dir: str,
        img_size: Tuple[int, int] = (640, 640),
        cache_images: bool = False,
        memory_efficient: bool = True,
        logger=None
    ):
        """
        Inisialisasi MultilayerLoader.
        
        Args:
            config: Konfigurasi aplikasi
            data_dir: Direktori data
            img_size: Ukuran gambar target
            cache_images: Apakah meng-cache gambar di memori
            memory_efficient: Menggunakan mode hemat memori
            logger: Logger kustom (opsional)
        """
        self.config = config
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.cache_images = cache_images
        self.memory_efficient = memory_efficient
        self.logger = logger or get_logger()
        
        # Setup layer config
        self.layer_config = get_layer_config()
        self.active_layers = config.get('layers', self.layer_config.get_layer_names())
        
        # Setup transformer
        self.transformer = ImageTransformer(
            config=self.config,
            img_size=self.img_size,
            logger=self.logger
        )
        
        # Cache untuk dataset
        self._dataset_cache = {}
        
        self.logger.info(
            f"🔄 MultilayerLoader diinisialisasi dengan ukuran gambar {img_size}, "
            f"cache_images={cache_images}, memory_efficient={memory_efficient}"
        )
    
    def get_dataset(
        self, 
        split: str, 
        transform=None, 
        force_reload: bool = False,
        require_all_layers: bool = False,
        layers: List[str] = None
    ) -> MultilayerDataset:
        """
        Dapatkan dataset multilayer untuk split tertentu.
        
        Args:
            split: Split dataset ('train', 'valid', 'test')
            transform: Transformasi kustom (opsional)
            force_reload: Paksa reload dataset meski ada di cache
            require_all_layers: Apakah memerlukan semua layer dalam setiap gambar
            layers: Daftar layer yang akan digunakan (jika None, gunakan active_layers)
            
        Returns:
            Instance dari MultilayerDataset
        """
        # Normalisasi split name
        if split in ('val', 'validation'):
            split = 'valid'
        
        # Generate cache key
        cache_key = f"{split}_{require_all_layers}_{self.img_size[0]}x{self.img_size[1]}"
        if layers:
            cache_key += f"_layers={'_'.join(sorted(layers))}"
            
        # Cek cache jika tidak force_reload
        if not force_reload and cache_key in self._dataset_cache:
            self.logger.debug(f"🔄 Menggunakan dataset {split} dari cache")
            return self._dataset_cache[cache_key]
        
        # Tentukan path split
        split_path = self._get_split_path(split)
        
        # Dapatkan transformasi yang sesuai
        transform = transform or self.transformer.get_transform(split)
        
        # Set layer yang aktif
        active_layers = layers or self.active_layers
        
        # Buat dataset
        try:
            dataset = MultilayerDataset(
                data_path=split_path,
                img_size=self.img_size,
                mode=split,
                transform=transform,
                require_all_layers=require_all_layers,
                layers=active_layers,
                logger=self.logger,
                config=self.config
            )
            
            # Simpan ke cache jika valid
            if len(dataset) > 0:
                self._dataset_cache[cache_key] = dataset
                self.logger.info(f"📊 Dataset '{split}' dibuat dengan {len(dataset)} sampel")
            else:
                self.logger.warning(f"⚠️ Dataset '{split}' kosong, periksa path: {split_path}")
        
            return dataset
            
        except Exception as e:
            self.logger.error(f"❌ Error saat membuat dataset '{split}': {str(e)}")
            # Buat dummy dataset untuk menghindari error
            return MultilayerDataset(
                data_path=split_path,
                img_size=self.img_size,
                mode=split,
                transform=transform,
                require_all_layers=False,
                layers=active_layers,
                logger=self.logger,
                config=self.config
            )
    
    def get_dataset_stats(self, split: str) -> Dict[str, Any]:
        """
        Dapatkan statistik dataset untuk split tertentu.
        
        Args:
            split: Split dataset ('train', 'valid', 'test')
            
        Returns:
            Dictionary berisi statistik dataset
        """
        stats = {'status': 'unknown', 'sample_count': 0, 'layer_stats': {}, 'class_stats': {}}
        
        try:
            dataset = self.get_dataset(split)
            
            if len(dataset) > 0:
                stats.update({
                    'status': 'valid',
                    'sample_count': len(dataset),
                    'layer_stats': dataset.get_layer_statistics(),
                    'class_stats': dataset.get_class_statistics()
                })
            else:
                stats['status'] = 'empty'
                
        except Exception as e:
            stats.update({
                'status': 'error',
                'error': str(e)
            })
            
        return stats
    
    def clear_cache(self):
        """Bersihkan cache dataset untuk menghemat memori."""
        cache_size = len(self._dataset_cache)
        self._dataset_cache.clear()
        self.logger.info(f"🧹 Cache dibersihkan ({cache_size} dataset)")
        
        # Panggil garbage collector
        import gc
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def _get_split_path(self, split: str) -> Path:
        """
        Dapatkan path untuk split dataset tertentu.
        
        Args:
            split: Split dataset ('train', 'valid', 'test')
            
        Returns:
            Path ke direktori split
        """
        # Normalisasi nama split
        if split in ('val', 'validation'):
            split = 'valid'
            
        # Cek konfigurasi khusus untuk path split
        split_paths = self.config.get('data', {}).get('local', {})
        if split in split_paths:
            return Path(split_paths[split])
            
        # Fallback ke path default
        return self.data_dir / split