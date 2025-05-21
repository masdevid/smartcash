"""
File: smartcash/dataset/services/augmentor/dataset_augmentor.py
Deskripsi: Implementasi augmentor untuk dataset
"""

from typing import Dict, Any, List, Optional, Tuple
import os
import logging
from pathlib import Path
import concurrent.futures
from tqdm import tqdm

from smartcash.dataset.services.augmentor.image_augmentor import ImageAugmentor
from smartcash.dataset.services.augmentor.bbox_augmentor import BBoxAugmentor
from smartcash.dataset.services.augmentor.pipeline_factory import AugmentationPipelineFactory
from smartcash.dataset.services.augmentor.augmentation_worker import AugmentationWorker
from smartcash.dataset.services.augmentor.class_balancer import ClassBalancer
from smartcash.common.logger import get_logger

logger = get_logger(__name__)

class DatasetAugmentor:
    """
    Kelas untuk melakukan augmentasi pada dataset
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Inisialisasi DatasetAugmentor
        
        Args:
            config: Dictionary konfigurasi augmentasi
        """
        self.config = config
        self.image_augmentor = ImageAugmentor()
        self.bbox_augmentor = BBoxAugmentor()
        self.pipeline_factory = AugmentationPipelineFactory()
        self.class_balancer = ClassBalancer(config)
        
    def augment_dataset(
        self, 
        dataset_path: str, 
        output_dir: str,
        create_symlinks: bool = False
    ) -> bool:
        """
        Lakukan augmentasi pada dataset
        
        Args:
            dataset_path: Path ke dataset
            output_dir: Path untuk menyimpan hasil augmentasi
            create_symlinks: Buat symlink ke folder preprocessing
            
        Returns:
            True jika berhasil, False jika gagal
        """
        try:
            # Validasi input
            if not os.path.exists(dataset_path):
                logger.error(f"❌ Dataset path tidak ditemukan: {dataset_path}")
                return False
                
            # Buat output directory jika belum ada
            os.makedirs(output_dir, exist_ok=True)
            
            # Dapatkan pipeline augmentasi
            pipeline = self.pipeline_factory.create_pipeline(self.config)
            
            # Inisialisasi worker
            worker = AugmentationWorker(
                image_augmentor=self.image_augmentor,
                bbox_augmentor=self.bbox_augmentor,
                pipeline=pipeline,
                config=self.config
            )
            
            # Dapatkan daftar file yang akan diaugmentasi
            image_files = self._get_image_files(dataset_path)
            
            # Lakukan augmentasi dengan parallel processing
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.config['augmentation']['num_workers']) as executor:
                futures = []
                for image_file in image_files:
                    future = executor.submit(
                        worker.process_image,
                        image_file,
                        output_dir
                    )
                    futures.append(future)
                    
                # Tunggu semua proses selesai
                for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Augmenting images"):
                    try:
                        result = future.result()
                        if not result:
                            logger.warning("⚠️ Gagal mengaugmentasi beberapa gambar")
                    except Exception as e:
                        logger.error(f"❌ Error saat augmentasi: {str(e)}")
                        
            # Lakukan class balancing jika diaktifkan
            if self.config['augmentation']['balance_classes']:
                self.class_balancer.balance_classes(output_dir)
                
            logger.info("✅ Augmentasi dataset selesai")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saat augmentasi dataset: {str(e)}")
            return False
            
    def _get_image_files(self, dataset_path: str) -> List[str]:
        """
        Dapatkan daftar file gambar dari dataset
        
        Args:
            dataset_path: Path ke dataset
            
        Returns:
            List path file gambar
        """
        image_files = []
        for root, _, files in os.walk(dataset_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_files.append(os.path.join(root, file))
        return image_files 