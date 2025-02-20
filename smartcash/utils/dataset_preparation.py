"""
Dataset preparation utilities for SmartCash
"""

import os
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import yaml
import cv2
import numpy as np
from tqdm.auto import tqdm

from smartcash.utils.logger import SmartCashLogger
from smartcash.utils.preprocessing import ImagePreprocessor

class DatasetPreparator:
    """Helper class for dataset preparation"""
    
    def __init__(
        self,
        config_path: str,
        logger: Optional[SmartCashLogger] = None
    ):
        self.logger = logger or SmartCashLogger(__name__)
        self.config = self._load_config(config_path)
        self.preprocessor = ImagePreprocessor(
            config_path=config_path,
            logger=self.logger
        )
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def prepare_dataset(
        self,
        data_dir: str,
        output_dir: str,
        split: str = 'train',
        augment: bool = True
    ) -> Tuple[List[np.ndarray], List[str]]:
        """
        Prepare dataset by preprocessing images and labels
        
        Args:
            data_dir: Input data directory
            output_dir: Output directory for processed data
            split: Dataset split (train/valid/test)
            augment: Whether to apply augmentation
            
        Returns:
            Tuple of (processed_images, processed_labels)
        """
        # Setup paths
        image_dir = Path(data_dir) / split / 'images'
        label_dir = Path(data_dir) / split / 'labels'
        out_image_dir = Path(output_dir) / split / 'images'
        out_label_dir = Path(output_dir) / split / 'labels'
        
        # Create output dirs
        out_image_dir.mkdir(parents=True, exist_ok=True)
        out_label_dir.mkdir(parents=True, exist_ok=True)
        
        # Get file lists
        image_files = sorted(image_dir.glob('*.jpg'))
        label_files = sorted(label_dir.glob('*.txt'))
        
        if len(image_files) != len(label_files):
            raise ValueError(
                f"Mismatch between images ({len(image_files)}) "
                f"and labels ({len(label_files)})"
            )
            
        # Process each image-label pair
        processed_images = []
        processed_labels = []
        
        self.logger.info(f"Processing {split} dataset...")
        for img_path, lbl_path in tqdm(
            zip(image_files, label_files),
            total=len(image_files)
        ):
            # Process image and label
            img, lbl = self.preprocessor.process_image_and_label(
                image_path=str(img_path),
                label_path=str(lbl_path),
                save_dir=str(out_image_dir),
                augment=augment
            )
            
            if img is not None and lbl is not None:
                processed_images.append(img)
                processed_labels.append(lbl)
                
                # Save label
                out_lbl_path = out_label_dir / lbl_path.name
                with open(out_lbl_path, 'w') as f:
                    f.write(lbl)
                    
        self.logger.info(
            f"✅ Processed {len(processed_images)} {split} samples"
        )
        
        return processed_images, processed_labels
        
    def validate_dataset(
        self,
        data_dir: str,
        split: str = 'train'
    ) -> bool:
        """
        Validate processed dataset
        
        Args:
            data_dir: Data directory to validate
            split: Dataset split to validate
            
        Returns:
            True if validation passes
        """
        image_dir = Path(data_dir) / split / 'images'
        label_dir = Path(data_dir) / split / 'labels'
        
        # Check directories exist
        if not image_dir.exists() or not label_dir.exists():
            self.logger.error(f"❌ Missing directories for {split} split")
            return False
            
        # Check file counts match
        image_files = list(image_dir.glob('*.jpg'))
        label_files = list(label_dir.glob('*.txt'))
        
        if len(image_files) != len(label_files):
            self.logger.error(
                f"❌ Mismatch between images ({len(image_files)}) "
                f"and labels ({len(label_files)}) for {split} split"
            )
            return False
            
        # Validate each pair
        for img_path in image_files:
            # Check label exists
            lbl_path = label_dir / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                self.logger.error(f"❌ Missing label for {img_path}")
                return False
                
            # Check image can be read
            img = cv2.imread(str(img_path))
            if img is None:
                self.logger.error(f"❌ Cannot read image {img_path}")
                return False
                
            # Check image dimensions
            h, w = img.shape[:2]
            if h != self.config['model']['img_size'][0] or \
               w != self.config['model']['img_size'][1]:
                self.logger.error(
                    f"❌ Wrong dimensions for {img_path}: "
                    f"got {(h,w)}, expected {self.config['model']['img_size']}"
                )
                return False
                
        self.logger.info(f"✅ Validation passed for {split} split")
        return True
