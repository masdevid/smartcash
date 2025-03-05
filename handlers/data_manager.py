import os
import torch
import cv2
import numpy as np
import albumentations as A
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from torch.utils.data import Dataset, DataLoader
from concurrent.futures import ThreadPoolExecutor
import random
import yaml
from tqdm.auto import tqdm

from smartcash.utils.logger import SmartCashLogger, get_logger
from smartcash.utils.preprocessing_cache import PreprocessingCache

class MultilayerDataset(Dataset):
    """Dataset for multi-layer banknote detection."""
    
    def __init__(
        self,
        data_path: str,
        img_size: Tuple[int, int] = (640, 640),
        mode: str = 'train',
        transform = None,
        logger: Optional[SmartCashLogger] = None
    ):
        """
        Initialize dataset.
        
        Args:
            data_path: Path to dataset directory
            img_size: Target image size
            mode: Dataset mode ('train', 'val', 'test')
            transform: Custom transformations
            logger: Logger instance
        """
        self.data_path = Path(data_path)
        self.img_size = img_size
        self.mode = mode
        self.transform = transform
        self.logger = logger or get_logger("multilayer_dataset", log_to_file=False)
        
        # Layer configuration
        self.layer_config = {
            'banknote': {
                'classes': ['001', '002', '005', '010', '020', '050', '100'],
                'class_ids': list(range(7))  # 0-6
            },
            'nominal': {
                'classes': ['l2_001', 'l2_002', 'l2_005', 'l2_010', 'l2_020', 'l2_050', 'l2_100'],
                'class_ids': list(range(7, 14))  # 7-13
            },
            'security': {
                'classes': ['l3_sign', 'l3_text', 'l3_thread'],
                'class_ids': list(range(14, 17))  # 14-16
            }
        }
        
        # Setup paths
        self.images_dir = self.data_path / 'images'
        self.labels_dir = self.data_path / 'labels'
        
        if not self.images_dir.exists():
            self.logger.warning(f"Images directory not found: {self.images_dir}")
            self.images_dir.mkdir(parents=True, exist_ok=True)
            
        if not self.labels_dir.exists():
            self.logger.warning(f"Labels directory not found: {self.labels_dir}")
            self.labels_dir.mkdir(parents=True, exist_ok=True)
            
        self.image_files = self._find_image_files()
        
        if len(self.image_files) == 0:
            self.logger.warning(f"No images found in {self.images_dir}")
            
    def _find_image_files(self) -> List[Path]:
        """Find all image files with various extensions."""
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(list(self.images_dir.glob(ext)))
        return sorted(image_files)
        
    def __len__(self) -> int:
        return len(self.image_files)
        
    def _normalize_bbox(self, bbox: List[float]) -> List[float]:
        """
        Normalize bounding box coordinates to ensure they're in [0, 1] range.
        
        Args:
            bbox: Bounding box coordinates [x_center, y_center, width, height]
            
        Returns:
            Normalized coordinates [x_center, y_center, width, height]
        """
        # Ensure x_center and y_center are in [0, 1]
        x_center = max(0.0, min(1.0, bbox[0]))
        y_center = max(0.0, min(1.0, bbox[1]))
        
        # Ensure width and height are valid
        width = max(0.01, min(1.0, bbox[2]))
        if x_center + width/2 > 1.0:
            width = 2 * (1.0 - x_center)
            
        height = max(0.01, min(1.0, bbox[3]))
        if y_center + height/2 > 1.0:
            height = 2 * (1.0 - y_center)
            
        return [x_center, y_center, width, height]
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get dataset item with bounding box validation."""
        try:
            # Handle empty dataset
            if len(self.image_files) == 0:
                dummy_img = torch.zeros((3, self.img_size[1], self.img_size[0]))
                dummy_targets = torch.zeros((17, 5))  # 17 classes, 5 values (x,y,w,h,conf)
                return dummy_img, dummy_targets
                
            img_path = self.image_files[idx]
            label_path = self.labels_dir / f"{img_path.stem}.txt"
            
            # Load and validate image
            img = cv2.imread(str(img_path))
            if img is None:
                self.logger.warning(f"Cannot read image: {img_path}")
                dummy_img = torch.zeros((3, self.img_size[1], self.img_size[0]))
                dummy_targets = torch.zeros((17, 5))
                return dummy_img, dummy_targets
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Load and validate labels
            bboxes = []
            class_labels = []
            
            if label_path.exists():
                try:
                    with open(label_path, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                class_id = int(float(parts[0]))
                                coords = list(map(float, parts[1:5]))
                                normalized_coords = self._normalize_bbox(coords)
                                bboxes.append(normalized_coords)
                                class_labels.append(class_id)
                except Exception as e:
                    self.logger.warning(f"Error reading label {label_path}: {str(e)}")
            
            # Apply transformations
            if self.transform:
                transformed = self.transform(image=img, bboxes=bboxes, class_labels=class_labels)
            else:
                # Default transform (resize and normalize)
                transform = A.Compose([
                    A.Resize(height=self.img_size[1], width=self.img_size[0]),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
                transformed = transform(image=img, bboxes=bboxes, class_labels=class_labels)
                
            # Validate transformed bboxes
            validated_bboxes = []
            validated_class_labels = []
            
            for bbox, cls_id in zip(transformed['bboxes'], transformed['class_labels']):
                try:
                    for coord in bbox:
                        if not (0 <= coord <= 1):
                            raise ValueError(f"Invalid bbox coordinate {bbox}")
                    validated_bboxes.append(bbox)
                    validated_class_labels.append(cls_id)
                except ValueError as e:
                    self.logger.warning(f"Skipping invalid bbox: {bbox}")
                    continue
                    
            # Update transformed data
            transformed['bboxes'] = validated_bboxes
            transformed['class_labels'] = validated_class_labels
            
            # Prepare final tensors
            img_tensor = torch.from_numpy(transformed['image']).permute(2, 0, 1).float()
            
            # Ensure all class labels are integers
            class_labels = [int(cl) for cl in transformed['class_labels']]
            
            # Prepare multi-layer targets
            total_classes = sum(len(layer['classes']) for layer in self.layer_config.values())
            targets = torch.zeros((total_classes, 5))  # [class_id, x, y, w, h]
            
            for bbox, cls_id in zip(transformed['bboxes'], class_labels):
                if 0 <= cls_id < total_classes:
                    x_center, y_center, width, height = bbox
                    targets[cls_id, 0] = x_center
                    targets[cls_id, 1] = y_center
                    targets[cls_id, 2] = width
                    targets[cls_id, 3] = height
                    targets[cls_id, 4] = 1.0  # Confidence
                    
            return img_tensor, targets
            
        except Exception as e:
            self.logger.error(f"Error loading item {idx}: {str(e)}")
            # Return zero tensors as fallback
            img_tensor = torch.zeros((3, self.img_size[1], self.img_size[0]))
            total_classes = sum(len(layer['classes']) for layer in self.layer_config.values())
            targets = torch.zeros((total_classes, 5))
            return img_tensor, targets

class DataManager:
    """
    Unified data management class for SmartCash that handles:
    - Data loading and preprocessing
    - Dataset preparation and splitting
    - Data augmentation
    - Cache management
    - Multi-layer dataset handling
    """
    
    def __init__(
        self,
        config_path: str,
        data_dir: Optional[str] = None,
        cache_size_gb: float = 1.0,
        logger: Optional[SmartCashLogger] = None
    ):
        """
        Initialize the DataManager.
        
        Args:
            config_path: Path to configuration file
            data_dir: Base directory for dataset (optional)
            cache_size_gb: Size of preprocessing cache in GB
            logger: Custom logger instance
        """
        self.logger = logger or get_logger("data_manager")
        self.config = self._load_config(config_path)
        
        # Setup paths
        self.data_dir = Path(data_dir or self.config.get('data_dir', 'data'))
        self.target_size = tuple(self.config['model']['img_size'])
        
        # Initialize cache
        self.cache = PreprocessingCache(
            max_size_gb=cache_size_gb,
            logger=self.logger
        )
        
        # Layer configuration for multi-layer detection
        self.layer_config = {
            'banknote': {
                'classes': ['001', '002', '005', '010', '020', '050', '100'],
                'class_ids': list(range(7))  # 0-6
            },
            'nominal': {
                'classes': ['l2_001', 'l2_002', 'l2_005', 'l2_010', 'l2_020', 'l2_050', 'l2_100'],
                'class_ids': list(range(7, 14))  # 7-13
            },
            'security': {
                'classes': ['l3_sign', 'l3_text', 'l3_thread'],
                'class_ids': list(range(14, 17))  # 14-16
            }
        }
        
        # Setup augmentation pipelines
        self._setup_augmentation_pipelines()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
            
    def _setup_augmentation_pipelines(self):
        """Setup data augmentation pipelines."""
        # Base transformations
        self.base_transform = A.Compose([
            A.Resize(height=self.target_size[1], width=self.target_size[0]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        
        # Training augmentations
        self.train_transform = A.Compose([
            A.Resize(height=self.target_size[1], width=self.target_size[0]),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=0.7
            ),
            A.RandomShadow(p=0.5),
            A.SafeRotate(limit=30, p=0.7),
            A.Perspective(scale=(0.05, 0.1), p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=45,
                p=0.5
            ),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        
    def prepare_dataset(
        self,
        input_dir: str,
        output_dir: str,
        augment: bool = True,
        num_workers: int = 4
    ) -> None:
        """
        Prepare dataset by preprocessing images and labels.
        
        Args:
            input_dir: Input data directory
            output_dir: Output directory for processed data
            augment: Whether to apply augmentation
            num_workers: Number of worker threads
        """
        self.logger.info(f"🎬 Starting dataset preparation from {input_dir}")
        
        try:
            input_dir = Path(input_dir)
            output_dir = Path(output_dir)
            
            # Setup directories
            images_dir = input_dir / 'images'
            labels_dir = input_dir / 'labels'
            out_images_dir = output_dir / 'images'
            out_labels_dir = output_dir / 'labels'
            
            os.makedirs(out_images_dir, exist_ok=True)
            os.makedirs(out_labels_dir, exist_ok=True)
            
            # Get image files
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                image_files.extend(list(images_dir.glob(ext)))
                
            if not image_files:
                self.logger.warning(f"No images found in {images_dir}")
                return
                
            def process_image_label_pair(image_path: Path):
                try:
                    # Find corresponding label
                    label_path = labels_dir / f"{image_path.stem}.txt"
                    
                    # Process image
                    img = cv2.imread(str(image_path))
                    if img is None:
                        self.logger.warning(f"Cannot read image: {image_path}")
                        return
                        
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Get labels if exists
                    bboxes = []
                    class_labels = []
                    if label_path.exists():
                        with open(label_path, 'r') as f:
                            for line in f:
                                parts = line.strip().split()
                                if len(parts) >= 5:
                                    class_id = int(float(parts[0]))
                                    coords = list(map(float, parts[1:5]))
                                    bboxes.append(coords)
                                    class_labels.append(class_id)
                    
                    # Apply transformations
                    transform = self.train_transform if augment else self.base_transform
                    transformed = transform(
                        image=img,
                        bboxes=bboxes,
                        class_labels=class_labels
                    )
                    
                    # Save processed image
                    processed_img = cv2.cvtColor(
                        transformed['image'],
                        cv2.COLOR_RGB2BGR
                    )
                    cv2.imwrite(
                        str(out_images_dir / image_path.name),
                        processed_img
                    )
                    
                    # Save processed label
                    if bboxes:
                        with open(out_labels_dir / label_path.name, 'w') as f:
                            for bbox, cls_id in zip(
                                transformed['bboxes'],
                                transformed['class_labels']
                            ):
                                f.write(f"{cls_id} {' '.join(map(str, bbox))}\n")
                                
                except Exception as e:
                    self.logger.error(f"Error processing {image_path}: {str(e)}")
            
            # Process files in parallel
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                list(tqdm(
                    executor.map(process_image_label_pair, image_files),
                    total=len(image_files),
                    desc="💫 Processing"
                ))
                
            self.logger.success(
                f"✨ Dataset preparation completed!\n"
                f"💾 Output: {output_dir}"
            )
            
        except Exception as e:
            self.logger.error(f"❌ Dataset preparation failed: {str(e)}")
            raise e
            
    def split_dataset(
        self,
        data_dir: str,
        train_ratio: float = 0.7,
        valid_ratio: float = 0.15,
        test_ratio: float = 0.15
    ) -> Dict[str, int]:
        """
        Split dataset into train, validation and test sets.
        
        Args:
            data_dir: Data directory containing images and labels
            train_ratio: Ratio for training set
            valid_ratio: Ratio for validation set
            test_ratio: Ratio for test set
            
        Returns:
            Dict with split statistics
        """
        # Validate ratios
        total_ratio = train_ratio + valid_ratio + test_ratio
        if not np.isclose(total_ratio, 1.0):
            self.logger.warning(
                f"Total ratio should be 1.0, got {total_ratio}. "
                "Using default ratios."
            )
            train_ratio, valid_ratio, test_ratio = 0.7, 0.15, 0.15
            
        data_dir = Path(data_dir)
        images_dir = data_dir / 'images'
        labels_dir = data_dir / 'labels'
        
        # Get all image files
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(list(images_dir.glob(ext)))
            
        if not image_files:
            self.logger.error(f"No images found in {images_dir}")
            return {'train': 0, 'valid': 0, 'test': 0}
            
        # Shuffle files
        random.shuffle(image_files)
        
        # Calculate split sizes
        n_total = len(image_files)
        n_train = int(n_total * train_ratio)
        n_valid = int(n_total * valid_ratio)
        n_test = n_total - n_train - n_valid
        
        # Create split directories
        splits = {
            'train': image_files[:n_train],
            'valid': image_files[n_train:n_train + n_valid],
            'test': image_files[n_train + n_valid:]
        }
        
        # Move files to split directories
        for split_name, split_files in splits.items():
            split_dir = data_dir / split_name
            split_images_dir = split_dir / 'images'
            split_labels_dir = split_dir / 'labels'
            
            os.makedirs(split_images_dir, exist_ok=True)
            os.makedirs(split_labels_dir, exist_ok=True)
            
            for img_path in split_files:
                # Move image
                shutil.copy2(
                    img_path,
                    split_images_dir / img_path.name
                )
                
                # Move label if exists
                label_path = labels_dir / f"{img_path.stem}.txt"
                if label_path.exists():
                    shutil.copy2(
                        label_path,
                        split_labels_dir / label_path.name
                    )
                    
        split_stats = {
            'train': n_train,
            'valid': n_valid,
            'test': n_test
        }
        
        self.logger.success(
            f"✨ Dataset split completed!\n"
            f"📊 Statistics:\n"
            f"   Train: {n_train} images\n"
            f"   Valid: {n_valid} images\n"
            f"   Test: {n_test} images"
        )
        
        return split_stats
        
    def get_dataloader(
        self,
        data_path: str,
        batch_size: int = 32,
        num_workers: int = 4,
        mode: str = 'train'
    ) -> DataLoader:
        """
        Get DataLoader for specified dataset split.
        
        Args:
            data_path: Path to dataset directory
            batch_size: Batch size
            num_workers: Number of worker processes
            mode: Dataset mode ('train', 'val', 'test')
            
        Returns:
            DataLoader instance
        """
        dataset = MultilayerDataset(
            data_path=data_path,
            img_size=self.target_size,
            mode=mode,
            transform=self.train_transform if mode == 'train' else self.base_transform,
            logger=self.logger
        )
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=mode == 'train',
            num_workers=num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True
        )
        
    def _collate_fn(self, batch):
        """Custom collate function for batching items."""
        # Filter out None values
        batch = [b for b in batch if b is not None and isinstance(b, tuple) and len(b) == 2]
        
        if len(batch) == 0:
            return (
                torch.zeros((0, 3, self.target_size[0], self.target_size[1])),
                torch.zeros((0, 17, 5))
            )
            
        imgs, targets = zip(*batch)
        
        # Stack images
        imgs = torch.stack(imgs)
        
        # Stack targets if they're tensors
        if isinstance(targets[0], torch.Tensor):
            targets = torch.stack(targets)
            
        return imgs, targets
        
    def get_class_names(self) -> List[str]:
        """Get list of all class names across layers."""
        class_names = []
        for layer_info in self.layer_config.values():
            class_names.extend(layer_info['classes'])
        return class_names
        
    def get_dataset_stats(self, data_dir: str) -> Dict[str, Dict[str, int]]:
        """
        Get dataset statistics.
        
        Args:
            data_dir: Data directory to analyze
            
        Returns:
            Dict with statistics per split
        """
        stats = {}
        data_dir = Path(data_dir)
        
        for split in ['train', 'valid', 'test']:
            split_dir = data_dir / split
            if not split_dir.exists():
                continue
                
            images_dir = split_dir / 'images'
            labels_dir = split_dir / 'labels'
            
            n_images = len(list(images_dir.glob('*.jpg'))) if images_dir.exists() else 0
            n_labels = len(list(labels_dir.glob('*.txt'))) if labels_dir.exists() else 0
            
            stats[split] = {
                'images': n_images,
                'labels': n_labels,
                'complete_pairs': min(n_images, n_labels)
            }
            
        return stats
        
    def clear_cache(self) -> None:
        """Clear preprocessing cache."""
        try:
            self.cache.clear()
            self.logger.success("🧹 Cache cleared successfully")
        except Exception as e:
            self.logger.error(f"❌ Failed to clear cache: {str(e)}")
            raise e
