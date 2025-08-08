"""
Evaluation data loader for test scenarios.
Handles loading scenario test data (images and labels).
"""

from pathlib import Path
from typing import Dict, List, Optional
from PIL import Image
import numpy as np
import torch

from smartcash.common.logger import get_logger


class EvaluationDataLoader:
    """Load test data for evaluation scenarios with on-the-fly normalization"""
    
    def __init__(self, base_evaluation_dir: Path = None, normalize_images: bool = True, img_size: int = 640):
        self.logger = get_logger('evaluation_data_loader')
        self.base_evaluation_dir = base_evaluation_dir or Path('data/evaluation/scenarios')
        self.normalize_images = normalize_images
        self.img_size = img_size
        
        # Standard ImageNet normalization values
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
    
    def _normalize_image(self, img: Image.Image) -> np.ndarray:
        """Normalize image for evaluation (on-the-fly normalization)"""
        
        # Resize image while maintaining aspect ratio
        img = self._letterbox_resize(img, self.img_size)
        
        # Convert to numpy array and normalize to [0, 1]
        img_array = np.array(img).astype(np.float32) / 255.0
        
        if self.normalize_images:
            # Apply ImageNet normalization
            img_array = (img_array - self.mean) / self.std
        
        # Convert to torch tensor and change from HWC to CHW format
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
        
        return img_tensor
    
    def _letterbox_resize(self, img: Image.Image, target_size: int) -> Image.Image:
        """Resize image with letterbox padding to maintain aspect ratio"""
        
        # Calculate scaling factor
        w, h = img.size
        scale = min(target_size / w, target_size / h)
        
        # Calculate new size
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Resize image
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # Create letterbox image
        letterbox_img = Image.new('RGB', (target_size, target_size), (114, 114, 114))
        
        # Calculate padding offsets
        pad_x = (target_size - new_w) // 2
        pad_y = (target_size - new_h) // 2
        
        # Paste resized image onto letterbox
        letterbox_img.paste(img, (pad_x, pad_y))
        
        return letterbox_img
    
    def load_scenario_data(self, scenario_dir: Path) -> Dict[str, List]:
        """📁 Load scenario test data"""
        
        images_dir = scenario_dir / 'images'
        labels_dir = scenario_dir / 'labels'
        
        # Load images
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png']:
            image_files.extend(list(images_dir.glob(f'*{ext}')))
        
        # Load corresponding labels
        labels = []
        images = []
        
        for img_file in image_files:
            label_file = labels_dir / f"{img_file.stem}.txt"
            
            if label_file.exists():
                # Load and normalize image
                try:
                    img = Image.open(img_file).convert('RGB')
                    
                    # Store both original and normalized versions
                    normalized_img = self._normalize_image(img)
                    
                    images.append({
                        'image': img,  # Original PIL image
                        'normalized_image': normalized_img,  # Normalized tensor
                        'filename': img_file.name,
                        'path': str(img_file)
                    })
                    
                    # Load label
                    annotations = []
                    with open(label_file, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                class_id = int(float(parts[0]))
                                bbox = [float(x) for x in parts[1:5]]
                                annotations.append({
                                    'class_id': class_id,
                                    'bbox': bbox
                                })
                    
                    labels.append({
                        'filename': img_file.name,
                        'annotations': annotations
                    })
                    
                except Exception as e:
                    self.logger.debug(f"⚠️ Error loading {img_file.name}: {str(e)}")
        
        self.logger.info(f"📁 Loaded {len(images)} test images with labels")
        return {'images': images, 'labels': labels}
    
    def load_test_data_from_dir(self, test_dir: Path) -> Dict[str, List]:
        """📁 Load test data from a specific directory"""
        images_dir = test_dir / 'images' if (test_dir / 'images').exists() else test_dir
        labels_dir = test_dir / 'labels' if (test_dir / 'labels').exists() else test_dir
        
        # Handle the case where images and labels are in the same directory
        if not images_dir.exists():
            images_dir = test_dir
        if not labels_dir.exists():
            labels_dir = test_dir
        
        # Load images
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png']:
            if images_dir.exists():
                image_files.extend(list(images_dir.glob(f'*{ext}')))
            if images_dir != test_dir:
                image_files.extend(list(test_dir.glob(f'*{ext}')))
        
        # Remove duplicates
        image_files = list(set(image_files))
        
        # Load corresponding labels
        labels = []
        images = []
        
        for img_file in image_files:
            # Try multiple label file locations
            label_candidates = [
                labels_dir / f"{img_file.stem}.txt",
                test_dir / f"{img_file.stem}.txt",
                img_file.parent / f"{img_file.stem}.txt"
            ]
            
            label_file = None
            for candidate in label_candidates:
                if candidate.exists():
                    label_file = candidate
                    break
            
            if label_file and label_file.exists():
                # Load and normalize image
                try:
                    img = Image.open(img_file).convert('RGB')
                    
                    # Store both original and normalized versions
                    normalized_img = self._normalize_image(img)
                    
                    images.append({
                        'image': img,  # Original PIL image
                        'normalized_image': normalized_img,  # Normalized tensor
                        'filename': img_file.name,
                        'path': str(img_file)
                    })
                    
                    # Load label
                    annotations = []
                    with open(label_file, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                class_id = int(float(parts[0]))
                                bbox = [float(x) for x in parts[1:5]]
                                annotations.append({
                                    'class_id': class_id,
                                    'bbox': bbox
                                })
                    
                    labels.append({
                        'filename': img_file.name,
                        'annotations': annotations
                    })
                    
                except Exception as e:
                    self.logger.debug(f"⚠️ Error loading {img_file.name}: {str(e)}")
            else:
                self.logger.debug(f"⚠️ No label file found for {img_file.name}")
        
        self.logger.info(f"📁 Loaded {len(images)} test images with labels from {test_dir}")
        return {'images': images, 'labels': labels}


def create_evaluation_data_loader(base_evaluation_dir: Path = None) -> EvaluationDataLoader:
    """Factory function to create evaluation data loader"""
    return EvaluationDataLoader(base_evaluation_dir=base_evaluation_dir)