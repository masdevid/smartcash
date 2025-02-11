# File: src/handlers/augmentation_handlers.py
# Author: Alfrida Sabar
# Deskripsi: Handler untuk operasi augmentasi SmartCash Detector

from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Optional
from data.augmentation import RupiahAugmentation
from utils.logging import ColoredLogger

class AugmentationHandler:
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.logger = ColoredLogger('AugmentationHandler')

    def apply_augmentation(self, 
                          mode: str,
                          aug_factor: int,
                          params: Optional[Dict] = None) -> Dict:
        """Apply data augmentation"""
        if mode == 'currency':
            return self._apply_currency_augmentation(aug_factor, params)
        elif mode == 'advanced':
            return self._apply_advanced_augmentation(aug_factor)
        else:
            return self._apply_standard_augmentation(aug_factor)

    def _apply_currency_augmentation(self, aug_factor: int, params: Dict) -> Dict:
        """Apply currency-specific augmentation"""
        stats = {'processed': 0, 'augmented': 0, 'errors': 0}
        
        try:
            augmentor = RupiahAugmentation(
                enable_lighting=params.get('lighting', True),
                enable_geometric=params.get('geometric', True),
                enable_condition=params.get('condition', True)
            )
            
            img_dir = self.data_dir / 'train' / 'images'
            label_dir = self.data_dir / 'train' / 'labels'
            
            # Get list of original images
            img_files = [f for f in img_dir.glob('*.jpg') if '_aug' not in f.stem]
            total_files = len(img_files)
            
            with tqdm(total=total_files, desc="Applying currency augmentation") as pbar:
                for img_path in img_files:
                    try:
                        # Read image and label
                        img = cv2.imread(str(img_path))
                        label_path = label_dir / f"{img_path.stem}.txt"
                        
                        if label_path.exists():
                            with open(label_path) as f:
                                labels = np.array([
                                    list(map(float, line.strip().split()))
                                    for line in f
                                ])
                            
                            # Apply augmentation
                            for i in range(aug_factor):
                                augmented = augmentor(image=img, labels=labels)
                                
                                # Save augmented image
                                aug_img_path = img_dir / f"{img_path.stem}_aug{i}.jpg"
                                cv2.imwrite(str(aug_img_path), augmented['image'])
                                
                                # Save augmented labels
                                aug_label_path = label_dir / f"{img_path.stem}_aug{i}.txt"
                                with open(aug_label_path, 'w') as f:
                                    for label in augmented['labels']:
                                        f.write(' '.join(map(str, label)) + '\n')
                                
                                stats['augmented'] += 1
                            
                            stats['processed'] += 1
                            
                    except Exception as e:
                        self.logger.error(f"Error processing {img_path}: {str(e)}")
                        stats['errors'] += 1
                        
                    pbar.update(1)
                    
        except Exception as e:
            self.logger.error(f"Error in currency augmentation: {str(e)}")
            stats['errors'] += 1
            
        return stats

    def _apply_advanced_augmentation(self, aug_factor: int) -> Dict:
        """Apply advanced augmentation"""
        stats = {'processed': 0, 'augmented': 0, 'errors': 0}
        
        # TODO: Implement advanced augmentation
        
        return stats

    def _apply_standard_augmentation(self, aug_factor: int) -> Dict:
        """Apply standard augmentation"""
        stats = {'processed': 0, 'augmented': 0, 'errors': 0}
        
        # TODO: Implement standard augmentation
        
        return stats

    def get_augmentation_info(self) -> Dict:
        """Get information about dataset size and recommended augmentation"""
        img_dir = self.data_dir / 'train' / 'images'
        
        # Count original and augmented images
        stats = {
            'original': len([f for f in img_dir.glob('*.jpg') if '_aug' not in f.stem]),
            'augmented': len([f for f in img_dir.glob('*.jpg') if '_aug' in f.stem])
        }
        
        # Calculate recommended factor based on dataset size
        if stats['original'] > 1000:
            stats['recommended_factor'] = 2
        elif stats['original'] > 500:
            stats['recommended_factor'] = 4
        else:
            stats['recommended_factor'] = 6
            
        return stats

    def get_augmentation_descriptions(self) -> Dict:
        """Get descriptions of available augmentation types"""
        return {
            'standard': {
                'name': 'Augmentasi Standar',
                'description': 'Rotasi dan variasi pencahayaan dasar',
                'features': [
                    'Rotasi: Simulasi uang dengan orientasi berbeda (±30°)',
                    'Pencahayaan: Variasi kecerahan dan kontras',
                ],
                'use_case': 'Dataset dengan kondisi pencahayaan seragam'
            },
            'advanced': {
                'name': 'Augmentasi Lanjutan',
                'description': 'Termasuk noise dan simulasi kondisi lingkungan',
                'features': [
                    'Semua fitur Augmentasi Standar',
                    'Noise: Simulasi kualitas kamera yang berbeda',
                    'Blur: Simulasi gerakan dan fokus kamera',
                    'Cuaca: Simulasi kondisi pencahayaan ekstrem'
                ],
                'use_case': 'Penggunaan di berbagai kondisi lingkungan'
            },
            'currency': {
                'name': 'Augmentasi Khusus Mata Uang',
                'description': 'Optimized untuk deteksi uang kertas',
                'features': [
                    'Variasi Sudut: Optimized untuk dimensi uang kertas',
                    'Lipatan: Simulasi uang terlipat atau kusut',
                    'Oklusi: Simulasi uang tertutup sebagian',
                    'Degradasi: Simulasi uang lusuh atau rusak'
                ],
                'use_case': 'Deteksi nominal uang dalam kondisi nyata'
            }
        }