# File: src/data/augmentation.py
# Author: Alfrida Sabar
# Deskripsi: Augmentasi data untuk SmartCash Detector

import albumentations as A
import numpy as np

class RupiahAugmentation:
    def __init__(self):
        self.transform = A.Compose([
            A.OneOf([
                A.RandomBrightnessContrast(p=0.8),
                A.RandomGamma(p=0.8),
            ], p=0.5),
            A.OneOf([
                A.GaussNoise(p=0.8),
                A.ISONoise(p=0.8),
            ], p=0.3),
            A.OneOf([
                A.MotionBlur(p=0.8),
                A.GaussianBlur(p=0.8),
            ], p=0.2),
            A.OneOf([
                A.RandomRain(p=0.8),
                A.RandomShadow(p=0.8),
            ], p=0.3),
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels']
        ))

    def __call__(self, image, labels):
        if len(labels) == 0:
            return image, labels
            
        transformed = self.transform(
            image=image,
            bboxes=labels[:, :4],
            class_labels=labels[:, 4]
        )
        
        if len(transformed['bboxes']):
            labels = np.column_stack([
                transformed['bboxes'],
                transformed['class_labels']
            ])
            
        return transformed['image'], labels