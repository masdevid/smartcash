# File: src/data/prepare.py
# Author: Alfrida Sabar
# Deskripsi: Persiapan dataset dan manajemen augmentasi

import cv2
from pathlib import Path
from tqdm import tqdm
from termcolor import colored
import albumentations as A
import re

class DatasetPrep:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.augment = A.Compose([
            A.RandomBrightnessContrast(p=0.5),
            A.GaussNoise(p=0.3),
            A.Rotate(limit=20, p=0.5),
            A.RandomShadow(p=0.3),
            A.Blur(blur_limit=3, p=0.2),
        ])

    def remove_augmented_data(self, split='train'):
        """Hapus data hasil augmentasi dari dataset"""
        print(colored('🔄 Mencari data augmentasi...', 'cyan'))
        
        img_dir = self.data_dir / split / 'images'
        label_dir = self.data_dir / split / 'labels'
        
        # Pattern untuk mencocokkan file augmentasi (nama_aug0.jpg, nama_aug1.jpg, dst)
        aug_pattern = re.compile(r'.*_aug\d+\.(jpg|txt)$')
        
        # Hapus file gambar yang diaugmentasi
        removed_images = 0
        for img_file in tqdm(list(img_dir.glob('*.jpg')), desc='Menghapus gambar'):
            if aug_pattern.match(img_file.name):
                img_file.unlink()
                removed_images += 1
        
        # Hapus file label yang diaugmentasi
        removed_labels = 0
        for label_file in tqdm(list(label_dir.glob('*.txt')), desc='Menghapus label'):
            if aug_pattern.match(label_file.name):
                label_file.unlink()
                removed_labels += 1
        
        print(colored(f'✅ Berhasil menghapus {removed_images} gambar dan {removed_labels} label hasil augmentasi', 'green'))

    def prepare_dataset(self):
        """Persiapan struktur folder dataset"""
        print(colored('📁 Menyiapkan struktur dataset...', 'cyan'))
        
        splits = ['train', 'val', 'test']
        subdirs = ['images', 'labels']
        
        for split in splits:
            for subdir in subdirs:
                (self.data_dir / split / subdir).mkdir(parents=True, exist_ok=True)
        
        print(colored('✅ Struktur folder siap', 'green'))

    def apply_augmentation(self, split='train', aug_factor=2):
        """Aplikasi augmentasi pada dataset training"""
        img_dir = self.data_dir / split / 'images'
        print(colored(f'🔄 Mengaplikasikan augmentasi pada {split}...', 'cyan'))
        
        for img_path in tqdm(list(img_dir.glob('*.jpg')), desc='Augmentasi'):
            if '_aug' in img_path.stem:  # Skip file yang sudah diaugmentasi
                continue
                
            img = cv2.imread(str(img_path))
            label_path = img_path.parent.parent / 'labels' / f'{img_path.stem}.txt'
            
            if not label_path.exists():
                continue
                
            # Load labels
            with open(label_path) as f:
                labels = [line.strip().split() for line in f]
            
            # Apply augmentation
            for i in range(aug_factor):
                augmented = self.augment(image=img)
                aug_img = augmented['image']
                
                # Save augmented image and copy labels
                aug_name = f'{img_path.stem}_aug{i}{img_path.suffix}'
                cv2.imwrite(str(img_dir / aug_name), aug_img)
                
                aug_label_path = label_path.parent / f'{img_path.stem}_aug{i}.txt'
                with open(aug_label_path, 'w') as f:
                    f.writelines([' '.join(label) + '\n' for label in labels])
        
        print(colored('✅ Augmentasi selesai', 'green'))

    def verify_dataset(self):
        """Verifikasi integritas dataset"""
        print(colored('🔍 Memverifikasi dataset...', 'cyan'))
        
        for split in ['train', 'val', 'test']:
            img_dir = self.data_dir / split / 'images'
            label_dir = self.data_dir / split / 'labels'
            
            n_images = len(list(img_dir.glob('*.jpg')))
            n_labels = len(list(label_dir.glob('*.txt')))
            
            # Hitung file augmentasi
            n_aug_images = len([f for f in img_dir.glob('*.jpg') if '_aug' in f.stem])
            n_aug_labels = len([f for f in label_dir.glob('*.txt') if '_aug' in f.stem])
            
            print(colored(f'📊 {split}:', 'yellow'), 
                  colored(f'{n_images} images ({n_aug_images} augmented),', 'cyan'),
                  colored(f'{n_labels} labels ({n_aug_labels} augmented)', 'cyan'))
            
            if n_images != n_labels:
                print(colored(f'⚠️ Peringatan: Jumlah image dan label tidak sama di {split}', 'red'))