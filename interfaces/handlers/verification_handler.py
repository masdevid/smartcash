# File: src/interfaces/handlers/verification_handler.py
# Author: Alfrida Sabar
# Deskripsi: Handler untuk verifikasi dataset dengan dukungan shape fleksibel dan progress tracking

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from interfaces.handlers.base_handler import BaseHandler

class DataVerificationHandler(BaseHandler):
    """Handler untuk verifikasi dataset dengan dukungan berbagai format label"""
    def __init__(self, config):
        super().__init__(config)
        self.min_size = getattr(config.data, 'min_size', (32, 32))
        self.max_size = getattr(config.data, 'max_size', (1920, 1920))
        self.valid_classes = range(7)  # 7 kelas mata uang
        
    def verify_dataset(self) -> Dict:
        """Verifikasi keseluruhan dataset dengan progress tracking"""
        results = {}
        try:
            for split in ['train', 'val', 'test']:
                results[split] = self._verify_split(split)
            self.update_stats('verification', results)
        except Exception as e:
            self.log_operation("Verifikasi dataset", "failed", str(e))
        return results
        
    def _verify_split(self, split: str) -> Dict:
        """Verifikasi satu split dataset dengan progress tracking"""
        img_dir = self.rupiah_dir / split / 'images'
        label_dir = self.rupiah_dir / split / 'labels'
        
        stats = {
            'images': 0,
            'labels': 0,
            'corrupt': 0,
            'invalid': 0,
            'size_issues': [],
            'label_issues': [],
            'pairing_issues': []
        }
        
        if not img_dir.exists() or not label_dir.exists():
            return stats

        # Verifikasi paralel untuk gambar dan label
        with ThreadPoolExecutor() as executor:
            # Verifikasi gambar
            futures_img = []
            for img_path in img_dir.glob('*.jpg'):
                stats['images'] += 1
                future = executor.submit(self._verify_image, img_path)
                futures_img.append((img_path, future))

            # Verifikasi label
            futures_label = []
            for label_path in label_dir.glob('*.txt'):
                stats['labels'] += 1
                future = executor.submit(self._verify_label, label_path)
                futures_label.append((label_path, future))

            # Track image verification progress
            with tqdm(total=len(futures_img), desc=f"🖼️  Verifikasi gambar {split}") as pbar:
                for img_path, future in futures_img:
                    try:
                        result = future.result()
                        if not result['valid']:
                            stats['corrupt'] += 1
                            if 'size' in result['issues']:
                                stats['size_issues'].append({
                                    'file': img_path.name,
                                    'issue': result['issues']['size']
                                })
                    except Exception:
                        stats['corrupt'] += 1
                    pbar.update(1)

            # Track label verification progress
            with tqdm(total=len(futures_label), desc=f"🏷️  Verifikasi label {split}") as pbar:
                for label_path, future in futures_label:
                    try:
                        result = future.result()
                        if not result['valid']:
                            stats['invalid'] += 1
                            stats['label_issues'].append({
                                'file': label_path.name,
                                'issues': result['issues']
                            })
                    except Exception:
                        stats['invalid'] += 1
                    pbar.update(1)

        # Verifikasi pasangan file
        self._verify_pairs(img_dir, label_dir, stats)
        
        return stats

    def _verify_image(self, img_path: Path) -> Dict:
        """Verifikasi satu file gambar"""
        result = {'valid': True, 'issues': {}}
        
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                result['valid'] = False
                result['issues']['corrupt'] = True
                return result

            h, w = img.shape[:2]
            if h < self.min_size[0] or w < self.min_size[1]:
                result['valid'] = False
                result['issues']['size'] = f'Terlalu kecil ({w}x{h})'
            elif h > self.max_size[0] or w > self.max_size[1]:
                result['valid'] = False
                result['issues']['size'] = f'Terlalu besar ({w}x{h})'
                
        except Exception:
            result['valid'] = False
            result['issues']['corrupt'] = True
            
        return result

    def _verify_label(self, label_path: Path) -> Dict:
        """Verifikasi satu file label dengan dukungan format fleksibel"""
        result = {'valid': True, 'issues': []}
        
        try:
            with open(label_path) as f:
                for line_num, line in enumerate(f, 1):
                    values = line.strip().split()
                    
                    # Validasi kelas (indeks pertama)
                    try:
                        class_idx = int(float(values[0]))
                        if class_idx not in self.valid_classes:
                            result['valid'] = False
                            result['issues'].append(f'Kelas tidak valid: {class_idx}')
                    except (ValueError, IndexError):
                        result['valid'] = False
                        result['issues'].append(f'Format kelas tidak valid')
                        continue

                    # Validasi koordinat (semua nilai setelah indeks kelas)
                    try:
                        coords = [float(v) for v in values[1:]]
                        if not all(0 <= v <= 1 for v in coords):
                            result['valid'] = False
                            result['issues'].append('Koordinat di luar rentang [0,1]')
                    except ValueError:
                        result['valid'] = False
                        result['issues'].append('Format koordinat tidak valid')
                        
        except Exception:
            result['valid'] = False
            result['issues'].append('Gagal membaca file label')
            
        return result

    def _verify_pairs(self, img_dir: Path, label_dir: Path, stats: Dict):
        """Verifikasi pasangan file gambar-label"""
        img_files = set(p.stem for p in img_dir.glob('*.jpg'))
        label_files = set(p.stem for p in label_dir.glob('*.txt'))
        
        orphaned_imgs = img_files - label_files
        orphaned_labels = label_files - img_files
        
        if orphaned_imgs or orphaned_labels:
            with tqdm(total=len(orphaned_imgs) + len(orphaned_labels),
                     desc="🔍 Verifikasi pasangan") as pbar:
                for stem in orphaned_imgs:
                    stats['pairing_issues'].append({
                        'file': f'{stem}.jpg',
                        'issue': 'Gambar tanpa label'
                    })
                    pbar.update(1)
                
                for stem in orphaned_labels:
                    stats['pairing_issues'].append({
                        'file': f'{stem}.txt',
                        'issue': 'Label tanpa gambar'
                    })
                    pbar.update(1)

    def check_dataset_readiness(self) -> Tuple[bool, List[str]]:
        """Periksa kesiapan dataset untuk training"""
        results = self.get_stats('verification')
        if not results:
            return False, ['Verifikasi dataset belum dilakukan']
            
        issues = []
        total_images = 0
        has_all_splits = True
        
        for split in ['train', 'val', 'test']:
            split_stats = results.get(split, {})
            total_images += split_stats.get('images', 0)
            
            if split_stats.get('images', 0) == 0:
                has_all_splits = False
                issues.append(f'Split {split} tidak memiliki gambar')
            if split_stats.get('corrupt', 0) > 0:
                issues.append(f'Terdapat {split_stats["corrupt"]} gambar rusak di {split}')
            if split_stats.get('invalid', 0) > 0:
                issues.append(f'Terdapat {split_stats["invalid"]} label tidak valid di {split}')
                
        if total_images < 100:
            issues.append('Dataset terlalu kecil (minimal 100 gambar)')
            
        return len(issues) == 0 and has_all_splits, issues