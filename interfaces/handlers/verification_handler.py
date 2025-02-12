# File: src/interfaces/handlers/verification_handler.py
# Author: Alfrida Sabar
# Deskripsi: Handler untuk verifikasi dan validasi dataset

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from interfaces.handlers.base_handler import BaseHandler

class DataVerificationHandler(BaseHandler):
    """Handler for dataset verification operations"""
    def __init__(self, config):
        super().__init__(config)
        self.min_size = getattr(config.data, 'min_size', (32, 32))
        self.max_size = getattr(config.data, 'max_size', (1920, 1920))
        self.valid_classes = range(7)  # 7 kelas mata uang
        
    def verify_dataset(self) -> Dict:
        """Verifikasi komprehensif dataset"""
        self.logger.info("🔍 Memulai verifikasi dataset...")
        results = {}
        
        try:
            # Verifikasi per split
            for split in ['train', 'val', 'test']:
                results[split] = self._verify_split(split)
                
            # Update operation stats
            self.update_stats('verification', results)
            
            # Log completion
            self.log_operation("Verifikasi dataset", "success")
            
        except Exception as e:
            self.log_operation("Verifikasi dataset", "failed", str(e))
            
        return results
        
    def _verify_split(self, split: str) -> Dict:
        """Verifikasi satu split dataset"""
        img_dir = self.rupiah_dir / split / 'images'
        label_dir = self.rupiah_dir / split / 'labels'
        
        stats = {
            'images': 0,
            'labels': 0,
            'corrupt': 0,
            'invalid': 0,
            'augmented': 0,
            'original': 0,
            'size_issues': [],
            'label_issues': [],
            'pairing_issues': []
        }
        
        if not img_dir.exists() or not label_dir.exists():
            return stats
            
        # Image validation
        stats.update(self._verify_images(img_dir))
        
        # Label validation
        stats.update(self._verify_labels(label_dir))
        
        # Pairing validation
        stats.update(self._verify_pairs(img_dir, label_dir))
        
        return stats
        
    def _verify_images(self, img_dir: Path) -> Dict:
        """Verifikasi integritas dan format gambar"""
        stats = {
            'images': 0,
            'corrupt': 0,
            'augmented': 0,
            'original': 0,
            'size_issues': []
        }
        
        for img_path in img_dir.glob('*.jpg'):
            stats['images'] += 1
            
            # Track augmented vs original
            if '_aug' in img_path.stem:
                stats['augmented'] += 1
            else:
                stats['original'] += 1
                
            try:
                # Check image integrity
                img = cv2.imread(str(img_path))
                if img is None:
                    stats['corrupt'] += 1
                    continue
                    
                # Check dimensions
                h, w = img.shape[:2]
                if h < self.min_size[0] or w < self.min_size[1]:
                    stats['size_issues'].append({
                        'file': img_path.name,
                        'issue': f'Terlalu kecil ({w}x{h})',
                        'current': (w, h),
                        'expected': f">= {self.min_size}"
                    })
                elif h > self.max_size[0] or w > self.max_size[1]:
                    stats['size_issues'].append({
                        'file': img_path.name,
                        'issue': f'Terlalu besar ({w}x{h})',
                        'current': (w, h),
                        'expected': f"<= {self.max_size}"
                    })
                    
            except Exception as e:
                stats['corrupt'] += 1
                self.logger.error(f"Error pada {img_path.name}: {str(e)}")
                
        return stats
        
    def _verify_labels(self, label_dir: Path) -> Dict:
        """Verifikasi format dan validitas label"""
        stats = {
            'labels': 0,
            'invalid': 0,
            'label_issues': []
        }
        
        for label_path in label_dir.glob('*.txt'):
            stats['labels'] += 1
            
            try:
                with open(label_path) as f:
                    for i, line in enumerate(f, 1):
                        try:
                            # Parse values
                            values = list(map(float, line.strip().split()))
                            
                            # Validate format
                            if len(values) != 5:
                                stats['invalid'] += 1
                                stats['label_issues'].append({
                                    'file': label_path.name,
                                    'line': i,
                                    'issue': 'Format tidak valid',
                                    'found': len(values),
                                    'expected': 5
                                })
                                continue
                                
                            # Validate class
                            if values[0] not in self.valid_classes:
                                stats['invalid'] += 1
                                stats['label_issues'].append({
                                    'file': label_path.name,
                                    'line': i,
                                    'issue': 'Kelas tidak valid',
                                    'found': values[0],
                                    'expected': f"0-{len(self.valid_classes)-1}"
                                })
                                
                            # Validate coordinates
                            if not all(0 <= v <= 1 for v in values[1:]):
                                stats['invalid'] += 1
                                stats['label_issues'].append({
                                    'file': label_path.name,
                                    'line': i,
                                    'issue': 'Koordinat tidak valid',
                                    'found': values[1:],
                                    'expected': '0-1'
                                })
                                
                        except ValueError:
                            stats['invalid'] += 1
                            stats['label_issues'].append({
                                'file': label_path.name,
                                'line': i,
                                'issue': 'Tidak dapat diparse'
                            })
                            
            except Exception as e:
                stats['invalid'] += 1
                self.logger.error(f"Error pada {label_path.name}: {str(e)}")
                
        return stats
        
    def _verify_pairs(self, img_dir: Path, label_dir: Path) -> Dict:
        """Verifikasi pasangan gambar-label"""
        stats = {'pairing_issues': []}
        
        # Check orphaned labels
        for label_path in label_dir.glob('*.txt'):
            img_path = img_dir / f"{label_path.stem}.jpg"
            if not img_path.exists():
                stats['pairing_issues'].append({
                    'file': label_path.name,
                    'issue': 'Label tanpa gambar'
                })
                
        # Check orphaned images
        for img_path in img_dir.glob('*.jpg'):
            label_path = label_dir / f"{img_path.stem}.txt"
            if not label_path.exists():
                stats['pairing_issues'].append({
                    'file': img_path.name,
                    'issue': 'Gambar tanpa label'
                })
                
        return stats
        
    def get_verification_summary(self) -> Dict:
        """Generate summary of verification results"""
        results = self.get_stats('verification')
        if not results:
            return {}
            
        summary = {
            'total_images': sum(s['images'] for s in results.values()),
            'total_labels': sum(s['labels'] for s in results.values()),
            'corrupt_files': sum(s['corrupt'] for s in results.values()),
            'invalid_labels': sum(s['invalid'] for s in results.values()),
            'original_images': sum(s['original'] for s in results.values()),
            'augmented_images': sum(s['augmented'] for s in results.values()),
            'issues': {
                'size': sum(len(s['size_issues']) for s in results.values()),
                'label': sum(len(s['label_issues']) for s in results.values()),
                'pairing': sum(len(s['pairing_issues']) for s in results.values())
            }
        }
        
        return summary
        
    def check_dataset_readiness(self) -> Tuple[bool, List[str]]:
        """Check if dataset is ready for training"""
        issues = []
        
        # Get verification results
        results = self.get_stats('verification')
        if not results:
            return False, ['Verifikasi dataset belum dilakukan']
            
        # Check minimum requirements
        for split in ['train', 'val', 'test']:
            split_stats = results[split]
            
            # Check for images
            if split_stats['images'] == 0:
                issues.append(f"Tidak ada gambar di split {split}")
                
            # Check for labels
            if split_stats['labels'] == 0:
                issues.append(f"Tidak ada label di split {split}")
                
            # Check for corrupt files
            if split_stats['corrupt'] > 0:
                issues.append(f"Terdapat {split_stats['corrupt']} file rusak di {split}")
                
            # Check for invalid labels
            if split_stats['invalid'] > 0:
                issues.append(f"Terdapat {split_stats['invalid']} label tidak valid di {split}")
                
        # Dataset is ready if there are no issues
        return len(issues) == 0, issues