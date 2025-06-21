"""
File: smartcash/dataset/services/explorer/bbox_explorer.py
Deskripsi: Explorer untuk analisis statistik bounding box dalam dataset
"""

import numpy as np
from typing import Dict, Any, List

from smartcash.dataset.services.explorer.base_explorer import BaseExplorer


class BBoxExplorer(BaseExplorer):
    """Explorer khusus untuk analisis bounding box."""
    
    def analyze_bbox_statistics(self, split: str, sample_size: int = 0) -> Dict[str, Any]:
        """
        Analisis statistik bounding box dalam dataset.
        
        Args:
            split: Split dataset yang akan dianalisis
            sample_size: Jumlah sampel (0 = semua)
            
        Returns:
            Hasil analisis statistik bbox
        """
        self.logger.info(f"📏 Analisis statistik bbox untuk split {split}")
        split_path, images_dir, labels_dir, valid = self._validate_directories(split)
        
        if not valid:
            return {'status': 'error', 'message': f"Direktori tidak lengkap: {split_path}"}
        
        # Dapatkan file gambar valid
        image_files = self._get_valid_files(images_dir, labels_dir, sample_size)
        if not image_files:
            return {'status': 'error', 'message': f"Tidak ada gambar valid ditemukan"}
        
        # Analisis bbox
        widths, heights, areas, aspect_ratios = [], [], [], []
        bbox_by_class, bbox_by_layer = {}, {}
        
        for img_path in image_files:
            label_path = labels_dir / f"{img_path.stem}.txt"
            if not label_path.exists():
                continue
            
            # Baca gambar untuk mendapatkan dimensi aktual (opsional)
            img_shape = None
            try:
                img = self.utils.load_image(img_path)
                img_shape = img.shape  # (h, w, c)
            except Exception:
                pass
            
            # Parse label
            bbox_data = self.utils.parse_yolo_label(label_path)
            
            for box in bbox_data:
                # Format [x_center, y_center, width, height]
                bbox = box['bbox']
                width, height = bbox[2], bbox[3]
                
                # Statistik dasar
                widths.append(width)
                heights.append(height)
                areas.append(width * height)
                aspect_ratios.append(width / height if height > 0 else 0)
                
                # Statistik per kelas
                if 'class_name' in box:
                    class_name = box['class_name']
                    if class_name not in bbox_by_class:
                        bbox_by_class[class_name] = {'widths': [], 'heights': [], 'areas': []}
                    
                    bbox_by_class[class_name]['widths'].append(width)
                    bbox_by_class[class_name]['heights'].append(height)
                    bbox_by_class[class_name]['areas'].append(width * height)
                
                # Statistik per layer
                if 'layer' in box:
                    layer = box['layer']
                    if layer not in bbox_by_layer:
                        bbox_by_layer[layer] = {'widths': [], 'heights': [], 'areas': []}
                    
                    bbox_by_layer[layer]['widths'].append(width)
                    bbox_by_layer[layer]['heights'].append(height)
                    bbox_by_layer[layer]['areas'].append(width * height)
        
        # Jika tidak ada bbox valid
        if not widths:
            return {'status': 'error', 'message': f"Tidak ada bbox valid ditemukan"}
        
        # Kategori ukuran bbox
        area_categories = self._categorize_areas(areas)
        
        # Kompilasi hasil
        result = {
            'status': 'success',
            'total_bbox': len(widths),
            'width': self._calc_stats(widths),
            'height': self._calc_stats(heights),
            'area': self._calc_stats(areas),
            'aspect_ratio': self._calc_stats(aspect_ratios),
            'area_categories': area_categories
        }
        
        # Tambahkan statistik per kelas dan layer jika ada
        if bbox_by_class:
            result['by_class'] = {}
            for cls, stats in bbox_by_class.items():
                result['by_class'][cls] = {
                    'count': len(stats['widths']),
                    'width': self._calc_stats(stats['widths']),
                    'height': self._calc_stats(stats['heights']),
                    'area': self._calc_stats(stats['areas'])
                }
        
        if bbox_by_layer:
            result['by_layer'] = {}
            for layer, stats in bbox_by_layer.items():
                result['by_layer'][layer] = {
                    'count': len(stats['widths']),
                    'width': self._calc_stats(stats['widths']),
                    'height': self._calc_stats(stats['heights']),
                    'area': self._calc_stats(stats['areas'])
                }
        
        # Log hasil
        self.logger.info(
            f"📊 Statistik bbox di {split}:\n"
            f"   • Total bbox: {result['total_bbox']}\n"
            f"   • Width: min={result['width']['min']:.3f}, max={result['width']['max']:.3f}, mean={result['width']['mean']:.3f}\n"
            f"   • Height: min={result['height']['min']:.3f}, max={result['height']['max']:.3f}, mean={result['height']['mean']:.3f}\n"
            f"   • Area: min={result['area']['min']:.3f}, max={result['area']['max']:.3f}, mean={result['area']['mean']:.3f}\n"
            f"   • Aspect ratio: min={result['aspect_ratio']['min']:.3f}, max={result['aspect_ratio']['max']:.3f}, mean={result['aspect_ratio']['mean']:.3f}"
        )
        
        return result
    
    def _calc_stats(self, values: List[float]) -> Dict[str, float]:
        """Hitung statistik dasar untuk list nilai."""
        if not values:
            return {'min': 0, 'max': 0, 'mean': 0, 'median': 0, 'std': 0}
            
        return {
            'min': min(values),
            'max': max(values),
            'mean': np.mean(values),
            'median': np.median(values),
            'std': np.std(values)
        }
    
    def _categorize_areas(self, areas: List[float]) -> Dict[str, Any]:
        """Kategorikan area bbox sebagai kecil, sedang, besar."""
        small = sum(1 for a in areas if a < 0.02)    # < 2%
        medium = sum(1 for a in areas if 0.02 <= a <= 0.1)  # 2-10%
        large = sum(1 for a in areas if a > 0.1)     # > 10%
        
        total = len(areas)
        
        return {
            'small': small,
            'medium': medium,
            'large': large,
            'total': total,
            'small_pct': (small / total) * 100 if total > 0 else 0,
            'medium_pct': (medium / total) * 100 if total > 0 else 0,
            'large_pct': (large / total) * 100 if total > 0 else 0
        }