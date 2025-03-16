"""
File: smartcash/dataset/services/explorer/layer_explorer.py
Deskripsi: Explorer untuk analisis distribusi layer dalam dataset
"""

import collections
from typing import Dict, Any

from smartcash.dataset.services.explorer.base_explorer import BaseExplorer


class LayerExplorer(BaseExplorer):
    """Explorer khusus untuk analisis distribusi layer."""
    
    def analyze_distribution(self, split: str, sample_size: int = 0) -> Dict[str, Any]:
        """
        Analisis distribusi layer dalam dataset.
        
        Args:
            split: Split dataset yang akan dianalisis
            sample_size: Jumlah sampel (0 = semua)
            
        Returns:
            Hasil analisis distribusi layer
        """
        self.logger.info(f"📊 Analisis distribusi layer untuk split {split}")
        split_path, images_dir, labels_dir, valid = self._validate_directories(split)
        
        if not valid:
            return {'status': 'error', 'message': f"Direktori tidak lengkap: {split_path}"}
        
        # Dapatkan file gambar valid
        image_files = self._get_valid_files(images_dir, labels_dir, sample_size)
        if not image_files:
            return {'status': 'error', 'message': f"Tidak ada gambar valid ditemukan"}
        
        # Hitung frekuensi layer
        layer_counts = collections.Counter()
        image_layer_counts = collections.defaultdict(set)  # Layer -> set img_stems
        
        for img_path in image_files:
            label_path = labels_dir / f"{img_path.stem}.txt"
            if not label_path.exists():
                continue
            
            # Parse label dan hitung layer
            available_layers = self.utils.get_available_layers(label_path)
            for layer in available_layers:
                layer_counts[layer] += 1
                image_layer_counts[layer].add(img_path.stem)
        
        # Statistik distribusi
        total_objects = sum(layer_counts.values())
        layer_percentages = {}
        images_per_layer = {}
        
        if total_objects > 0:
            for layer, count in layer_counts.items():
                layer_percentages[layer] = (count / total_objects) * 100
                
            for layer, img_stems in image_layer_counts.items():
                images_per_layer[layer] = len(img_stems)
            
            # Hitung ketidakseimbangan
            if len(layer_counts) > 1:
                counts = list(layer_counts.values())
                max_count, min_count = max(counts), min(counts)
                imbalance_score = min(10.0, (max_count / max(min_count, 1) - 1) / 2)
            else:
                imbalance_score = 0.0
        else:
            layer_percentages = {}
            images_per_layer = {}
            imbalance_score = 0.0
        
        # Log hasil
        self.logger.info(
            f"📊 Distribusi layer di {split}:\n"
            f"   • Total objek: {total_objects}\n"
            f"   • Jumlah layer: {len(layer_counts)}\n"
            f"   • Skor ketidakseimbangan: {imbalance_score:.2f}/10"
        )
        
        for layer, count in layer_counts.items():
            percentage = layer_percentages.get(layer, 0)
            self.logger.info(f"   • {layer}: {count} ({percentage:.1f}%)")
        
        return {
            'status': 'success',
            'total_objects': total_objects,
            'layer_count': len(layer_counts),
            'counts': dict(layer_counts),
            'percentages': layer_percentages,
            'images_per_layer': images_per_layer,
            'imbalance_score': imbalance_score,
            'layers': list(layer_counts.keys())
        }