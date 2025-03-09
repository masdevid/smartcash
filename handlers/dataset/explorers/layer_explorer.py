# File: smartcash/handlers/dataset/explorers/layer_explorer.py
# Author: Alfrida Sabar
# Deskripsi: Explorer khusus untuk distribusi layer dalam dataset

import collections
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
import numpy as np
from tqdm.auto import tqdm

from smartcash.handlers.dataset.explorers.base_explorer import BaseExplorer

class LayerExplorer(BaseExplorer):
    """
    Explorer khusus untuk distribusi layer dalam dataset.
    Menganalisis keseimbangan, representasi, dan statistik layer.
    """
    
    def explore(self, split: str, sample_size: int = 0) -> Dict[str, Any]:
        """
        Analisis distribusi layer dalam dataset.
        
        Args:
            split: Split dataset ('train', 'valid', 'test')
            sample_size: Jumlah sampel (0 = semua)
            
        Returns:
            Dict hasil analisis layer
        """
        self.logger.info(f"🔍 Analisis distribusi layer: {split}")
        
        # Tentukan path split
        split_dir = self._get_split_path(split)
        images_dir = split_dir / 'images'
        labels_dir = split_dir / 'labels'
        
        if not (images_dir.exists() and labels_dir.exists()):
            self.logger.warning(f"⚠️ Split {split} tidak ditemukan atau tidak lengkap")
            return {'error': f"Split {split} tidak ditemukan atau tidak lengkap"}
        
        # Cari semua file gambar
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(list(images_dir.glob(ext)))
        
        if not image_files:
            self.logger.warning(f"⚠️ Tidak ada file gambar di split {split}")
            return {'error': f"Tidak ada file gambar di split {split}"}
        
        # Batasi sampel jika diperlukan
        if 0 < sample_size < len(image_files):
            import random
            image_files = random.sample(image_files, sample_size)
            self.logger.info(f"🔍 Menggunakan {sample_size} sampel untuk analisis layer")
        
        # Analisis distribusi layer
        layer_balance = self._analyze_layer_distribution(labels_dir, image_files)
        
        # Log hasil analisis
        total_objects = layer_balance.get('total_objects', 0)
        layer_counts = layer_balance.get('layer_counts', {})
        layer_percentages = layer_balance.get('layer_percentages', {})
        imbalance_score = layer_balance.get('imbalance_score', 0)
        
        self.logger.info(
            f"📊 Hasil analisis layer '{split}':\n"
            f"   • Total objek: {total_objects}\n"
            f"   • Jumlah layer terdeteksi: {len(layer_counts)}\n"
            f"   • Skor ketidakseimbangan: {imbalance_score:.2f}/10"
        )
        
        # Log layer counts & percentages
        for layer, count in layer_counts.items():
            percentage = layer_percentages.get(layer, 0)
            self.logger.info(f"   • Layer {layer}: {count} objek ({percentage:.1f}%)")
        
        # Buat matriks ko-okurens layer
        if len(layer_counts) > 1:
            layer_cooccurrence = self._analyze_layer_cooccurrence(labels_dir, image_files)
            layer_balance['layer_cooccurrence'] = layer_cooccurrence
        
        return layer_balance
    
    def get_layer_statistics(self, split: str) -> Dict[str, Any]:
        """
        Dapatkan statistik layer untuk split tertentu.
        
        Args:
            split: Split dataset ('train', 'valid', 'test')
            
        Returns:
            Dict statistik layer
        """
        self.logger.info(f"📊 Mengumpulkan statistik layer untuk split: {split}")
        
        # Tentukan path split
        split_dir = self._get_split_path(split)
        images_dir = split_dir / 'images'
        labels_dir = split_dir / 'labels'
        
        if not (images_dir.exists() and labels_dir.exists()):
            self.logger.warning(f"⚠️ Split {split} tidak ditemukan atau tidak lengkap")
            return {'error': f"Split {split} tidak ditemukan atau tidak lengkap"}
        
        # Cari semua file label
        label_files = list(labels_dir.glob('*.txt'))
        
        # Hitung jumlah objek per layer
        layer_counts = collections.Counter()
        images_per_layer = {layer: set() for layer in self.layer_config_manager.get_layer_names()}
        
        for label_path in tqdm(label_files, desc=f"Menganalisis layer {split}"):
            try:
                if not label_path.exists():
                    continue
                    
                # Kumpulkan layer yang ada dalam gambar ini
                layers_in_image = set()
                
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            try:
                                cls_id = int(float(parts[0]))
                                if cls_id in self.class_to_layer:
                                    layer = self.class_to_layer[cls_id]
                                    layer_counts[layer] += 1
                                    layers_in_image.add(layer)
                            except (ValueError, IndexError):
                                continue
                
                # Update set gambar per layer
                for layer in layers_in_image:
                    images_per_layer[layer].add(label_path.stem)
                    
            except Exception as e:
                self.logger.debug(f"⚠️ Error membaca {label_path}: {str(e)}")
        
        # Total objek dan persentase
        total_objects = sum(layer_counts.values())
        layer_percentages = {}
        
        if total_objects > 0:
            for layer, count in layer_counts.items():
                layer_percentages[layer] = (count / total_objects) * 100
        
        # Statistik jumlah gambar per layer
        images_count = {layer: len(images) for layer, images in images_per_layer.items()}
        
        return {
            'total_objects': total_objects,
            'layer_counts': dict(layer_counts),
            'layer_percentages': layer_percentages,
            'images_per_layer': images_count
        }
    
    def _analyze_layer_distribution(
        self,
        labels_dir: Path,
        image_files: List[Path]
    ) -> Dict[str, Any]:
        """
        Analisis distribusi layer dalam dataset.
        
        Args:
            labels_dir: Direktori label
            image_files: List file gambar
            
        Returns:
            Dict berisi statistik distribusi layer
        """
        self.logger.info("🔍 Menganalisis distribusi layer...")
        
        # Hitung jumlah objek per layer
        layer_counts = collections.Counter()
        
        for img_path in tqdm(image_files, desc="Menganalisis layer"):
            label_path = labels_dir / f"{img_path.stem}.txt"
            if not label_path.exists():
                continue
            
            try:
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            try:
                                cls_id = int(float(parts[0]))
                                if cls_id in self.class_to_layer:
                                    layer = self.class_to_layer[cls_id]
                                    layer_counts[layer] += 1
                            except (ValueError, IndexError):
                                continue
            except Exception:
                # Skip file jika ada error
                continue
        
        # Total objek
        total_objects = sum(layer_counts.values())
        
        # Hitung persentase per layer
        layer_percentages = {}
        if total_objects > 0:
            for layer, count in layer_counts.items():
                layer_percentages[layer] = (count / total_objects) * 100
        
        # Identifikasi layer yang tidak seimbang
        if layer_counts and len(layer_counts) > 1:
            avg_count = total_objects / len(layer_counts)
            
            # Hitung skor ketidakseimbangan (0-10)
            counts = list(layer_counts.values())
            if len(counts) > 1:
                max_count = max(counts)
                min_count = min(counts)
                
                if min_count > 0:
                    imbalance_ratio = max_count / min_count
                    imbalance_score = min(10, (imbalance_ratio - 1) / 2)
                else:
                    imbalance_score = 10
            else:
                imbalance_score = 0
                
        else:
            imbalance_score = 0
        
        return {
            'total_objects': total_objects,
            'layer_counts': dict(layer_counts),
            'layer_percentages': layer_percentages,
            'imbalance_score': imbalance_score
        }
    
    def _analyze_layer_cooccurrence(
        self,
        labels_dir: Path,
        image_files: List[Path]
    ) -> Dict[str, Dict[str, int]]:
        """
        Analisis ko-okurens layer dalam gambar.
        
        Args:
            labels_dir: Direktori label
            image_files: List file gambar
            
        Returns:
            Dict berisi matriks ko-okurens layer
        """
        self.logger.info("🔍 Menganalisis ko-okurens layer dalam gambar...")
        
        # Definisikan layer aktif
        active_layers = set(self.active_layers)
        
        # Matriks ko-okurens
        cooccurrence = {layer1: {layer2: 0 for layer2 in active_layers} for layer1 in active_layers}
        
        # Hitung gambar dengan setiap kombinasi layer
        for img_path in tqdm(image_files, desc="Menganalisis ko-okurens"):
            label_path = labels_dir / f"{img_path.stem}.txt"
            if not label_path.exists():
                continue
                
            try:
                # Tentukan layer yang ada dalam gambar ini
                layers_in_image = set()
                
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            try:
                                cls_id = int(float(parts[0]))
                                if cls_id in self.class_to_layer:
                                    layer = self.class_to_layer[cls_id]
                                    if layer in active_layers:
                                        layers_in_image.add(layer)
                            except (ValueError, IndexError):
                                continue
                
                # Hitung ko-okurens untuk setiap pasangan layer
                for layer1 in layers_in_image:
                    for layer2 in layers_in_image:
                        cooccurrence[layer1][layer2] += 1
                        
            except Exception as e:
                self.logger.debug(f"⚠️ Error membaca {label_path}: {str(e)}")
        
        return cooccurrence