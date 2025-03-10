# File: smartcash/handlers/dataset/explorers/distribution_explorer.py
# Author: Alfrida Sabar
# Deskripsi: Explorer untuk analisis distribusi kelas dan layer dalam dataset

import collections
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
import numpy as np
from tqdm.auto import tqdm

from smartcash.handlers.dataset.explorers.base_explorer import BaseExplorer

class DistributionExplorer(BaseExplorer):
    """Explorer untuk analisis distribusi kelas dan layer dalam dataset."""
    
    def explore(self, split: str, sample_size: int = 0, entity: str = 'class') -> Dict[str, Any]:
        """Analisis distribusi kelas atau layer dalam dataset."""
        if entity not in ('class', 'layer'):
            raise ValueError(f"Entity harus 'class' atau 'layer', bukan '{entity}'")
            
        self.logger.info(f"🔍 Analisis distribusi {entity}: {split}")
        
        # Setup paths dan validasi
        split_dir, images_dir, labels_dir, image_files = self._setup_and_validate(split, sample_size)
        if isinstance(image_files, dict): # Error occurred
            return image_files
        
        # Lakukan analisis
        return self._analyze_class_distribution(labels_dir, image_files) if entity == 'class' else self._analyze_layer_distribution(labels_dir, image_files)
    
    def _setup_and_validate(self, split: str, sample_size: int = 0):
        """Setup path dan validasi dataset."""
        split_dir = self._get_split_path(split)
        images_dir, labels_dir = split_dir / 'images', split_dir / 'labels'
        
        # Validasi direktori
        if not (images_dir.exists() and labels_dir.exists()):
            self.logger.warning(f"⚠️ Split {split} tidak ditemukan atau tidak lengkap")
            return split_dir, images_dir, labels_dir, {'error': f"Split {split} tidak ditemukan atau tidak lengkap"}
        
        # Cari file gambar
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(list(images_dir.glob(ext)))
        
        if not image_files:
            self.logger.warning(f"⚠️ Tidak ada file gambar di split {split}")
            return split_dir, images_dir, labels_dir, {'error': f"Tidak ada file gambar di split {split}"}
        
        # Batasi sampel jika diperlukan
        if 0 < sample_size < len(image_files):
            import random
            image_files = random.sample(image_files, sample_size)
            self.logger.info(f"🔍 Menggunakan {sample_size} sampel untuk analisis")
            
        return split_dir, images_dir, labels_dir, image_files
    
    def analyze_class_distribution(self, split: str, sample_size: int = 0) -> Dict[str, Any]:
        """Analisis distribusi kelas dalam dataset."""
        return self.explore(split, sample_size, 'class')
    
    def analyze_layer_distribution(self, split: str, sample_size: int = 0) -> Dict[str, Any]:
        """Analisis distribusi layer dalam dataset."""
        return self.explore(split, sample_size, 'layer')
    
    def get_class_statistics(self, split: str) -> Dict[str, Any]:
        """Dapatkan statistik kelas untuk split tertentu."""
        self.logger.info(f"📊 Mengumpulkan statistik kelas untuk split: {split}")
        
        # Setup dan validasi
        split_dir, images_dir, labels_dir, result = self._setup_and_validate(split)
        if isinstance(result, dict): # Error occurred
            return result
            
        # Analisis
        return self._collect_entity_statistics(labels_dir, 'class')
    
    def get_layer_statistics(self, split: str) -> Dict[str, Any]:
        """Dapatkan statistik layer untuk split tertentu.""" 
        self.logger.info(f"📊 Mengumpulkan statistik layer untuk split: {split}")
        
        # Setup dan validasi
        split_dir, images_dir, labels_dir, result = self._setup_and_validate(split)
        if isinstance(result, dict): # Error occurred
            return result
            
        # Analisis
        return self._collect_entity_statistics(labels_dir, 'layer')
    
    def _collect_entity_statistics(self, labels_dir: Path, entity_type: str) -> Dict[str, Any]:
        """Mengumpulkan statistik entitas (kelas atau layer)."""
        label_files = list(labels_dir.glob('*.txt'))
        entity_counts = collections.Counter()
        images_per_entity = {}
        is_class = entity_type == 'class'
        
        # Inisialisasi untuk layer
        if not is_class:
            images_per_entity = {layer: set() for layer in self.layer_config_manager.get_layer_names()}
        
        # Proses setiap label
        for label_path in tqdm(label_files, desc=f"Menganalisis {entity_type}"):
            try:
                if not label_path.exists(): continue
                
                entities_in_image = set()
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            try:
                                cls_id = int(float(parts[0]))
                                if is_class:
                                    # Analisis kelas
                                    class_name = self._get_class_name(cls_id)
                                    entity_counts[class_name] += 1
                                    entities_in_image.add(class_name)
                                elif cls_id in self.class_to_layer:
                                    # Analisis layer
                                    layer = self.class_to_layer[cls_id]
                                    entity_counts[layer] += 1
                                    entities_in_image.add(layer)
                            except (ValueError, IndexError):
                                continue
                
                # Update images per entity
                for entity in entities_in_image:
                    if entity not in images_per_entity:
                        images_per_entity[entity] = set()
                    images_per_entity[entity].add(label_path.stem)
            except Exception as e:
                self.logger.debug(f"⚠️ Error membaca {label_path}: {str(e)}")
        
        # Kalkulasi hasil
        total_objects = sum(entity_counts.values())
        entity_percentages = {}
        
        if total_objects > 0:
            for entity, count in entity_counts.items():
                entity_percentages[entity] = (count / total_objects) * 100
        
        # Hitung gambar per entitas
        images_count = {entity: len(images) for entity, images in images_per_entity.items()}
        
        # Return hasil
        return {
            'total_objects': total_objects,
            f'{entity_type}_counts': dict(entity_counts),
            f'{entity_type}_percentages': entity_percentages,
            f'images_per_{entity_type}': images_count
        }
    
    def _analyze_class_distribution(self, labels_dir: Path, image_files: List[Path]) -> Dict[str, Any]:
        """Analisis distribusi kelas dalam dataset."""
        self.logger.info("🔍 Menganalisis distribusi kelas...")
        
        # Hitung jumlah objek per kelas
        class_counts = collections.Counter()
        
        for img_path in tqdm(image_files, desc="Menganalisis kelas"):
            label_path = labels_dir / f"{img_path.stem}.txt"
            if not label_path.exists(): continue
            
            try:
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            try:
                                cls_id = int(float(parts[0]))
                                class_counts[cls_id] += 1
                            except (ValueError, IndexError):
                                continue
            except Exception:
                continue
        
        # Total objek dan persentase
        total_objects = sum(class_counts.values())
        class_percentages = {self._get_class_name(cls_id): (count / total_objects) * 100 
                           for cls_id, count in class_counts.items()} if total_objects > 0 else {}
        
        # Analisis ketidakseimbangan
        if class_counts:
            avg_count = total_objects / len(class_counts)
            
            # Identifikasi kelas
            underrepresented = [self._get_class_name(cls_id) for cls_id, count in class_counts.items() 
                               if count < avg_count * 0.5]
            overrepresented = [self._get_class_name(cls_id) for cls_id, count in class_counts.items() 
                              if count > avg_count * 2]
            
            # Skor ketidakseimbangan
            counts = list(class_counts.values())
            if len(counts) > 1:
                max_count, min_count = max(counts), min(counts)
                imbalance_score = min(10, (max_count / max(min_count, 1) - 1) / 2)
            else:
                imbalance_score = 0
        else:
            underrepresented, overrepresented, imbalance_score = [], [], 0
        
        return {
            'total_objects': total_objects,
            'class_counts': {self._get_class_name(cls_id): count for cls_id, count in class_counts.items()},
            'class_percentages': class_percentages,
            'underrepresented_classes': underrepresented,
            'overrepresented_classes': overrepresented,
            'imbalance_score': imbalance_score
        }
    
    def _analyze_layer_distribution(self, labels_dir: Path, image_files: List[Path]) -> Dict[str, Any]:
        """Analisis distribusi layer dalam dataset."""
        self.logger.info("🔍 Menganalisis distribusi layer...")
        
        # Hitung jumlah objek per layer
        layer_counts = collections.Counter()
        
        for img_path in tqdm(image_files, desc="Menganalisis layer"):
            label_path = labels_dir / f"{img_path.stem}.txt"
            if not label_path.exists(): continue
            
            try:
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            try:
                                cls_id = int(float(parts[0]))
                                if cls_id in self.class_to_layer:
                                    layer_counts[self.class_to_layer[cls_id]] += 1
                            except (ValueError, IndexError):
                                continue
            except Exception:
                continue
        
        # Total objek dan persentase
        total_objects = sum(layer_counts.values())
        layer_percentages = {layer: (count / total_objects) * 100 
                            for layer, count in layer_counts.items()} if total_objects > 0 else {}
        
        # Skor ketidakseimbangan
        imbalance_score = 0
        if layer_counts and len(layer_counts) > 1:
            counts = list(layer_counts.values())
            if len(counts) > 1:
                max_count, min_count = max(counts), min(counts)
                if min_count > 0:
                    imbalance_score = min(10, (max_count / min_count - 1) / 2)
                else:
                    imbalance_score = 10
        
        # Matriks ko-okurens layer jika ada lebih dari satu layer
        layer_cooccurrence = None
        if len(layer_counts) > 1 and len(self.active_layers) > 1:
            layer_cooccurrence = self._analyze_layer_cooccurrence(labels_dir, image_files)
        
        return {
            'total_objects': total_objects,
            'layer_counts': dict(layer_counts),
            'layer_percentages': layer_percentages,
            'imbalance_score': imbalance_score,
            'layer_cooccurrence': layer_cooccurrence
        }
    
    def _analyze_layer_cooccurrence(self, labels_dir: Path, image_files: List[Path]) -> Dict[str, Dict[str, int]]:
        """Analisis ko-okurens layer dalam gambar."""
        self.logger.info("🔍 Menganalisis ko-okurens layer dalam gambar...")
        
        active_layers = set(self.active_layers)
        cooccurrence = {layer1: {layer2: 0 for layer2 in active_layers} for layer1 in active_layers}
        
        for img_path in tqdm(image_files, desc="Menganalisis ko-okurens"):
            label_path = labels_dir / f"{img_path.stem}.txt"
            if not label_path.exists(): continue
                
            try:
                # Temukan layer dalam gambar ini
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
                
                # Hitung ko-okurens
                for layer1 in layers_in_image:
                    for layer2 in layers_in_image:
                        cooccurrence[layer1][layer2] += 1
            except Exception as e:
                self.logger.debug(f"⚠️ Error membaca {label_path}: {str(e)}")
        
        return cooccurrence