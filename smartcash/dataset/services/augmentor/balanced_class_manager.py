"""
File: smartcash/dataset/services/augmentor/balanced_class_manager.py
Deskripsi: Manager untuk balancing kelas dengan logger yang diperbaiki (tanpa circular dependency)
"""

from typing import Dict, List, Any, Set, Optional, Tuple
from collections import defaultdict, Counter
import os
from pathlib import Path
from smartcash.common.logger import get_logger

class BalancedClassManager:
    """Manager untuk balancing kelas dengan fokus pada layer detection utama."""
    
    # Definisi layer berdasarkan class index
    LAYER_1_CLASSES = list(range(0, 7))    # Index 0-6: Banknote detection layer
    LAYER_2_CLASSES = list(range(7, 14))   # Index 7-13: Nominal detection layer  
    LAYER_3_CLASSES = list(range(14, 17))  # Index 14-16: Security features (diabaikan)
    
    def __init__(self, logger=None):
        """
        Inisialisasi BalancedClassManager.
        
        Args:
            logger: Logger untuk logging
        """
        self.logger = logger or get_logger(__name__)
        self.target_classes = self.LAYER_1_CLASSES + self.LAYER_2_CLASSES
        
        self.logger.info(f"🎯 Balancing fokus pada {len(self.target_classes)} kelas (Layer 1 & 2)")
    
    def analyze_class_distribution(
        self, 
        image_files: List[str], 
        labels_dir: str
    ) -> Dict[str, Any]:
        """
        Analisis distribusi kelas dengan fokus pada layer 1 dan 2.
        
        Args:
            image_files: List file gambar
            labels_dir: Direktori label
            
        Returns:
            Dictionary hasil analisis
        """
        class_counts = defaultdict(int)
        files_by_class = defaultdict(list)
        layer_stats = {
            'layer_1': defaultdict(int),
            'layer_2': defaultdict(int),
            'layer_3': defaultdict(int),
            'unknown': defaultdict(int)
        }
        
        self.logger.info(f"🔍 Menganalisis {len(image_files)} file untuk distribusi kelas")
        
        for img_path in image_files:
            img_name = Path(img_path).stem
            label_path = os.path.join(labels_dir, f"{img_name}.txt")
            
            if not os.path.exists(label_path):
                continue
            
            try:
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                
                file_classes = set()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        try:
                            class_idx = int(parts[0])
                            file_classes.add(class_idx)
                            class_counts[class_idx] += 1
                            
                            # Kategorisasi per layer
                            if class_idx in self.LAYER_1_CLASSES:
                                layer_stats['layer_1'][class_idx] += 1
                            elif class_idx in self.LAYER_2_CLASSES:
                                layer_stats['layer_2'][class_idx] += 1
                            elif class_idx in self.LAYER_3_CLASSES:
                                layer_stats['layer_3'][class_idx] += 1
                            else:
                                layer_stats['unknown'][class_idx] += 1
                                
                        except ValueError:
                            continue
                
                # Tambahkan file ke kelas-kelas yang relevan (layer 1 & 2 saja)
                for class_idx in file_classes:
                    if class_idx in self.target_classes:
                        files_by_class[class_idx].append(img_path)
                        
            except Exception as e:
                self.logger.warning(f"Error reading label {label_path}: {str(e)}")
                continue
        
        # Hitung statistik layer
        layer_summary = {
            'layer_1_total': sum(layer_stats['layer_1'].values()),
            'layer_2_total': sum(layer_stats['layer_2'].values()),
            'layer_3_total': sum(layer_stats['layer_3'].values()),
            'target_classes_total': sum(class_counts[cls] for cls in self.target_classes)
        }
        
        self.logger.info(f"📊 Layer 1: {layer_summary['layer_1_total']}, Layer 2: {layer_summary['layer_2_total']}, Layer 3: {layer_summary['layer_3_total']} (diabaikan)")
        
        return {
            'class_counts': dict(class_counts),
            'files_by_class': dict(files_by_class),
            'layer_stats': layer_stats,
            'layer_summary': layer_summary,
            'target_classes': self.target_classes
        }
    
    def calculate_balancing_needs(
        self, 
        class_distribution: Dict[str, Any], 
        target_count: int
    ) -> Dict[str, Any]:
        """
        Hitung kebutuhan balancing untuk target classes.
        
        Args:
            class_distribution: Hasil dari analyze_class_distribution
            target_count: Target jumlah instance per kelas
            
        Returns:
            Dictionary kebutuhan balancing
        """
        class_counts = class_distribution['class_counts']
        augmentation_needs = {}
        
        # Hitung kebutuhan hanya untuk target classes (layer 1 & 2)
        for class_idx in self.target_classes:
            current_count = class_counts.get(class_idx, 0)
            needed = max(0, target_count - current_count)
            augmentation_needs[class_idx] = needed
        
        # Statistik kebutuhan
        classes_needing_augmentation = sum(1 for need in augmentation_needs.values() if need > 0)
        total_augmentation_needed = sum(augmentation_needs.values())
        
        self.logger.info(f"🎯 {classes_needing_augmentation}/{len(self.target_classes)} kelas perlu augmentasi ({total_augmentation_needed} total)")
        
        return {
            'augmentation_needs': augmentation_needs,
            'classes_needing_augmentation': classes_needing_augmentation,
            'total_augmentation_needed': total_augmentation_needed,
            'target_count': target_count
        }
    
    def select_files_for_balancing(
        self, 
        class_distribution: Dict[str, Any],
        balancing_needs: Dict[str, Any]
    ) -> List[str]:
        """
        Pilih file untuk augmentasi berdasarkan kebutuhan balancing.
        
        Args:
            class_distribution: Distribusi kelas
            balancing_needs: Kebutuhan balancing
            
        Returns:
            List file yang dipilih untuk augmentasi
        """
        files_by_class = class_distribution['files_by_class']
        augmentation_needs = balancing_needs['augmentation_needs']
        
        selected_files = set()
        
        # Prioritaskan kelas dengan kebutuhan tertinggi
        sorted_needs = sorted(augmentation_needs.items(), key=lambda x: x[1], reverse=True)
        
        for class_idx, needed in sorted_needs:
            if needed <= 0:
                continue
            
            available_files = files_by_class.get(class_idx, [])
            if not available_files:
                self.logger.warning(f"⚠️ Kelas {class_idx}: tidak ada file tersedia")
                continue
            
            # Pilih file untuk kelas ini
            files_to_use = available_files.copy()
            selected_files.update(files_to_use)
            
            self.logger.info(f"✅ Kelas {class_idx}: dipilih {len(files_to_use)} file (butuh {needed} instance)")
        
        result_list = list(selected_files)
        self.logger.info(f"🎯 Total {len(result_list)} file unik dipilih untuk balancing")
        
        return result_list
    
    def prepare_balanced_augmentation(
        self, 
        image_files: List[str], 
        labels_dir: str, 
        target_count: int = 500
    ) -> Dict[str, Any]:
        """
        Persiapkan data untuk augmentasi yang seimbang.
        
        Args:
            image_files: List file gambar
            labels_dir: Direktori label
            target_count: Target jumlah instance per kelas
            
        Returns:
            Dictionary data siap untuk augmentasi
        """
        self.logger.info(f"🔄 Mempersiapkan balancing untuk {len(image_files)} file")
        
        # Analisis distribusi
        class_distribution = self.analyze_class_distribution(image_files, labels_dir)
        
        # Hitung kebutuhan
        balancing_needs = self.calculate_balancing_needs(class_distribution, target_count)
        
        # Pilih file
        selected_files = self.select_files_for_balancing(class_distribution, balancing_needs)
        
        # Jika tidak ada yang perlu dibalance, gunakan sebagian file untuk augmentasi umum
        if not selected_files and image_files:
            selected_files = image_files[:min(100, len(image_files))]
            self.logger.info(f"ℹ️ Tidak ada yang perlu dibalance, menggunakan {len(selected_files)} file untuk augmentasi umum")
        
        return {
            'class_counts': class_distribution['class_counts'],
            'files_by_class': class_distribution['files_by_class'],
            'augmentation_needs': balancing_needs['augmentation_needs'],
            'selected_files': selected_files,
            'layer_summary': class_distribution['layer_summary'],
            'target_classes': self.target_classes,
            'balancing_enabled': len(selected_files) > 0
        }
    
    def get_layer_info(self, class_idx: int) -> str:
        """
        Dapatkan informasi layer untuk class index.
        
        Args:
            class_idx: Index kelas
            
        Returns:
            String informasi layer
        """
        if class_idx in self.LAYER_1_CLASSES:
            return "Layer 1 (Banknote Detection)"
        elif class_idx in self.LAYER_2_CLASSES:
            return "Layer 2 (Nominal Detection)"
        elif class_idx in self.LAYER_3_CLASSES:
            return "Layer 3 (Security Features - Diabaikan)"
        else:
            return "Unknown Layer"

def get_balanced_class_manager(logger=None) -> BalancedClassManager:
    """
    Factory function untuk mendapatkan balanced class manager.
    
    Args:
        logger: Logger
        
    Returns:
        Instance BalancedClassManager
    """
    return BalancedClassManager(logger)