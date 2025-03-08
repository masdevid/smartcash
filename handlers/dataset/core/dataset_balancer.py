# File: smartcash/handlers/dataset/core/dataset_balancer.py
# Author: Alfrida Sabar
# Deskripsi: Komponen untuk menyeimbangkan distribusi kelas dalam dataset

import shutil
import random
import collections
from pathlib import Path
from typing import Dict, List, Optional, Any, Counter
from tqdm.auto import tqdm

from smartcash.utils.logger import SmartCashLogger
from smartcash.utils.layer_config_manager import get_layer_config


class DatasetBalancer:
    """
    Komponen untuk menyeimbangkan distribusi kelas dalam dataset.
    """
    
    def __init__(
        self,
        data_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        logger: Optional[SmartCashLogger] = None
    ):
        """
        Inisialisasi DatasetBalancer.
        
        Args:
            data_dir: Direktori dataset
            output_dir: Direktori output (jika None, gunakan data_dir/balanced)
            logger: Logger kustom (opsional)
        """
        self.data_dir = Path(data_dir) if data_dir else Path('data')
        self.output_dir = Path(output_dir) if output_dir else self.data_dir / 'balanced'
        self.logger = logger or SmartCashLogger(__name__)
        
        # Inisialisasi layer config untuk mendapatkan informasi kelas
        self.layer_config = get_layer_config()
        
        self.logger.info(f"⚖️ DatasetBalancer diinisialisasi: {self.data_dir}")
    
    def analyze_class_distribution(
        self, 
        split: str = 'train',
        per_layer: bool = True
    ) -> Dict[str, Any]:
        """
        Analisis distribusi kelas dalam split dataset.
        
        Args:
            split: Split dataset ('train', 'valid', 'test')
            per_layer: Jika True, analisis per layer
            
        Returns:
            Dict berisi statistik distribusi kelas
        """
        split_dir = self.data_dir / split
        images_dir = split_dir / 'images'
        labels_dir = split_dir / 'labels'
        
        if not (images_dir.exists() and labels_dir.exists()):
            self.logger.warning(f"⚠️ Split {split} tidak ditemukan atau tidak lengkap")
            return {}
            
        self.logger.info(f"🔍 Menganalisis distribusi kelas di split '{split}'")
        
        # Inisialisasi counter untuk seluruh dataset
        class_counts: Counter[int] = collections.Counter()
        
        # Inisialisasi counter per layer jika diminta
        layer_class_counts = {}
        if per_layer:
            for layer_name in self.layer_config.get_layer_names():
                layer_class_counts[layer_name] = collections.Counter()
        
        # Baca semua file label
        label_files = list(labels_dir.glob('*.txt'))
        
        # Progress bar
        with tqdm(label_files, desc=f"Analyzing {split}") as pbar:
            for label_path in pbar:
                # Pastikan file gambar yang sesuai ada
                img_stem = label_path.stem
                img_found = False
                
                for ext in ['.jpg', '.jpeg', '.png']:
                    if (images_dir / f"{img_stem}{ext}").exists():
                        img_found = True
                        break
                
                if not img_found:
                    continue
                
                # Baca file label dan hitung kelas
                try:
                    with open(label_path, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if parts and len(parts) >= 5:
                                try:
                                    cls_id = int(float(parts[0]))
                                    # Update counter global
                                    class_counts[cls_id] += 1
                                    
                                    # Update counter per layer jika diminta
                                    if per_layer:
                                        # Cari layer yang memiliki kelas ini
                                        for layer_name in self.layer_config.get_layer_names():
                                            layer_config = self.layer_config.get_layer_config(layer_name)
                                            if cls_id in layer_config['class_ids']:
                                                layer_class_counts[layer_name][cls_id] += 1
                                                break
                                except (ValueError, IndexError):
                                    continue
                except Exception as e:
                    self.logger.debug(f"⚠️ Gagal membaca {label_path}: {str(e)}")
        
        # Hitung statistik distribusi
        total_objects = sum(class_counts.values())
        class_percentages = {}
        
        if total_objects > 0:
            for cls_id, count in class_counts.items():
                class_percentages[cls_id] = (count / total_objects) * 100
                
            # Temukan kelas dominan dan minoritas
            most_common = class_counts.most_common()
            dominant_classes = [cls_id for cls_id, _ in most_common[:3]] if most_common else []
            minority_classes = [cls_id for cls_id, _ in most_common[-3:]] if len(most_common) >= 3 else []
            
            # Hitung ketidakseimbangan
            min_count = most_common[-1][1] if most_common else 0
            max_count = most_common[0][1] if most_common else 0
            
            imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        else:
            dominant_classes = []
            minority_classes = []
            imbalance_ratio = 0
        
        # Format hasil
        result = {
            'total_objects': total_objects,
            'class_counts': dict(class_counts),
            'class_percentages': class_percentages,
            'dominant_classes': dominant_classes,
            'minority_classes': minority_classes,
            'imbalance_ratio': imbalance_ratio,
            'classes_count': len(class_counts)
        }
        
        # Tambahkan statistik per layer jika diminta
        if per_layer:
            layer_stats = {}
            for layer_name, counts in layer_class_counts.items():
                layer_total = sum(counts.values())
                
                if layer_total > 0:
                    layer_most_common = counts.most_common()
                    layer_min_count = layer_most_common[-1][1] if layer_most_common else 0
                    layer_max_count = layer_most_common[0][1] if layer_most_common else 0
                    
                    layer_imbalance = layer_max_count / layer_min_count if layer_min_count > 0 else float('inf')
                    
                    layer_stats[layer_name] = {
                        'total_objects': layer_total,
                        'class_counts': dict(counts),
                        'imbalance_ratio': layer_imbalance
                    }
                else:
                    layer_stats[layer_name] = {
                        'total_objects': 0,
                        'class_counts': {},
                        'imbalance_ratio': 0
                    }
            
            result['layer_stats'] = layer_stats
        
        # Log hasil
        self.logger.info(
            f"📊 Analisis distribusi kelas di '{split}':\n"
            f"   • Total objek: {total_objects}\n"
            f"   • Jumlah kelas: {len(class_counts)}\n"
            f"   • Rasio ketidakseimbangan: {imbalance_ratio:.2f}x"
        )
        
        if dominant_classes:
            # Konversi ID kelas ke nama kelas jika memungkinkan
            dominant_names = []
            for cls_id in dominant_classes:
                cls_name = self._get_class_name(cls_id)
                dominant_names.append(f"{cls_name} ({class_percentages.get(cls_id, 0):.1f}%)")
                
            self.logger.info(f"   • Kelas dominan: {', '.join(dominant_names)}")
            
        if minority_classes:
            # Konversi ID kelas ke nama kelas jika memungkinkan
            minority_names = []
            for cls_id in minority_classes:
                cls_name = self._get_class_name(cls_id)
                minority_names.append(f"{cls_name} ({class_percentages.get(cls_id, 0):.1f}%)")
                
            self.logger.info(f"   • Kelas minoritas: {', '.join(minority_names)}")
        
        return result
    
    def balance_by_undersampling(
        self,
        split: str = 'train',
        max_samples_per_class: Optional[int] = None,
        min_samples_per_class: int = 0,
        target_ratio: float = 1.0,
        random_seed: int = 42
    ) -> Dict[str, Any]:
        """
        Seimbangkan dataset dengan mengurangi jumlah sampel kelas dominan (undersampling).
        
        Args:
            split: Split dataset ('train', 'valid', 'test')
            max_samples_per_class: Jumlah sampel maksimum per kelas (opsional)
            min_samples_per_class: Jumlah sampel minimum per kelas
            target_ratio: Rasio maksimum antara kelas dengan sampel terbanyak dan tersedikit
            random_seed: Seed untuk random state
            
        Returns:
            Dict berisi statistik penyeimbangan
        """
        # Set random seed
        random.seed(random_seed)
        
        # Analisis distribusi awal
        self.logger.info(f"📊 Menganalisis distribusi sebelum penyeimbangan...")
        initial_distribution = self.analyze_class_distribution(split, per_layer=False)
        
        # Buat output directory
        output_split_dir = self.output_dir / split
        output_images_dir = output_split_dir / 'images'
        output_labels_dir = output_split_dir / 'labels'
        
        output_images_dir.mkdir(parents=True, exist_ok=True)
        output_labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Cari semua file label dan kelompokkan berdasarkan kelas
        split_dir = self.data_dir / split
        images_dir = split_dir / 'images'
        labels_dir = split_dir / 'labels'
        
        # Struktur data untuk menyimpan gambar per kelas
        files_by_class: Dict[int, List] = {}
        
        self.logger.info(f"🔍 Mengelompokkan file berdasarkan kelas...")
        
        # Baca semua file label
        label_files = list(labels_dir.glob('*.txt'))
        
        # Progress bar
        with tqdm(label_files, desc="Grouping by class") as pbar:
            for label_path in pbar:
                # Pastikan file gambar yang sesuai ada
                img_stem = label_path.stem
                img_path = None
                
                for ext in ['.jpg', '.jpeg', '.png']:
                    img_candidate = images_dir / f"{img_stem}{ext}"
                    if img_candidate.exists():
                        img_path = img_candidate
                        break
                
                if img_path is None:
                    continue
                
                # Baca file label dan kelompokkan berdasarkan kelas
                try:
                    with open(label_path, 'r') as f:
                        classes_in_file = []
                        
                        for line in f:
                            parts = line.strip().split()
                            if parts and len(parts) >= 5:
                                try:
                                    cls_id = int(float(parts[0]))
                                    classes_in_file.append(cls_id)
                                except (ValueError, IndexError):
                                    continue
                        
                        # Tambahkan file ke setiap kelas yang ditemukan
                        for cls_id in classes_in_file:
                            if cls_id not in files_by_class:
                                files_by_class[cls_id] = []
                            
                            files_by_class[cls_id].append((img_path, label_path))
                except Exception as e:
                    self.logger.debug(f"⚠️ Gagal membaca {label_path}: {str(e)}")
        
        # Tentukan jumlah target per kelas
        class_counts = initial_distribution.get('class_counts', {})
        
        if not class_counts:
            self.logger.warning("⚠️ Tidak ada data kelas yang ditemukan")
            return {}
            
        # Jika max_samples_per_class tidak ditentukan, gunakan kelas dengan sampel paling sedikit
        if max_samples_per_class is None:
            min_count = min(class_counts.values())
            max_samples_per_class = max(min_count, min_samples_per_class)
            
            # Pertimbangkan target_ratio jika ditentukan
            if target_ratio > 1.0:
                # max_count * (1/target_ratio) akan mendekati min_count
                max_count = max(class_counts.values())
                max_samples_per_class = int(max_count / target_ratio)
        
        self.logger.info(f"📊 Target maksimum sampel per kelas: {max_samples_per_class}")
        
        # Pilih sampel per kelas
        selected_files = []
        selected_counts = {}
        
        for cls_id, files in files_by_class.items():
            # Jika jumlah file melebihi max_samples_per_class, lakukan undersampling
            if len(files) > max_samples_per_class:
                # Acak urutan file
                random.shuffle(files)
                
                # Pilih sejumlah file
                selected = files[:max_samples_per_class]
            else:
                # Gunakan semua file
                selected = files
            
            selected_files.extend(selected)
            selected_counts[cls_id] = len(selected)
            
            cls_name = self._get_class_name(cls_id)
            self.logger.info(f"   • Kelas {cls_name}: {len(selected)}/{len(files)} sampel dipilih")
        
        # Salin file yang terpilih ke direktori output
        self.logger.info(f"📦 Menyalin {len(selected_files)} file ke {output_split_dir}...")
        
        # Set untuk melacak file yang sudah disalin
        copied_files = set()
        
        # Progress bar
        with tqdm(selected_files, desc="Copying selected files") as pbar:
            for img_path, label_path in pbar:
                # Skip jika file gambar sudah disalin (untuk menghindari duplikasi)
                if img_path.name in copied_files:
                    continue
                    
                # Salin file
                dest_img = output_images_dir / img_path.name
                dest_label = output_labels_dir / label_path.name
                
                if not dest_img.exists():
                    shutil.copy2(img_path, dest_img)
                    
                if not dest_label.exists():
                    shutil.copy2(label_path, dest_label)
                    
                copied_files.add(img_path.name)
        
        # Analisis distribusi akhir
        self.logger.info(f"📊 Menganalisis distribusi setelah penyeimbangan...")
        
        # Ubah data_dir sementara untuk analisis
        original_data_dir = self.data_dir
        self.data_dir = self.output_dir
        final_distribution = self.analyze_class_distribution(split, per_layer=False)
        self.data_dir = original_data_dir  # Kembalikan data_dir asli
        
        # Hitung penurunan rasio ketidakseimbangan
        initial_ratio = initial_distribution.get('imbalance_ratio', 0)
        final_ratio = final_distribution.get('imbalance_ratio', 0)
        
        improvement = 0
        if initial_ratio > 0:
            improvement = ((initial_ratio - final_ratio) / initial_ratio) * 100
        
        self.logger.success(
            f"✅ Penyeimbangan dengan undersampling selesai:\n"
            f"   • Rasio ketidakseimbangan awal: {initial_ratio:.2f}x\n"
            f"   • Rasio ketidakseimbangan akhir: {final_ratio:.2f}x\n"
            f"   • Perbaikan: {improvement:.1f}%\n"
            f"   • Total sampel: {len(copied_files)}"
        )
        
        # Return statistik
        return {
            'initial_distribution': initial_distribution,
            'final_distribution': final_distribution,
            'improvement': improvement,
            'samples_per_class': selected_counts,
            'total_samples': len(copied_files),
            'directory': str(output_split_dir)
        }
    
    def _get_class_name(self, cls_id: int) -> str:
        """
        Dapatkan nama kelas berdasarkan ID kelas.
        
        Args:
            cls_id: ID kelas
            
        Returns:
            Nama kelas atau string ID jika tidak ditemukan
        """
        for layer_name in self.layer_config.get_layer_names():
            layer_config = self.layer_config.get_layer_config(layer_name)
            class_ids = layer_config['class_ids']
            classes = layer_config['classes']
            
            if cls_id in class_ids:
                idx = class_ids.index(cls_id)
                if idx < len(classes):
                    return classes[idx]
        
        return f"Class-{cls_id}"