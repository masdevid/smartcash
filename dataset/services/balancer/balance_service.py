"""
File: smartcash/dataset/services/balancer/balance_service.py
Deskripsi: Layanan untuk menyeimbangkan dataset dengan teknik undersampling dan oversampling
"""

import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor
from collections import Counter

from smartcash.common.logger import get_logger
from smartcash.dataset.utils.dataset_utils import DatasetUtils


class BalanceService:
    """Service untuk menyeimbangkan dataset untuk mengatasi ketidakseimbangan kelas."""
    
    def __init__(self, config: Dict, data_dir: str, logger=None, num_workers: int = 4):
        """
        Inisialisasi BalanceService.
        
        Args:
            config: Konfigurasi aplikasi
            data_dir: Direktori data
            logger: Logger kustom (opsional)
            num_workers: Jumlah worker untuk proses paralel
        """
        self.config = config
        self.data_dir = Path(data_dir)
        self.logger = logger or get_logger()
        self.num_workers = num_workers
        
        # Setup utils
        self.utils = DatasetUtils(config, data_dir, logger)
        
        self.logger.info(f"⚖️ BalanceService diinisialisasi dengan {num_workers} workers")
    
    def balance_by_undersampling(
        self,
        split: str = 'train',
        strategy: str = 'random',
        target_count: Optional[int] = None,
        target_ratio: Optional[float] = None,
        exclude_classes: List[str] = None,
        output_dir: Optional[str] = None,
        random_seed: int = 42
    ) -> Dict[str, Any]:
        """
        Seimbangkan dataset dengan undersampling.
        
        Args:
            split: Split dataset yang akan diseimbangkan
            strategy: Strategi undersampling ('random', 'deterministic')
            target_count: Jumlah sampel target per kelas (opsional)
            target_ratio: Rasio antara kelas minoritas dan mayoritas (opsional)
            exclude_classes: Kelas yang dikecualikan dari balancing
            output_dir: Direktori output (opsional)
            random_seed: Seed untuk random
            
        Returns:
            Hasil balancing
        """
        # Set random seed
        random.seed(random_seed)
        
        # Analisis distribusi kelas
        split_path = self.utils.get_split_path(split)
        images_dir, labels_dir = split_path / 'images', split_path / 'labels'
        
        if not (images_dir.exists() and labels_dir.exists()):
            self.logger.error(f"❌ Direktori dataset tidak lengkap: {split_path}")
            return {'status': 'error', 'message': f"Direktori dataset tidak lengkap"}
        
        # Setup direktori output
        if output_dir:
            output_path = Path(output_dir)
        else:
            output_path = self.data_dir / f"{split}_balanced"
            
        output_images_dir = output_path / 'images'
        output_labels_dir = output_path / 'labels'
        
        # Buat direktori output
        output_images_dir.mkdir(parents=True, exist_ok=True)
        output_labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Analisis distribusi kelas
        self.logger.info(f"🔍 Menganalisis distribusi kelas di {split}...")
        class_distribution, file_by_class = self._analyze_class_distribution(images_dir, labels_dir)
        
        if not class_distribution:
            self.logger.warning(f"⚠️ Tidak ada kelas ditemukan di {split}")
            return {'status': 'warning', 'message': 'Tidak ada kelas ditemukan'}
        
        # Log distribusi kelas
        self.logger.info(f"📊 Distribusi kelas di {split}:")
        sorted_classes = sorted(class_distribution.items(), key=lambda x: x[1], reverse=True)
        for cls, count in sorted_classes:
            self.logger.info(f"   • {cls}: {count} sampel")
            
        # Kelas minoritas dan mayoritas
        majority_cls, majority_count = sorted_classes[0]
        minority_cls, minority_count = sorted_classes[-1]
        
        # Hitung target count untuk undersampling
        if target_count is not None:
            # Gunakan target count yang eksplisit
            target = target_count
        elif target_ratio is not None:
            # Gunakan target ratio
            target = int(minority_count / target_ratio)
        else:
            # Gunakan minority count sebagai target default
            target = minority_count
            
        self.logger.info(f"⚖️ Target balancing: {target} sampel per kelas")
        
        # Proses undersampling
        selected_files = {}
        for cls, files in file_by_class.items():
            if exclude_classes and cls in exclude_classes:
                # Jika kelas dikecualikan, gunakan semua file
                selected_files[cls] = files
                self.logger.info(f"   • {cls} dikecualikan dari balancing, menggunakan semua {len(files)} file")
                continue
                
            current_count = len(files)
            
            if current_count <= target:
                # Jika jumlah file sudah <= target, gunakan semua
                selected_files[cls] = files
                self.logger.info(f"   • {cls}: {current_count} ≤ {target}, tidak perlu undersampling")
            else:
                # Lakukan undersampling
                if strategy == 'random':
                    # Random undersampling
                    selected = random.sample(files, target)
                elif strategy == 'deterministic':
                    # Deterministik (ambil pertama), berguna untuk reproduksibilitas
                    selected = files[:target]
                else:
                    # Default ke random
                    selected = random.sample(files, target)
                    
                selected_files[cls] = selected
                self.logger.info(f"   • {cls}: {current_count} → {len(selected)} dengan {strategy} undersampling")
        
        # Salin file terpilih ke direktori output
        total_copied = 0
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            
            for cls, files in selected_files.items():
                for img_path, label_path in files:
                    futures.append(executor.submit(
                        self._copy_file_pair,
                        img_path,
                        label_path,
                        output_images_dir,
                        output_labels_dir
                    ))
            
            # Process results
            with tqdm(total=len(futures), desc="📋 Copying balanced files") as pbar:
                for future in futures:
                    if future.result():
                        total_copied += 1
                    pbar.update(1)
        
        # Hasil
        result = {
            'status': 'success',
            'split': split,
            'strategy': strategy,
            'target_count': target,
            'original_distribution': dict(class_distribution),
            'balanced_distribution': {cls: len(files) for cls, files in selected_files.items()},
            'total_copied': total_copied,
            'output_dir': str(output_path)
        }
        
        self.logger.success(
            f"✅ Balancing dataset selesai:\n"
            f"   • Split: {split}\n"
            f"   • Strategi: {strategy}\n"
            f"   • Target count: {target}\n"
            f"   • Total file: {total_copied}\n"
            f"   • Output: {output_path}"
        )
        
        return result
    
    def balance_by_oversampling(
        self,
        split: str = 'train',
        target_count: Optional[int] = None,
        class_list: List[str] = None,
        augmentation_types: List[str] = None,
        output_dir: Optional[str] = None,
        random_seed: int = 42
    ) -> Dict[str, Any]:
        """
        Seimbangkan dataset dengan oversampling (menggunakan augmentasi).
        
        Args:
            split: Split dataset yang akan diseimbangkan
            target_count: Jumlah sampel target per kelas (opsional)
            class_list: Daftar kelas yang akan di-oversample (opsional)
            augmentation_types: Jenis augmentasi untuk oversampling
            output_dir: Direktori output (opsional)
            random_seed: Seed untuk random
            
        Returns:
            Hasil balancing
        """
        # Ini menggunakan AugmentationService dengan target_balance=True
        # untuk menyeimbangkan dataset dengan oversampling
        from smartcash.dataset.services.augmentor.augmentation_service import AugmentationService
        
        augmentation_service = AugmentationService(
            self.config, str(self.data_dir), self.logger, self.num_workers
        )
        
        # Default augmentation types jika tidak disediakan
        if not augmentation_types:
            augmentation_types = ['flip', 'rotate', 'brightness', 'contrast']
            
        # Jalankan augmentasi dengan target balancing
        result = augmentation_service.augment_dataset(
            split=split,
            augmentation_types=augmentation_types,
            target_count=target_count,
            target_balance=True,
            class_list=class_list,
            output_dir=output_dir,
            random_seed=random_seed
        )
        
        # Update status
        if result['status'] == 'success':
            result['balancing_method'] = 'oversampling'
            self.logger.success(f"✅ Balancing dengan oversampling selesai")
            
        return result
    
    def calculate_weights(self, split: str = 'train') -> Dict[str, float]:
        """
        Hitung bobot sampling untuk weighted dataset.
        
        Args:
            split: Split dataset
            
        Returns:
            Dictionary berisi bobot per kelas
        """
        # Analisis distribusi kelas
        split_path = self.utils.get_split_path(split)
        images_dir, labels_dir = split_path / 'images', split_path / 'labels'
        
        if not (images_dir.exists() and labels_dir.exists()):
            self.logger.error(f"❌ Direktori dataset tidak lengkap: {split_path}")
            return {}
            
        # Analisis distribusi kelas
        class_distribution, _ = self._analyze_class_distribution(images_dir, labels_dir)
        
        if not class_distribution:
            self.logger.warning(f"⚠️ Tidak ada kelas ditemukan di {split}")
            return {}
            
        # Hitung bobot berdasarkan frekuensi invers
        total_samples = sum(class_distribution.values())
        weights = {}
        
        for cls, count in class_distribution.items():
            # Frekuensi invers: semakin sedikit sampel, semakin tinggi bobot
            weights[cls] = total_samples / (len(class_distribution) * count)
            
        # Normalisasi bobot
        max_weight = max(weights.values())
        for cls in weights:
            weights[cls] /= max_weight
            
        self.logger.info(f"⚖️ Bobot sampling per kelas:")
        for cls, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
            self.logger.info(f"   • {cls}: {weight:.4f}")
            
        return weights
    
    def _analyze_class_distribution(
        self,
        images_dir: Path,
        labels_dir: Path
    ) -> Tuple[Dict[str, int], Dict[str, List[Tuple[Path, Path]]]]:
        """
        Analisis distribusi kelas dalam dataset.
        
        Args:
            images_dir: Direktori gambar
            labels_dir: Direktori label
            
        Returns:
            Tuple (distribusi kelas, file per kelas)
        """
        # Cari semua file gambar
        image_files = self.utils.find_image_files(images_dir)
        if not image_files:
            return {}, {}
            
        # Analisis distribusi kelas
        class_counts = Counter()
        file_by_class = {}
        
        for img_path in image_files:
            label_path = labels_dir / f"{img_path.stem}.txt"
            if not label_path.exists():
                continue
                
            # Parse label
            bbox_data = self.utils.parse_yolo_label(label_path)
            img_classes = set()
            
            for box in bbox_data:
                if 'class_name' in box:
                    class_name = box['class_name']
                    img_classes.add(class_name)
                    
                    if class_name not in file_by_class:
                        file_by_class[class_name] = []
                        
                    if (img_path, label_path) not in file_by_class[class_name]:
                        file_by_class[class_name].append((img_path, label_path))
            
            # Update counts - satu gambar dihitung satu kali per kelas
            for cls in img_classes:
                class_counts[cls] += 1
                
        return dict(class_counts), file_by_class
    
    def _copy_file_pair(
        self,
        img_path: Path,
        label_path: Path,
        output_images_dir: Path,
        output_labels_dir: Path
    ) -> bool:
        """
        Salin pasangan file gambar dan label ke direktori output.
        
        Args:
            img_path: Path ke file gambar
            label_path: Path ke file label
            output_images_dir: Direktori output untuk gambar
            output_labels_dir: Direktori output untuk label
            
        Returns:
            Sukses atau tidak
        """
        try:
            # Salin gambar
            shutil.copy2(img_path, output_images_dir / img_path.name)
            
            # Salin label
            shutil.copy2(label_path, output_labels_dir / label_path.name)
            
            return True
        except Exception as e:
            self.logger.warning(f"⚠️ Gagal menyalin file {img_path.name}: {str(e)}")
            return False