"""
File: smartcash/dataset/services/augmentor/augmentation_service.py
Deskripsi: Layanan untuk melakukan augmentasi dataset guna memperkaya variasi data
"""

import os
import time
import random
import shutil
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor

from smartcash.common.logger import get_logger
from smartcash.dataset.utils.dataset_utils import DatasetUtils


class AugmentationService:
    """Service untuk augmentasi dataset untuk meningkatkan variasi data."""
    
    def __init__(self, config: Dict, data_dir: str, logger=None, num_workers: int = 4):
        """
        Inisialisasi AugmentationService.
        
        Args:
            config: Konfigurasi aplikasi
            data_dir: Direktori data
            logger: Logger kustom (opsional)
            num_workers: Jumlah worker untuk proses paralel
        """
        self.config = config
        self.data_dir = Path(data_dir)
        self.logger = logger or get_logger("augmentation_service")
        self.num_workers = num_workers
        
        # Setup utils
        self.utils = DatasetUtils(config, data_dir, logger)
        
        self.logger.info(f"🔄 AugmentationService diinisialisasi dengan {num_workers} workers")
    
    def augment_dataset(
        self,
        split: str = 'train',
        augmentation_types: List[str] = None,
        target_count: int = None, 
        target_factor: float = None,
        target_balance: bool = False,
        class_list: List[str] = None,
        output_dir: Optional[str] = None,
        random_seed: int = 42
    ) -> Dict[str, Any]:
        """
        Augmentasi dataset untuk satu split.
        
        Args:
            split: Split dataset yang akan diaugmentasi
            augmentation_types: Jenis augmentasi yang akan diterapkan
            target_count: Target jumlah sampel per kelas (opsional)
            target_factor: Faktor pengali jumlah sampel (opsional)
            target_balance: Apakah menyeimbangkan jumlah sampel antar kelas
            class_list: Daftar kelas yang akan diaugmentasi (opsional)
            output_dir: Direktori output (opsional)
            random_seed: Seed untuk random
            
        Returns:
            Hasil augmentasi
        """
        start_time = time.time()
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Default augmentation types jika tidak disediakan
        if not augmentation_types:
            augmentation_types = ['flip', 'rotate', 'brightness', 'contrast']
            
        # Dapatkan pipeline augmentasi
        pipeline = self._get_augmentation_pipeline(augmentation_types)
        
        # Direktori dataset
        split_path = self.utils.get_split_path(split)
        images_dir, labels_dir = split_path / 'images', split_path / 'labels'
        
        # Cek direktori
        if not (images_dir.exists() and labels_dir.exists()):
            self.logger.error(f"❌ Direktori dataset tidak lengkap: {split_path}")
            return {'status': 'error', 'message': f"Direktori dataset tidak lengkap"}
        
        # Setup direktori output
        if output_dir:
            output_path = Path(output_dir)
        else:
            output_path = self.data_dir / f"{split}_augmented"
            
        output_images_dir = output_path / 'images'
        output_labels_dir = output_path / 'labels'
        
        # Buat direktori output
        output_images_dir.mkdir(parents=True, exist_ok=True)
        output_labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Analisis distribusi kelas untuk target balancing
        class_distribution = {}
        if target_balance or class_list:
            from smartcash.dataset.services.explorer.class_explorer import ClassExplorer
            explorer = ClassExplorer(self.config, str(self.data_dir), self.logger, self.num_workers)
            result = explorer.analyze_distribution(split)
            
            if result['status'] == 'success':
                class_distribution = result['counts']
                
                # Filter kelas yang akan diaugmentasi jika disediakan
                if class_list:
                    class_distribution = {cls: count for cls, count in class_distribution.items() 
                                       if cls in class_list}
        
        # Dapatkan semua file gambar valid
        image_files = self.utils.find_image_files(images_dir)
        if not image_files:
            self.logger.warning(f"⚠️ Tidak ada gambar ditemukan di {images_dir}")
            return {'status': 'warning', 'message': f"Tidak ada gambar ditemukan"}
        
        # Buat mapping file berdasarkan kelas
        class_to_files = {}
        for img_path in image_files:
            label_path = labels_dir / f"{img_path.stem}.txt"
            if not label_path.exists():
                continue
                
            # Parse label untuk menentukan kelas
            bbox_data = self.utils.parse_yolo_label(label_path)
            for box in bbox_data:
                if 'class_name' in box:
                    class_name = box['class_name']
                    if class_name not in class_to_files:
                        class_to_files[class_name] = []
                    if img_path not in class_to_files[class_name]:
                        class_to_files[class_name].append((img_path, label_path))
        
        # Hitung target jumlah per kelas
        targets = self._compute_augmentation_targets(
            class_distribution, class_to_files, target_count, target_factor, target_balance
        )
        
        # Log rencana augmentasi
        self.logger.info(f"🔍 Augmentasi dataset {split} dengan {len(augmentation_types)} jenis transformasi")
        for cls, target in targets.items():
            current = len(class_to_files.get(cls, []))
            to_generate = max(0, target - current)
            if to_generate > 0:
                self.logger.info(f"   • {cls}: {current} → {target} (+{to_generate})")
        
        # Lakukan augmentasi untuk setiap kelas
        stats = {'original': 0, 'generated': 0, 'source_classes': len(class_to_files)}
        
        # Copy file original ke output dir terlebih dahulu
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            copy_tasks = []
            for files in class_to_files.values():
                for img_path, label_path in files:
                    copy_tasks.append(executor.submit(
                        self._copy_file_pair, 
                        img_path, 
                        label_path, 
                        output_images_dir, 
                        output_labels_dir
                    ))
            
            # Process results
            for future in tqdm(copy_tasks, desc="📋 Copying original files"):
                if future.result():
                    stats['original'] += 1
            
        # Proses augmentasi per kelas
        with tqdm(total=sum(max(0, targets[cls] - len(class_to_files.get(cls, []))) 
                         for cls in targets), 
                 desc="🔄 Augmentasi dataset") as pbar:
            
            for cls, target in targets.items():
                if cls not in class_to_files or not class_to_files[cls]:
                    continue
                    
                current = len(class_to_files[cls])
                to_generate = max(0, target - current)
                
                if to_generate <= 0:
                    continue
                
                # Generate augmentasi untuk kelas ini
                generated = self._augment_class(
                    class_to_files[cls],
                    pipeline,
                    to_generate,
                    output_images_dir,
                    output_labels_dir,
                    pbar
                )
                
                stats['generated'] += generated
        
        # Rekap hasil
        elapsed_time = time.time() - start_time
        stats.update({
            'status': 'success',
            'augmentation_types': augmentation_types,
            'split': split,
            'output_dir': str(output_path),
            'duration': elapsed_time,
            'total_files': stats['original'] + stats['generated']
        })
        
        self.logger.success(
            f"✅ Augmentasi dataset selesai ({elapsed_time:.1f}s):\n"
            f"   • File asli: {stats['original']}\n"
            f"   • File baru: {stats['generated']}\n"
            f"   • Total: {stats['total_files']}\n"
            f"   • Output: {output_path}"
        )
        
        return stats
    
    def _get_augmentation_pipeline(self, augmentation_types: List[str]) -> Any:
        """
        Dapatkan pipeline augmentasi berdasarkan jenis yang dipilih.
        
        Args:
            augmentation_types: Jenis augmentasi yang akan diterapkan
            
        Returns:
            Pipeline augmentasi
        """
        from smartcash.dataset.utils.transform.image_transform import ImageTransformer
        
        # Ambil parameter augmentasi
        aug_config = self.config.get('augmentation', {})
        position_params = aug_config.get('position', {})
        lighting_params = aug_config.get('lighting', {})
        
        # Override parameter berdasarkan jenis augmentasi yang dipilih
        params = {}
        
        if 'flip' in augmentation_types:
            params['fliplr'] = position_params.get('fliplr', 0.5)
        else:
            params['fliplr'] = 0.0
            
        if 'rotate' in augmentation_types:
            params['degrees'] = position_params.get('degrees', 15)
            params['translate'] = position_params.get('translate', 0.1)
            params['scale'] = position_params.get('scale', 0.1)
        else:
            params['degrees'] = 0
            params['translate'] = 0
            params['scale'] = 0
            
        if 'brightness' in augmentation_types:
            params['brightness'] = lighting_params.get('brightness', 0.3)
        else:
            params['brightness'] = 0
            
        if 'contrast' in augmentation_types:
            params['contrast'] = lighting_params.get('contrast', 0.3)
        else:
            params['contrast'] = 0
            
        if 'hsv' in augmentation_types:
            params['hsv_h'] = lighting_params.get('hsv_h', 0.015)
            params['hsv_s'] = lighting_params.get('hsv_s', 0.7)
            params['hsv_v'] = lighting_params.get('hsv_v', 0.4)
        else:
            params['hsv_h'] = 0
            params['hsv_s'] = 0
            params['hsv_v'] = 0
            
        # Buat transformer dengan parameter yang disesuaikan
        transformer = ImageTransformer(self.config, logger=self.logger)
        pipeline = transformer.create_custom_transform(**params)
        
        return pipeline
    
    def _compute_augmentation_targets(
        self,
        class_distribution: Dict[str, int],
        class_to_files: Dict[str, List[Tuple[Path, Path]]],
        target_count: Optional[int],
        target_factor: Optional[float],
        target_balance: bool
    ) -> Dict[str, int]:
        """
        Hitung jumlah target augmentasi untuk setiap kelas.
        
        Args:
            class_distribution: Distribusi kelas dalam dataset
            class_to_files: Mapping kelas ke file gambar
            target_count: Target jumlah sampel per kelas
            target_factor: Faktor pengali jumlah sampel
            target_balance: Apakah menyeimbangkan jumlah sampel antar kelas
            
        Returns:
            Target jumlah sampel per kelas
        """
        targets = {}
        
        # Jika tidak ada parameter, gunakan target_factor=2.0 sebagai default
        if not target_count and not target_factor and not target_balance:
            target_factor = 2.0
            
        # Jika target balancing, gunakan kelas dengan jumlah sampel terbanyak
        if target_balance and class_distribution:
            max_count = max(class_distribution.values())
            for cls in class_distribution:
                targets[cls] = max_count
        
        # Jika ada target count, gunakan itu
        elif target_count:
            for cls in class_distribution:
                targets[cls] = target_count
        
        # Jika ada target factor, kalikan jumlah current
        elif target_factor:
            for cls, files in class_to_files.items():
                current = len(files)
                targets[cls] = int(current * target_factor)
                
        # Fallback ke current count jika tidak ada target
        else:
            for cls, files in class_to_files.items():
                targets[cls] = len(files)
                
        return targets
    
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
    
    def _augment_class(
        self,
        source_files: List[Tuple[Path, Path]],
        pipeline,
        count: int,
        output_images_dir: Path,
        output_labels_dir: Path,
        pbar: Optional[tqdm] = None
    ) -> int:
        """
        Lakukan augmentasi untuk satu kelas.
        
        Args:
            source_files: Daftar pasangan file gambar dan label
            pipeline: Pipeline augmentasi
            count: Jumlah sampel yang akan digenerate
            output_images_dir: Direktori output untuk gambar
            output_labels_dir: Direktori output untuk label
            pbar: Progress bar
            
        Returns:
            Jumlah file yang berhasil digenerate
        """
        import cv2
        
        generated = 0
        iterations = 0
        max_iterations = count * 3  # Batas iterasi untuk menghindari infinite loop
        
        while generated < count and iterations < max_iterations:
            iterations += 1
            
            # Pilih file sumber secara acak
            img_path, label_path = random.choice(source_files)
            
            try:
                # Load gambar
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Load label
                bbox_data = self.utils.parse_yolo_label(label_path)
                if not bbox_data:
                    continue
                
                # Ekstrak bounding box dan class untuk albumentations
                bboxes = []
                class_labels = []
                
                for box in bbox_data:
                    bbox = box['bbox']  # Format: [x_center, y_center, width, height]
                    bboxes.append(bbox)
                    class_labels.append(box['class_id'])
                
                # Terapkan augmentasi
                try:
                    transformed = pipeline(image=img, bboxes=bboxes, class_labels=class_labels)
                    
                    aug_img = transformed['image']
                    aug_bboxes = transformed['bboxes']
                    aug_labels = transformed['class_labels']
                    
                    if not aug_bboxes:
                        # Skip jika augmentasi menghilangkan semua bbox
                        continue
                        
                    # Generate nama file baru
                    timestamp = int(time.time() * 1000)
                    new_stem = f"{img_path.stem}_aug_{timestamp}_{generated}"
                    
                    # Simpan gambar baru
                    new_img_path = output_images_dir / f"{new_stem}.jpg"
                    cv2.imwrite(str(new_img_path), cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
                    
                    # Simpan label baru
                    new_label_path = output_labels_dir / f"{new_stem}.txt"
                    with open(new_label_path, 'w') as f:
                        for cls_id, bbox in zip(aug_labels, aug_bboxes):
                            line = f"{cls_id} {' '.join(map(str, bbox))}"
                            f.write(line + '\n')
                    
                    generated += 1
                    
                    # Update progress bar
                    if pbar:
                        pbar.update(1)
                        
                except Exception as e:
                    self.logger.debug(f"⚠️ Error saat augmentasi {img_path.name}: {str(e)}")
                    continue
                    
            except Exception as e:
                self.logger.debug(f"⚠️ Error saat memproses {img_path.name}: {str(e)}")
                continue
                
        return generated

    def get_pipeline(self, augmentation_types: List[str]):
        """
        Dapatkan pipeline augmentasi untuk digunakan di luar service.
        
        Args:
            augmentation_types: Jenis augmentasi yang akan diterapkan
            
        Returns:
            Pipeline augmentasi
        """
        return self._get_augmentation_pipeline(augmentation_types)