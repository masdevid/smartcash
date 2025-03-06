# File: smartcash/utils/optimized_augmentation.py
# Author: Alfrida Sabar
# Deskripsi: Sistem augmentasi terpadu yang mengoptimalkan berbagai jenis augmentasi dataset

import os
import cv2
import numpy as np
import albumentations as A
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Set
import json
import random
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor
import time
import threading

from smartcash.utils.logger import SmartCashLogger
from smartcash.utils.layer_config_manager import get_layer_config

class OptimizedAugmentation:
    """
    Sistem augmentasi terpadu dengan dukungan untuk:
    - Augmentasi multilayer  
    - Pipeline augmentasi yang terspesialisasi
    - Validasi otomatis
    - Eksekusi paralel dengan checkpointing
    """
    
    def __init__(
        self,
        config: Dict,
        output_dir: Optional[str] = None,
        logger: Optional[SmartCashLogger] = None,
        num_workers: int = 4,
        checkpoint_interval: int = 50
    ):
        """
        Inisialisasi augmentasi teroptimasi.
        
        Args:
            config: Konfigurasi aplikasi
            output_dir: Direktori output
            logger: Logger kustom
            num_workers: Jumlah worker untuk multiprocessing
            checkpoint_interval: Interval checkpoint dalam jumlah gambar
        """
        self.logger = logger or SmartCashLogger(__name__)
        self.config = config
        self.num_workers = num_workers
        self.checkpoint_interval = checkpoint_interval
        
        # Setup path
        self.data_dir = Path(config.get('data_dir', 'data'))
        self.output_dir = Path(output_dir) if output_dir else self.data_dir
        
        # Load layer config
        self.layer_config_manager = get_layer_config()
        self.active_layers = config.get('layers', ['banknote'])
        
        # Validasi layer yang aktif
        for layer in self.active_layers:
            if layer not in self.layer_config_manager.get_layer_names():
                self.logger.warning(f"⚠️ Layer tidak dikenali: {layer}")
                self.active_layers.remove(layer)
        
        if not self.active_layers:
            self.logger.warning("⚠️ Tidak ada layer aktif yang valid, fallback ke 'banknote'")
            self.active_layers = ['banknote']
        
        # Setup pipeline augmentasi
        self._setup_augmentation_pipelines()
        
        # Thread lock untuk statistik
        self._stats_lock = threading.Lock()
        
        # Track progress dan checkpoint
        self._last_checkpoint = None
        self._checkpoint_file = self.output_dir / ".aug_checkpoint.json"
        
        # Inisialisasi statistik
        self.reset_stats()
    
    def _setup_augmentation_pipelines(self):
        """Setup pipeline augmentasi berdasarkan konfigurasi."""
        # Ekstrak parameter dari config
        aug_config = self.config.get('training', {})
        
        # Pipeline posisi - variasi posisi/orientasi uang
        self.position_aug = A.Compose([
            A.SafeRotate(limit=30, p=0.7),
            A.Perspective(scale=(0.05, 0.1), p=0.5),
            A.HorizontalFlip(p=aug_config.get('fliplr', 0.5)),
            A.ShiftScaleRotate(
                shift_limit=aug_config.get('translate', 0.1),
                scale_limit=aug_config.get('scale', 0.5),
                rotate_limit=aug_config.get('degrees', 45),
                p=0.5
            )
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels'],
            min_visibility=0.3
        ))
        
        # Pipeline pencahayaan - variasi cahaya dan kontras
        self.lighting_aug = A.Compose([
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=0.7
            ),
            A.RandomShadow(p=0.5),
            A.HueSaturationValue(
                hue_shift_limit=aug_config.get('hsv_h', 0.015),
                sat_shift_limit=aug_config.get('hsv_s', 0.7),
                val_shift_limit=aug_config.get('hsv_v', 0.4),
                p=0.3
            ),
            A.ImageCompression(
                quality_lower=80,
                quality_upper=100,
                p=0.2
            )
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels'],
            min_visibility=0.3
        ))
        
        # Pipeline kombinasi - gabungan posisi dan pencahayaan
        self.combined_aug = A.Compose([
            A.OneOf([
                A.SafeRotate(limit=30, p=0.7),
                A.Perspective(scale=(0.05, 0.1), p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=aug_config.get('translate', 0.1),
                    scale_limit=aug_config.get('scale', 0.5),
                    rotate_limit=aug_config.get('degrees', 45),
                    p=0.5
                )
            ], p=0.7),
            A.OneOf([
                A.RandomBrightnessContrast(p=0.5),
                A.RandomShadow(p=0.5),
                A.HueSaturationValue(p=0.5)
            ], p=0.7),
            A.HorizontalFlip(p=aug_config.get('fliplr', 0.3)),
            A.OneOf([
                A.MotionBlur(p=0.2),
                A.GaussianBlur(p=0.2),
                A.GaussNoise(p=0.2)
            ], p=0.2)
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels'],
            min_visibility=0.3
        ))
        
        # Pipeline khusus untuk rotasi ekstrim
        self.extreme_rotation_aug = A.Compose([
            A.RandomRotate90(p=0.8),
            A.ShiftScaleRotate(
                shift_limit=0.2,
                scale_limit=0.2,
                rotate_limit=90,
                p=0.8
            ),
            A.RandomBrightnessContrast(p=0.3)
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels'],
            min_visibility=0.2
        ))
        
        self.logger.info(f"✅ Pipeline augmentasi siap: 4 variasi tersedia")
    
    def reset_stats(self):
        """Reset statistik augmentasi."""
        with self._stats_lock:
            self.stats = {
                'processed': 0,
                'augmented': 0,
                'failed': 0,
                'skipped_invalid': 0,
                'layer_stats': {layer: 0 for layer in self.active_layers},
                'per_type': {
                    'position': 0,
                    'lighting': 0,
                    'combined': 0,
                    'extreme_rotation': 0
                },
                'duration': 0.0,
                'start_time': time.time()
            }
    
    def augment_image(
        self,
        image_path: Path,
        label_path: Optional[Path] = None,
        augmentation_type: str = 'combined',
        output_prefix: str = 'aug',
        variations: int = 3
    ) -> Tuple[List[np.ndarray], List[Dict[str, List[Tuple[int, List[float]]]]], List[Path]]:
        """
        Augmentasi satu gambar dengan berbagai variasi.
        
        Args:
            image_path: Path ke file gambar
            label_path: Path ke file label (opsional)
            augmentation_type: Tipe augmentasi 
            output_prefix: Prefix untuk nama file output
            variations: Jumlah variasi yang dihasilkan
            
        Returns:
            Tuple of (augmented_images, augmented_layer_labels, output_paths)
        """
        try:
            # Baca gambar
            image = cv2.imread(str(image_path))
            if image is None:
                self.logger.warning(f"⚠️ Tidak dapat membaca gambar: {image_path}")
                with self._stats_lock:
                    self.stats['failed'] += 1
                return [], [], []
                
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Baca dan validasi label
            all_bboxes = []
            all_class_labels = []
            layer_labels = {layer: [] for layer in self.active_layers}
            
            # Load class-to-layer mapping
            class_to_layer_map = {}
            for layer_name in self.active_layers:
                layer_config = self.layer_config_manager.get_layer_config(layer_name)
                for cls_id in layer_config['class_ids']:
                    class_to_layer_map[cls_id] = layer_name
            
            if label_path and label_path.exists():
                try:
                    with open(label_path, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 5:  # class_id, x, y, w, h
                                try:
                                    cls_id = int(float(parts[0]))
                                    bbox = [float(x) for x in parts[1:5]]
                                    
                                    # Validasi nilai koordinat (harus antara 0-1)
                                    if any(not (0 <= coord <= 1) for coord in bbox):
                                        continue
                                        
                                    # Periksa apakah class ID termasuk dalam layer yang diaktifkan
                                    if cls_id in class_to_layer_map:
                                        layer_name = class_to_layer_map[cls_id]
                                        
                                        # Tambahkan ke list semua bboxes untuk augmentasi
                                        all_bboxes.append(bbox)
                                        all_class_labels.append(cls_id)
                                        
                                        # Tambahkan ke layer yang sesuai
                                        layer_labels[layer_name].append((cls_id, bbox))
                                        
                                        # Update statistik layer
                                        with self._stats_lock:
                                            self.stats['layer_stats'][layer_name] += 1
                                        
                                except (ValueError, IndexError):
                                    continue
                except Exception as e:
                    self.logger.warning(f"⚠️ Error membaca label {label_path}: {str(e)}")
            
            # Periksa apakah ada label yang valid
            if not all_bboxes and augmentation_type != 'lighting':
                # Untuk augmentasi selain lighting, perlu ada bbox
                with self._stats_lock:
                    self.stats['skipped_invalid'] += 1
                return [], [], []
            
            # Pilih pipeline augmentasi
            if augmentation_type == 'position':
                augmentor = self.position_aug
            elif augmentation_type == 'lighting':
                augmentor = self.lighting_aug
            elif augmentation_type == 'extreme_rotation':
                augmentor = self.extreme_rotation_aug
            else:  # default: combined
                augmentor = self.combined_aug
            
            # Generate variasi
            augmented_images = []
            augmented_layer_labels = []
            output_paths = []
            
            for i in range(variations):
                # Apply augmentation
                try:
                    if all_bboxes:
                        # Augmentasi dengan bounding box
                        augmented = augmentor(
                            image=image,
                            bboxes=all_bboxes,
                            class_labels=all_class_labels
                        )
                        
                        # Reorganisasi hasil augmentasi ke format per layer
                        augmented_labels = {layer: [] for layer in self.active_layers}
                        
                        for bbox, cls_id in zip(augmented['bboxes'], augmented['class_labels']):
                            if cls_id in class_to_layer_map:
                                layer_name = class_to_layer_map[cls_id]
                                augmented_labels[layer_name].append((cls_id, bbox))
                        
                        augmented_layer_labels.append(augmented_labels)
                    else:
                        # Augmentasi hanya gambar
                        augmented = augmentor(image=image)
                        
                        # Gunakan label kosong
                        augmented_layer_labels.append({layer: [] for layer in self.active_layers})
                    
                    augmented_images.append(augmented['image'])
                    
                    # Generate output paths
                    suffix = f"{output_prefix}_{augmentation_type}_{i+1}"
                    img_output_path = image_path.parent / f"{image_path.stem}_{suffix}{image_path.suffix}"
                    
                    output_paths.append(img_output_path)
                    with self._stats_lock:
                        self.stats['augmented'] += 1
                        self.stats['per_type'][augmentation_type] += 1
                    
                except Exception as e:
                    self.logger.error(f"❌ Augmentasi gagal untuk {image_path.name} ({augmentation_type}): {str(e)}")
                    continue
                    
            with self._stats_lock:
                self.stats['processed'] += 1
                
            return augmented_images, augmented_layer_labels, output_paths
            
        except Exception as e:
            self.logger.error(f"❌ Error saat mengaugmentasi {image_path}: {str(e)}")
            with self._stats_lock:
                self.stats['failed'] += 1
            return [], [], []
    
    def save_augmented_data(
        self,
        image: np.ndarray,
        layer_labels: Dict[str, List[Tuple[int, List[float]]]],
        image_path: Path,
        labels_dir: Path
    ) -> bool:
        """
        Simpan hasil augmentasi ke file dengan format multilayer.
        
        Args:
            image: Gambar hasil augmentasi
            layer_labels: Label per layer hasil augmentasi
            image_path: Path untuk menyimpan gambar
            labels_dir: Direktori untuk menyimpan label
            
        Returns:
            Boolean sukses/gagal
        """
        try:
            # Simpan gambar
            img_dir = image_path.parent
            img_dir.mkdir(parents=True, exist_ok=True)
            
            cv2.imwrite(
                str(image_path),
                cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            )
            
            # Simpan label jika ada
            has_labels = any(len(labels) > 0 for labels in layer_labels.values())
            
            if has_labels:
                labels_dir.mkdir(parents=True, exist_ok=True)
                label_path = labels_dir / f"{image_path.stem}.txt"
                
                with open(label_path, 'w') as f:
                    for layer, labels in layer_labels.items():
                        for cls_id, bbox in labels:
                            if len(bbox) == 4:  # x, y, w, h
                                line = f"{int(cls_id)} {' '.join(map(str, bbox))}\n"
                                f.write(line)
                            
            return True
        except Exception as e:
            self.logger.error(f"❌ Gagal menyimpan hasil augmentasi: {str(e)}")
            return False
    
    def load_checkpoint(self, split: str) -> List[str]:
        """
        Load checkpoint augmentasi jika ada.
        
        Args:
            split: Split dataset yang sedang diproses
            
        Returns:
            List file yang sudah diproses
        """
        # Checkpoint per split
        checkpoint_file = self.output_dir / f".aug_checkpoint_{split}.json"
        self._checkpoint_file = checkpoint_file
        
        if checkpoint_file.exists():
            try:
                with open(checkpoint_file, 'r') as f:
                    checkpoint_data = json.load(f)
                
                processed_files = checkpoint_data.get('processed_files', [])
                
                # Load statistik
                if 'stats' in checkpoint_data:
                    with self._stats_lock:
                        # Hanya update statistik yang ada di struktur saat ini
                        checkpoint_stats = checkpoint_data['stats']
                        for key, value in checkpoint_stats.items():
                            if key in self.stats:
                                if isinstance(value, dict) and isinstance(self.stats[key], dict):
                                    # Update nested dictionaries
                                    for subkey, subvalue in value.items():
                                        if subkey in self.stats[key]:
                                            self.stats[key][subkey] = subvalue
                                else:
                                    self.stats[key] = value
                
                if processed_files:
                    self.logger.info(f"🔄 Melanjutkan dari checkpoint: {len(processed_files)} file telah diproses")
                
                return processed_files
            except Exception as e:
                self.logger.warning(f"⚠️ Gagal memuat checkpoint: {str(e)}")
                return []
        else:
            return []
    
    def save_checkpoint(self, processed_files: List[str]):
        """
        Simpan checkpoint untuk melanjutkan proses yang terganggu.
        
        Args:
            processed_files: List file yang sudah diproses
        """
        try:
            # Buat data checkpoint
            checkpoint_data = {
                'timestamp': time.time(),
                'processed_files': processed_files,
                'stats': self.stats
            }
            
            # Simpan ke file
            with open(self._checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f)
                
            self._last_checkpoint = time.time()
            
        except Exception as e:
            self.logger.error(f"❌ Gagal menyimpan checkpoint: {str(e)}")
    
    def augment_dataset(
        self,
        split: str = 'train',
        augmentation_types: List[str] = ['combined'],
        num_variations: int = 2,
        output_prefix: str = 'aug',
        resume: bool = True,
        validate_results: bool = True
    ) -> Dict[str, int]:
        """
        Augmentasi dataset dengan validasi layer label.
        
        Args:
            split: Split dataset ('train', 'valid', 'test')
            augmentation_types: List tipe augmentasi yang akan digunakan
            num_variations: Jumlah variasi per tipe augmentasi
            output_prefix: Prefix untuk nama file output
            resume: Lanjutkan dari checkpoint jika ada
            validate_results: Validasi hasil augmentasi
            
        Returns:
            Dict statistik augmentasi
        """
        # Validasi tipe augmentasi
        valid_types = ['position', 'lighting', 'combined', 'extreme_rotation']
        augmentation_types = [t for t in augmentation_types if t in valid_types]
        
        if not augmentation_types:
            self.logger.warning("⚠️ Tidak ada tipe augmentasi valid yang ditentukan, menggunakan 'combined'")
            augmentation_types = ['combined']
        
        # Setup direktori
        input_dir = self.data_dir / split
        output_dir = self.output_dir / split
        
        images_dir = input_dir / 'images'
        labels_dir = input_dir / 'labels'
        output_images_dir = output_dir / 'images'
        output_labels_dir = output_dir / 'labels'
        
        # Pastikan direktori output ada
        output_images_dir.mkdir(parents=True, exist_ok=True)
        output_labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Reset statistik jika tidak melanjutkan
        if not resume:
            self.reset_stats()
        
        # Periksa file gambar yang tersedia
        image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.jpeg')) + list(images_dir.glob('*.png'))
        
        if not image_files:
            self.logger.warning(f"⚠️ Tidak ada gambar ditemukan di {images_dir}")
            return self.stats
        
        self.logger.info(
            f"🔍 Menemukan {len(image_files)} gambar di {split}\n"
            f"   Menggunakan {len(augmentation_types)} tipe augmentasi: {', '.join(augmentation_types)}\n"
            f"   Variasi per tipe: {num_variations}"
        )
        
        # Process gambar yang sudah diproses dari checkpoint jika resume
        processed_files = self.load_checkpoint(split) if resume else []
        processed_files_set = set(processed_files)
        
        # Filter gambar yang belum diproses
        unprocessed_images = [img for img in image_files if str(img) not in processed_files_set]
        
        self.logger.info(
            f"🔄 {len(processed_files)} gambar telah diproses sebelumnya\n"
            f"   {len(unprocessed_images)} gambar akan diproses"
        )
        
        if not unprocessed_images:
            self.logger.success("✅ Semua gambar telah diproses!")
            return self.stats
        
        # Fungsi untuk memproses satu gambar
        def process_image(img_path):
            # Simpan file paths yang telah berhasil diproses
            successful_files = []
            
            # Cari label yang sesuai
            label_path = labels_dir / f"{img_path.stem}.txt"
            
            for aug_type in augmentation_types:
                # Augmentasi
                aug_images, aug_layer_labels, out_paths = self.augment_image(
                    img_path,
                    label_path if label_path.exists() else None,
                    aug_type,
                    output_prefix,
                    variations=num_variations
                )
                
                # Simpan hasil
                for idx, (img, out_path) in enumerate(zip(aug_images, out_paths)):
                    # Ubah path ke direktori output
                    save_path = output_images_dir / out_path.name
                    
                    # Ambil label layer yang sesuai dengan gambar ini
                    layer_labels_to_save = aug_layer_labels[idx] if aug_layer_labels and idx < len(aug_layer_labels) else {}
                    
                    # Simpan hasil
                    if self.save_augmented_data(img, layer_labels_to_save, save_path, output_labels_dir):
                        successful_files.append(aug_type + "_" + save_path.name)
            
            # Jika berhasil, tambahkan path gambar asli ke daftar yang sudah diproses
            if successful_files:
                return str(img_path), successful_files
            else:
                return None, []
        
        # Proses menggunakan ThreadPoolExecutor dengan progress bar
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit semua tugas
            futures = {executor.submit(process_image, img_path): img_path for img_path in unprocessed_images}
            
            # Inisialisasi progress bar
            pbar = tqdm(
                total=len(unprocessed_images),
                desc=f"🎨 Augmentasi {split}",
                unit="img"
            )
            
            # Track waktu untuk checkpoint
            last_checkpoint_time = time.time()
            
            # Proses hasil setiap future yang selesai
            for future in futures:
                img_path, successful_files = future.result()
                pbar.update(1)
                
                if img_path:
                    processed_files.append(img_path)
                
                # Checkpoint setiap interval atau 5 menit
                current_time = time.time()
                if (len(processed_files) % self.checkpoint_interval == 0) or (current_time - last_checkpoint_time > 300):
                    self.save_checkpoint(processed_files)
                    last_checkpoint_time = current_time
                    
                    # Update statistik dalam progress bar
                    with self._stats_lock:
                        success_rate = (self.stats['augmented'] / max(1, self.stats['processed'] * len(augmentation_types) * num_variations)) * 100
                        pbar.set_postfix({
                            'augmented': self.stats['augmented'],
                            'success_rate': f"{success_rate:.1f}%"
                        })
            
            pbar.close()
        
        # Validasi hasil jika diminta
        if validate_results:
            self.logger.info("🔍 Memvalidasi hasil augmentasi...")
            validation_stats = self._validate_augmentation_results(output_dir)
            self.stats['validation'] = validation_stats
            
        # Simpan checkpoint final
        self.save_checkpoint(processed_files)
        
        # Perbarui statistik durasi
        with self._stats_lock:
            self.stats['duration'] = time.time() - self.stats['start_time']
            
            # Hitung dan tampilkan statistik
            total_images = len(image_files)
            total_augmented = self.stats['augmented']
            expected_augmented = total_images * len(augmentation_types) * num_variations
            completion_rate = (total_augmented / max(1, expected_augmented)) * 100
            
            self.logger.success(
                f"✨ Augmentasi {split} selesai dalam {self.stats['duration']:.1f} detik\n"
                f"   Gambar asli: {total_images}\n"
                f"   Total hasil augmentasi: {total_augmented}\n"
                f"   Rasio penyelesaian: {completion_rate:.1f}%\n"
                f"   Gagal: {self.stats['failed']}\n"
                f"   Dilewati (tidak valid): {self.stats['skipped_invalid']}"
            )
            
            # Log statistik per tipe augmentasi
            self.logger.info("📊 Statistik per tipe augmentasi:")
            for aug_type, count in self.stats['per_type'].items():
                if aug_type in augmentation_types:
                    pct = (count / max(1, total_augmented)) * 100
                    self.logger.info(f"   • {aug_type}: {count} ({pct:.1f}%)")
            
            # Log statistik layer
            self.logger.info("📊 Statistik per layer:")
            for layer, count in self.stats['layer_stats'].items():
                if layer in self.active_layers:
                    self.logger.info(f"   • {layer}: {count} objek")
        
        return self.stats
    
    def _validate_augmentation_results(self, output_dir: Path) -> Dict[str, int]:
        """
        Validasi hasil augmentasi untuk memastikan konsistensi label.
        
        Args:
            output_dir: Direktori output augmentasi
            
        Returns:
            Dict statistik validasi
        """
        validation_stats = {
            'valid_images': 0,
            'valid_labels': 0,
            'invalid_images': 0,
            'invalid_labels': 0,
            'layer_consistency': {layer: 0 for layer in self.active_layers}
        }
        
        images_dir = output_dir / 'images'
        labels_dir = output_dir / 'labels'
        
        if not images_dir.exists() or not labels_dir.exists():
            self.logger.warning(f"⚠️ Direktori output tidak lengkap untuk validasi")
            return validation_stats
        
        # Validasi gambar yang diaugmentasi (berisi 'aug' di nama file)
        augmented_images = [f for f in images_dir.glob('*.*') if 'aug' in f.name]
        
        if not augmented_images:
            self.logger.warning(f"⚠️ Tidak ada gambar hasil augmentasi ditemukan")
            return validation_stats
        
        self.logger.info(f"🔍 Memvalidasi {len(augmented_images)} gambar hasil augmentasi...")
        
        # Sampel untuk validasi (max 100 gambar untuk performa)
        sample_size = min(100, len(augmented_images))
        if len(augmented_images) > sample_size:
            validation_sample = random.sample(augmented_images, sample_size)
            self.logger.info(f"   Menggunakan sampel {sample_size} gambar untuk validasi")
        else:
            validation_sample = augmented_images
        
        # Validasi sampel
        for img_path in tqdm(validation_sample, desc="Validasi Augmentasi", ncols=80):
            # Baca gambar
            img = cv2.imread(str(img_path))
            if img is None or img.size == 0:
                validation_stats['invalid_images'] += 1
                continue
                
            validation_stats['valid_images'] += 1
            
            # Periksa label
            label_path = labels_dir / f"{img_path.stem}.txt"
            if not label_path.exists():
                continue
                
            try:
                layer_present = {layer: False for layer in self.active_layers}
                
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                
                if not lines:
                    continue
                    
                valid_label = True
                
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        try:
                            cls_id = int(float(parts[0]))
                            bbox = [float(x) for x in parts[1:5]]
                            
                            # Validasi nilai koordinat
                            if any(not (0 <= coord <= 1) for coord in bbox):
                                valid_label = False
                                break
                                
                            # Periksa layer
                            layer_name = self.layer_config_manager.get_layer_for_class_id(cls_id)
                            if layer_name in self.active_layers:
                                layer_present[layer_name] = True
                            
                        except (ValueError, IndexError):
                            valid_label = False
                            break
                
                if valid_label:
                    validation_stats['valid_labels'] += 1
                    
                    # Update statistik konsistensi layer
                    for layer, present in layer_present.items():
                        if present:
                            validation_stats['layer_consistency'][layer] += 1
                else:
                    validation_stats['invalid_labels'] += 1
                    
            except Exception as e:
                self.logger.error(f"❌ Error validasi {label_path}: {str(e)}")
                validation_stats['invalid_labels'] += 1
        
        # Ekstrapolasi hasil untuk seluruh dataset
        if len(validation_sample) < len(augmented_images):
            scaling_factor = len(augmented_images) / len(validation_sample)
            
            for key in ['valid_images', 'valid_labels', 'invalid_images', 'invalid_labels']:
                validation_stats[key] = int(validation_stats[key] * scaling_factor)
                
            for layer in validation_stats['layer_consistency']:
                validation_stats['layer_consistency'][layer] = int(
                    validation_stats['layer_consistency'][layer] * scaling_factor
                )
        
        # Log hasil validasi
        valid_pct = (validation_stats['valid_images'] / max(1, len(augmented_images))) * 100
        self.logger.info(
            f"✅ Validasi augmentasi:\n"
            f"   Gambar valid: {validation_stats['valid_images']}/{len(augmented_images)} ({valid_pct:.1f}%)\n"
            f"   Label valid: {validation_stats['valid_labels']}\n"
            f"   Gambar tidak valid: {validation_stats['invalid_images']}\n"
            f"   Label tidak valid: {validation_stats['invalid_labels']}"
        )
        
        # Log konsistensi layer
        for layer, count in validation_stats['layer_consistency'].items():
            if layer in self.active_layers:
                percentage = (count / max(1, validation_stats['valid_images'])) * 100
                self.logger.info(f"📊 Konsistensi layer '{layer}': {count} ({percentage:.1f}%)")
        
        return validation_stats