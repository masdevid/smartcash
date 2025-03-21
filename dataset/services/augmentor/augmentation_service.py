"""
File: smartcash/dataset/services/augmentor/augmentation_service.py
Deskripsi: Layanan untuk augmentasi dataset dengan peningkatan balancing kelas dan integrasi dengan data preprocessed
"""

import os, time, random, shutil, numpy as np, uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

from smartcash.common.logger import get_logger
from smartcash.dataset.utils.dataset_utils import DatasetUtils
from smartcash.components.observer import notify, EventTopics

class AugmentationService:
    """Service untuk augmentasi dataset dengan dukungan balancing class distribution dan integrasi preprocessed."""
    
    def __init__(self, config: Dict, data_dir: str, logger=None, num_workers: int = 4):
        """Inisialisasi AugmentationService dengan progress tracking."""
        self.config = config; self.data_dir = Path(data_dir)
        self.logger = logger or get_logger("augmentation_service")
        self.num_workers = num_workers; self.utils = DatasetUtils(config, data_dir, logger)
        self._progress_callback = None; self._current_operation = ""; self._total_items = 0
        
        # Default source adalah folder preprocessed
        self.source_dir = self.config.get('preprocessing', {}).get('preprocessed_dir', 'data/preprocessed')
        self.source_prefix = self.config.get('preprocessing', {}).get('file_prefix', 'rp')
        
        # Default output juga di folder preprocessed
        self.output_dir = self.config.get('preprocessing', {}).get('preprocessed_dir', 'data/preprocessed')
        
        self.logger.info(f"🔄 AugmentationService diinisialisasi dengan {num_workers} workers dan source dari {self.source_dir}")
    
    def register_progress_callback(self, callback: Callable) -> None:
        """Register callback untuk melaporkan progress."""
        self._progress_callback = callback
    
    def _report_progress(self, progress: int, total: int, message: str, **kwargs) -> None:
        """Laporkan progress ke callback dan observer."""
        if self._progress_callback: self._progress_callback(progress=progress, total=total, message=message, **kwargs)
        try: notify(EventTopics.AUGMENTATION_PROGRESS, sender="augmentation_service", message=message, progress=progress, total=total, **kwargs)
        except (ImportError, AttributeError): pass

    def augment_dataset(self, split: str = 'train', augmentation_types: List[str] = None, 
                        target_count: int = None, target_factor: float = None, target_balance: bool = False, 
                        class_list: List[str] = None, output_dir: Optional[str] = None, source_dir: Optional[str] = None,
                        num_variations: int = None, output_prefix: str = None, process_bboxes: bool = True, 
                        validate_results: bool = True, resume: bool = False, random_seed: int = 42, num_workers: int = 4) -> Dict[str, Any]:
        """Augmentasi dataset dengan balancing class distribution berdasarkan data preprocessed."""
        self.num_workers = num_workers
        # Inisialisasi tracking dan parameter
        start_time = time.time(); random.seed(random_seed); np.random.seed(random_seed)
        self._current_operation = "augmentation"; self._total_items = 0
        
        # Notifikasi start augmentasi
        try: notify(EventTopics.AUGMENTATION_START, sender="augmentation_service", message=f"Memulai augmentasi dataset {split}")
        except (ImportError, AttributeError): pass
        
        # Parameter default dan override dari config
        aug_config = self.config.get('augmentation', {})
        augmentation_types = augmentation_types or aug_config.get('types', ['flip', 'rotate', 'brightness', 'contrast'])
        num_variations = num_variations or aug_config.get('num_variations', 2)
        output_prefix = output_prefix or aug_config.get('output_prefix', 'aug')
        process_bboxes = process_bboxes if process_bboxes is not None else aug_config.get('process_bboxes', True)
        validate_results = validate_results if validate_results is not None else aug_config.get('validate_results', True)
        
        # Dapatkan pipeline augmentasi
        pipeline = self._get_augmentation_pipeline(augmentation_types)
        
        # Setup direktori dan validasi, jika disediakan gunakan itu, jika tidak gunakan default dari preprocessing
        source_dir = source_dir or self.source_dir
        output_path = Path(output_dir or self.output_dir)
        
        # Split path dari source dir
        split_path = Path(source_dir) / split
        images_dir, labels_dir = split_path / 'images', split_path / 'labels'
        
        if not (images_dir.exists() and labels_dir.exists()):
            error_msg = f"❌ Direktori dataset tidak lengkap: {split_path}"
            self.logger.error(error_msg); self._report_error(error_msg)
            return {'status': 'error', 'message': error_msg}
        
        # Setup direktori output
        output_images_dir, output_labels_dir = output_path / split / 'images', output_path / split / 'labels'
        output_images_dir.mkdir(parents=True, exist_ok=True); output_labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Analisis distribusi kelas untuk target balancing dengan mencari file dengan prefix
        class_to_files = self._create_class_to_files_mapping_from_preprocessed(
            images_dir, labels_dir, source_prefix=self.source_prefix
        )
        
        # Log jumlah file per kelas yang ditemukan
        self.logger.info(f"🔍 Ditemukan {sum(len(files) for files in class_to_files.values())} file dengan prefix {self.source_prefix} di {images_dir}")
        for cls, files in class_to_files.items():
            self.logger.info(f"   • Kelas {cls}: {len(files)} file")
        
        # Analisis distribusi kelas untuk target balancing
        class_distribution = {cls: len(files) for cls, files in class_to_files.items()}
        
        # Hitung target jumlah per kelas berdasarkan distribusi
        targets = self._compute_augmentation_targets(
            class_distribution, class_to_files, target_count, target_factor, target_balance
        )
        
        # Log rencana augmentasi
        self._log_augmentation_plan(augmentation_types, targets, class_to_files)
        
        # Lakukan augmentasi untuk setiap kelas
        stats = {'original': 0, 'generated': 0, 'source_classes': len(class_to_files)}
        
        # Copy file original ke output dir terlebih dahulu
        self._copy_original_files(class_to_files, output_images_dir, output_labels_dir, stats)
        
        # Proses augmentasi per kelas
        self._total_items = sum(max(0, targets[cls] - len(class_to_files.get(cls, []))) for cls in targets)
        with tqdm(total=self._total_items, desc="🔄 Augmentasi dataset") as pbar:
            for cls, target in targets.items():
                if cls not in class_to_files or not class_to_files[cls]: continue
                current = len(class_to_files[cls]); to_generate = max(0, target - current)
                if to_generate <= 0: continue
                
                # Generate augmentasi untuk kelas ini
                generated = self._augment_class(cls, class_to_files[cls], pipeline, to_generate, 
                                              output_images_dir, output_labels_dir, pbar, output_prefix)
                stats['generated'] += generated
        
        # Validasi hasil jika diminta
        if validate_results:
            self._validate_augmentation_results(output_images_dir, output_labels_dir)
        
        # Rekap hasil
        elapsed_time = time.time() - start_time
        stats.update({
            'status': 'success', 'augmentation_types': augmentation_types, 'split': split,
            'output_dir': str(output_path), 'duration': elapsed_time, 'total_files': stats['original'] + stats['generated'],
            'class_distribution': class_distribution, 'balanced_targets': targets
        })
        
        # Log hasil
        self.logger.success(
            f"✅ Augmentasi dataset selesai ({elapsed_time:.1f}s):\n"
            f"   • File asli: {stats['original']}\n"
            f"   • File baru: {stats['generated']}\n"
            f"   • Total: {stats['total_files']}\n"
            f"   • Output: {output_path}"
        )
        
        # Notifikasi selesai
        try: notify(EventTopics.AUGMENTATION_END, sender="augmentation_service", message=f"Augmentasi dataset selesai: {stats['generated']} file baru", result=stats)
        except (ImportError, AttributeError): pass
        
        return stats
    
    def _create_class_to_files_mapping_from_preprocessed(self, images_dir: Path, labels_dir: Path, source_prefix: str = 'rp') -> Dict[str, List[Tuple[Path, Path]]]:
        """Buat mapping kelas ke file gambar dari data preprocessed dengan filter prefix."""
        class_to_files = defaultdict(list)
        
        # Filter gambar dengan prefix tertentu
        preprocessed_images = [img for img in images_dir.glob(f"{source_prefix}_*.jpg")]
        
        # Metode 1: Ekstrak kelas dari nama file (format: {prefix}_{class}_{uuid})
        for img_path in preprocessed_images:
            # Parse nama file untuk mengekstrak kelas
            img_name = img_path.stem
            parts = img_name.split('_')
            if len(parts) < 3 or parts[0] != source_prefix:
                continue
                
            # Kelas ada di bagian tengah (prefix_class_uuid)
            # Untuk format yang mungkin punya multiple underscore di nama kelas
            # kita ambil semua bagian tengah kecuali bagian terakhir (uuid)
            class_name = '_'.join(parts[1:-1])
            
            # Cek file label yang sesuai
            label_path = labels_dir / f"{img_name}.txt"
            if not label_path.exists():
                continue
                
            # Tambahkan ke mapping
            class_to_files[class_name].append((img_path, label_path))
            
        # Jika hasil masih kosong dan tidak ada file dengan format penamaan yang sesuai,
        # fallback ke metode alternatif: baca langsung dari file label YOLO
        if not class_to_files:
            self.logger.warning(f"⚠️ Tidak ditemukan file dengan format penamaan {source_prefix}_class_uuid, mencoba metode alternatif")
            for img_path in images_dir.glob('*.jpg'):
                label_path = labels_dir / f"{img_path.stem}.txt"
                if not label_path.exists():
                    continue
                
                # Baca file label untuk menentukan kelas
                try:
                    with open(label_path, 'r') as f:
                        label_content = f.readline().strip().split()
                        if label_content:
                            class_id = label_content[0]
                            class_to_files[f"class{class_id}"].append((img_path, label_path))
                except Exception:
                    continue
        
        return dict(class_to_files)
    
    def _log_augmentation_plan(self, augmentation_types: List[str], targets: Dict[str, int], 
                              class_to_files: Dict[str, List[Tuple[Path, Path]]]) -> None:
        """Log rencana augmentasi dengan lebih terstruktur."""
        self.logger.info(f"🔍 Augmentasi dataset dengan {len(augmentation_types)} jenis transformasi: {', '.join(augmentation_types)}")
        for cls, target in targets.items():
            current = len(class_to_files.get(cls, []))
            to_generate = max(0, target - current)
            if to_generate > 0:
                self.logger.info(f"   • {cls}: {current} → {target} (+{to_generate})")
    
    def _copy_original_files(self, class_to_files: Dict[str, List[Tuple[Path, Path]]], 
                            output_images_dir: Path, output_labels_dir: Path,
                            stats: Dict[str, int]) -> None:
        """Copy file original ke direktori output dengan progress reporting."""
        self.logger.info("📋 Menyalin file original...")
        all_files = [file for files in class_to_files.values() for file in files]
        self._report_progress(0, len(all_files), "Menyalin file original...")
        
        copied = 0
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            for files in class_to_files.values():
                for img_path, label_path in files:
                    future = executor.submit(self._copy_file_pair, img_path, label_path, 
                                           output_images_dir, output_labels_dir)
                    futures.append(future)
            
            for i, future in enumerate(futures):
                if future.result(): stats['original'] += 1
                copied += 1
                if i % max(1, len(futures)//20) == 0:  # Report progress approximately every 5%
                    self._report_progress(copied, len(futures), f"Menyalin file original ({copied}/{len(futures)})")
    
    def _report_error(self, message: str) -> None:
        """Laporkan error ke observer system."""
        try: notify(EventTopics.AUGMENTATION_ERROR, sender="augmentation_service", message=message)
        except (ImportError, AttributeError): pass
        
    def _report_warning(self, message: str) -> None:
        """Laporkan warning ke observer system."""
        try: notify(EventTopics.AUGMENTATION_WARNING, sender="augmentation_service", message=message)
        except (ImportError, AttributeError): pass
        
    def _validate_augmentation_results(self, images_dir: Path, labels_dir: Path) -> None:
        """Validasi hasil augmentasi dengan reporting."""
        self.logger.info("🔍 Memvalidasi hasil augmentasi...")
        images = list(images_dir.glob('*.*'))
        labels = list(labels_dir.glob('*.*'))
        
        # Statistik dasar
        stats = {
            'total_images': len(images),
            'total_labels': len(labels),
            'matched_pairs': 0,
            'orphaned_images': 0,
            'orphaned_labels': 0
        }
        
        # Cek pasangan image-label
        image_stems = {img.stem for img in images}
        label_stems = {label.stem for label in labels}
        
        stats['matched_pairs'] = len(image_stems.intersection(label_stems))
        stats['orphaned_images'] = len(image_stems - label_stems)
        stats['orphaned_labels'] = len(label_stems - image_stems)
        
        # Log hasil validasi
        if stats['orphaned_images'] > 0 or stats['orphaned_labels'] > 0:
            self.logger.warning(
                f"⚠️ Hasil validasi augmentasi:\n"
                f"   • Pasangan valid: {stats['matched_pairs']}\n"
                f"   • Gambar tanpa label: {stats['orphaned_images']}\n"
                f"   • Label tanpa gambar: {stats['orphaned_labels']}"
            )
        else:
            self.logger.success(f"✅ Validasi hasil: {stats['matched_pairs']} pasangan gambar-label valid")
        
        # Report via observer
        try: 
            notify(EventTopics.AUGMENTATION_VALIDATION, sender="augmentation_service", 
                  message=f"Validasi hasil: {stats['matched_pairs']} valid, {stats['orphaned_images']} gambar tanpa label",
                  stats=stats)
        except (ImportError, AttributeError): pass
    
    def _get_augmentation_pipeline(self, augmentation_types: List[str]) -> Any:
        """Dapatkan pipeline augmentasi berdasarkan jenis transformasi."""
        from smartcash.dataset.services.augmentor.pipeline_factory import AugmentationPipelineFactory
        
        # Buat pipeline sesuai jenis augmentasi
        factory = AugmentationPipelineFactory(self.config, self.logger)
        
        # Set flag untuk setiap jenis transformasi
        use_flip = any(t in ['flip', 'position', 'combined'] for t in augmentation_types)
        use_position = any(t in ['rotate', 'position', 'combined'] for t in augmentation_types)
        use_lighting = any(t in ['brightness', 'contrast', 'lighting', 'combined'] for t in augmentation_types)
        use_hsv = any(t in ['hsv', 'lighting', 'combined'] for t in augmentation_types)
        use_weather = 'weather' in augmentation_types
        
        # Buat pipeline
        return factory.create_pipeline(
            augmentation_types=[t for t in ['flip', 'rotate', 'brightness', 'contrast', 'hsv', 'weather'] 
                              if (f'use_{t}' in locals() and locals()[f'use_{t}']) or t in augmentation_types],
            img_size=(640, 640),
            include_normalize=False, # We need pixel values for saving images
            intensity=1.0,
            bbox_format='yolo'
        )
    
    def _compute_augmentation_targets(self, class_distribution: Dict[str, int],
                                    class_to_files: Dict[str, List[Tuple[Path, Path]]],
                                    target_count: Optional[int],
                                    target_factor: Optional[float],
                                    target_balance: bool) -> Dict[str, int]:
        """Hitung jumlah target augmentasi untuk balancing class distribution."""
        targets = {}
        
        # Jika tidak ada parameter, gunakan target_factor=2.0 sebagai default
        if not target_count and not target_factor and not target_balance:
            target_factor = 2.0
            
        # Jika target balancing, gunakan kelas dengan jumlah sampel terbanyak
        if target_balance and class_distribution:
            # Menggunakan kelas dengan jumlah terbanyak sebagai target semua kelas
            max_class_count = max(class_distribution.values())
            targets = {cls: max_class_count for cls in class_distribution}
            self.logger.info(f"🎯 Target balancing: Semua kelas akan dibalancing ke {max_class_count} gambar")
        # Jika ada target count, gunakan itu
        elif target_count: 
            targets = {cls: target_count for cls in class_distribution or class_to_files.keys()}
            self.logger.info(f"🎯 Target count: Semua kelas akan diaugmentasi ke {target_count} gambar")
        # Jika ada target factor, kalikan jumlah current
        elif target_factor: 
            targets = {cls: int(len(files) * target_factor) for cls, files in class_to_files.items()}
            self.logger.info(f"🎯 Target factor: Semua kelas akan diaugmentasi dengan faktor {target_factor}x")
        # Fallback ke current count jika tidak ada target
        else: 
            targets = {cls: len(files) for cls, files in class_to_files.items()}
                
        return targets
    
    def _copy_file_pair(self, img_path: Path, label_path: Path,
                       output_images_dir: Path, output_labels_dir: Path) -> bool:
        """Salin pasangan file gambar dan label ke direktori output."""
        try:
            # Salin gambar
            shutil.copy2(img_path, output_images_dir / img_path.name)
            
            # Salin label
            shutil.copy2(label_path, output_labels_dir / label_path.name)
            
            return True
        except Exception as e:
            self.logger.warning(f"⚠️ Gagal menyalin file {img_path.name}: {str(e)}")
            return False
    
    def _augment_class(self, class_name: str, source_files: List[Tuple[Path, Path]], pipeline, count: int,
                      output_images_dir: Path, output_labels_dir: Path, 
                      pbar: Optional[tqdm] = None, output_prefix: str = 'aug') -> int:
        """Lakukan augmentasi untuk satu kelas dengan naming yang konsisten dan UUID."""
        import cv2
        
        generated = 0; iterations = 0; max_iterations = count * 3  # Batas iterasi untuk menghindari infinite loop
        
        while generated < count and iterations < max_iterations:
            iterations += 1
            
            # Pilih file sumber secara acak
            img_path, label_path = random.choice(source_files)
            
            try:
                # Load gambar dan label
                img = cv2.imread(str(img_path)); 
                if img is None: continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Load label
                bbox_data = self.utils.parse_yolo_label(label_path)
                if not bbox_data: continue
                
                # Ekstrak bounding box dan class untuk albumentations
                bboxes = [box['bbox'] for box in bbox_data]  # Format: [x_center, y_center, width, height]
                class_labels = [box['class_id'] for box in bbox_data]
                
                # Terapkan augmentasi
                try:
                    transformed = pipeline(image=img, bboxes=bboxes, class_labels=class_labels)
                    
                    aug_img = transformed['image']
                    aug_bboxes = transformed['bboxes']
                    aug_labels = transformed['class_labels']
                    
                    if not aug_bboxes: continue  # Skip jika augmentasi menghilangkan semua bbox
                        
                    # Generate nama file baru dengan format {prefix}_{class_name}_{unique_id}
                    unique_id = str(uuid.uuid4())[:8]  # 8 karakter pertama dari UUID
                    new_stem = f"{output_prefix}_{class_name}_{unique_id}"
                    
                    # Simpan gambar baru
                    new_img_path = output_images_dir / f"{new_stem}.jpg"
                    cv2.imwrite(str(new_img_path), cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
                    
                    # Simpan label baru
                    new_label_path = output_labels_dir / f"{new_stem}.txt"
                    with open(new_label_path, 'w') as f:
                        f.write("\n".join(f"{cls_id} {' '.join(map(str, bbox))}" for cls_id, bbox in zip(aug_labels, aug_bboxes)))
                    
                    generated += 1
                    
                    # Update progress bar dan laporan progress
                    if pbar: pbar.update(1)
                    if generated % 10 == 0 or generated == count:
                        self._report_progress(generated, count, 
                                            f"Augmentasi kelas {class_name}: {generated}/{count}", 
                                            current_progress=generated, current_total=count)
                        
                except Exception as e:
                    self.logger.debug(f"⚠️ Error saat augmentasi {img_path.name}: {str(e)}")
                    continue
                    
            except Exception as e:
                self.logger.debug(f"⚠️ Error saat memproses {img_path.name}: {str(e)}")
                continue
                
        return generated

    def get_pipeline(self, augmentation_types: List[str]):
        """Dapatkan pipeline augmentasi untuk digunakan di luar service."""
        return self._get_augmentation_pipeline(augmentation_types)