"""
File: smartcash/dataset/services/augmentor/augmentation_service.py
Deskripsi: Layanan untuk melakukan augmentasi dataset guna memperkaya variasi data dengan progress tracking terintegrasi
"""

import os, time, random, shutil, numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from tqdm.auto import tqdm
from collections import defaultdict

from smartcash.common.logger import get_logger
from smartcash.dataset.utils.dataset_utils import DatasetUtils
from smartcash.components.observer import notify, EventTopics

class AugmentationService:
    """Service untuk augmentasi dataset untuk meningkatkan variasi data dengan progress tracking terintegrasi."""
    
    def __init__(self, config: Dict, data_dir: str, logger=None, num_workers: int = 4):
        """Inisialisasi AugmentationService dengan progress tracking."""
        self.config = config; self.data_dir = Path(data_dir)
        self.logger = logger or get_logger("augmentation_service")
        self.num_workers = num_workers; self.utils = DatasetUtils(config, data_dir, logger)
        self._progress_callback = None; self._current_operation = ""; self._total_items = 0
        self.logger.info(f"🔄 AugmentationService diinisialisasi dengan {num_workers} workers")
    
    def register_progress_callback(self, callback: Callable) -> None:
        """Register callback untuk melaporkan progress."""
        self._progress_callback = callback
    
    def _report_progress(self, progress: int, total: int, message: str, **kwargs) -> None:
        """Laporkan progress ke callback dan observer."""
        # Update progress via callback jika ada
        if self._progress_callback: self._progress_callback(progress=progress, total=total, message=message, **kwargs)
        
        # Notifikasi via observer system
        try:
            notify(EventTopics.AUGMENTATION_PROGRESS, sender="augmentation_service", 
                  message=message, progress=progress, total=total, **kwargs)
        except (ImportError, AttributeError): pass

    def augment_dataset(self, split: str = 'train', augmentation_types: List[str] = None, 
                        target_count: int = None, target_factor: float = None, target_balance: bool = False, 
                        class_list: List[str] = None, output_dir: Optional[str] = None, 
                        num_variations: int = None, output_prefix: str = None, process_bboxes: bool = True, 
                        validate_results: bool = True, resume: bool = False, random_seed: int = 42) -> Dict[str, Any]:
        """Augmentasi dataset dengan progress tracking terintegrasi."""
        # Inisialisasi tracking dan parameter
        start_time = time.time(); random.seed(random_seed); np.random.seed(random_seed)
        self._current_operation = "augmentation"; self._total_items = 0
        
        # Notifikasi start augmentasi
        try: notify(EventTopics.AUGMENTATION_START, sender="augmentation_service", 
                   message=f"Memulai augmentasi dataset {split}")
        except (ImportError, AttributeError): pass
        
        # Parameter default dan override dari config
        aug_config = self.config.get('augmentation', {})
        augmentation_types = augmentation_types or aug_config.get('types', ['flip', 'rotate', 'brightness', 'contrast'])
        num_variations = num_variations or aug_config.get('num_variations', 2)
        output_prefix = output_prefix or aug_config.get('output_prefix', 'aug')
        process_bboxes = process_bboxes if process_bboxes is not None else aug_config.get('process_bboxes', True)
        validate_results = validate_results if validate_results is not None else aug_config.get('validate_results', True)
        resume = resume if resume is not None else aug_config.get('resume', False)
        
        # Dapatkan pipeline augmentasi
        pipeline = self._get_augmentation_pipeline(augmentation_types)
        
        # Setup direktori dan validasi
        split_path = self.utils.get_split_path(split)
        images_dir, labels_dir = split_path / 'images', split_path / 'labels'
        if not (images_dir.exists() and labels_dir.exists()):
            error_msg = f"❌ Direktori dataset tidak lengkap: {split_path}"
            self.logger.error(error_msg); self._report_error(error_msg)
            return {'status': 'error', 'message': error_msg}
        
        # Setup direktori output dengan fallback ke config
        output_path = Path(output_dir or aug_config.get('output_dir', self.data_dir / f"{split}_augmented"))
        output_images_dir, output_labels_dir = output_path / 'images', output_path / 'labels'
        output_images_dir.mkdir(parents=True, exist_ok=True); output_labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Analisis distribusi kelas untuk target balancing
        class_distribution = {}
        if target_balance or class_list:
            try:
                from smartcash.dataset.services.explorer.class_explorer import ClassExplorer
                explorer = ClassExplorer(self.config, str(self.data_dir), self.logger, self.num_workers)
                result = explorer.analyze_distribution(split)
                if result['status'] == 'success':
                    class_distribution = result['counts']
                    class_distribution = {cls: count for cls, count in class_distribution.items() if not class_list or cls in class_list}
            except ImportError: self.logger.warning("⚠️ ClassExplorer tidak tersedia, skip analisis distribusi kelas")
        
        # Dapatkan semua file gambar valid
        image_files = self.utils.find_image_files(images_dir)
        if not image_files:
            error_msg = f"⚠️ Tidak ada gambar ditemukan di {images_dir}"
            self.logger.warning(error_msg); self._report_warning(error_msg)
            return {'status': 'warning', 'message': error_msg}
        
        # Buat mapping file berdasarkan kelas
        class_to_files = self._create_class_to_files_mapping(image_files, labels_dir)
        
        # Hitung target jumlah per kelas
        targets = self._compute_augmentation_targets(class_distribution, class_to_files, target_count, target_factor, target_balance)
        
        # Log rencana augmentasi
        self._log_augmentation_plan(augmentation_types, targets, class_to_files)
        
        # Lakukan augmentasi untuk setiap kelas
        stats = {'original': 0, 'generated': 0, 'source_classes': len(class_to_files)}
        
        # Copy file original ke output dir terlebih dahulu
        self._copy_original_files(class_to_files, output_images_dir, output_labels_dir, stats)
        
        # Proses augmentasi per kelas
        self._total_items = sum(max(0, targets[cls] - len(class_to_files.get(cls, []))) for cls in targets)
        processed = 0
        with tqdm(total=self._total_items, desc="🔄 Augmentasi dataset") as pbar:
            for cls, target in targets.items():
                if cls not in class_to_files or not class_to_files[cls]: continue
                current = len(class_to_files[cls]); to_generate = max(0, target - current)
                if to_generate <= 0: continue
                
                # Generate augmentasi untuk kelas ini
                for i in range(to_generate):
                    # Cek apakah user meminta stop
                    if not hasattr(self, '_progress_callback') or not self._progress_callback: break
                    
                    # Pilih file sumber secara acak
                    img_path, label_path = random.choice(class_to_files[cls])
                    
                    try:
                        # Load gambar dan label
                        img = self._load_image(img_path)
                        if img is None: continue
                        
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
                                
                            # Generate nama file baru
                            timestamp = int(time.time() * 1000)
                            new_stem = f"{img_path.stem}_{output_prefix}_{timestamp}_{i}"
                            
                            # Simpan gambar baru
                            new_img_path = output_images_dir / f"{new_stem}.jpg"
                            self._save_image(aug_img, new_img_path)
                            
                            # Simpan label baru
                            new_label_path = output_labels_dir / f"{new_stem}.txt"
                            with open(new_label_path, 'w') as f:
                                f.write("\n".join(f"{cls_id} {' '.join(map(str, bbox))}" for cls_id, bbox in zip(aug_labels, aug_bboxes)))
                            
                            stats['generated'] += 1
                            processed += 1
                            
                            # Update progress bar dan laporan progress
                            pbar.update(1)
                            if processed % 10 == 0 or processed == self._total_items:
                                self._report_progress(processed, self._total_items, 
                                                    f"Augmentasi kelas: {processed}/{self._total_items}", 
                                                    current_progress=processed, current_total=self._total_items)
                                
                        except Exception as e:
                            self.logger.debug(f"⚠️ Error saat augmentasi {img_path.name}: {str(e)}")
                            continue
                            
                    except Exception as e:
                        self.logger.debug(f"⚠️ Error saat memproses {img_path.name}: {str(e)}")
                        continue
        
        # Validasi hasil jika diminta
        if validate_results:
            self._validate_augmentation_results(output_images_dir, output_labels_dir)
        
        # Rekap hasil
        elapsed_time = time.time() - start_time
        stats.update({
            'status': 'success', 'augmentation_types': augmentation_types, 'split': split,
            'output_dir': str(output_path), 'duration': elapsed_time, 'total_files': stats['original'] + stats['generated']
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
        try: notify(EventTopics.AUGMENTATION_END, sender="augmentation_service", 
                   message=f"Augmentasi dataset selesai: {stats['generated']} file baru", result=stats)
        except (ImportError, AttributeError): pass
        
        return stats
    
    def _create_class_to_files_mapping(self, image_files: List[Path], labels_dir: Path) -> Dict[str, List[Tuple[Path, Path]]]:
        """Buat mapping kelas ke file gambar dengan lebih efisien."""
        class_to_files = defaultdict(list)
        
        for img_path in image_files:
            label_path = labels_dir / f"{img_path.stem}.txt"
            if not label_path.exists(): continue
            
            # Parse label untuk menentukan kelas
            bbox_data = self.utils.parse_yolo_label(label_path)
            classes_in_image = {box.get('class_name', box.get('class_id', '')) for box in bbox_data}
            
            # Tambahkan file ke semua kelas yang ada di gambar
            for class_name in classes_in_image:
                if class_name and img_path not in [path for path, _ in class_to_files[class_name]]:
                    class_to_files[class_name].append((img_path, label_path))
        
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
        for files in class_to_files.values():
            for img_path, label_path in files:
                if self._copy_file_pair(img_path, label_path, output_images_dir, output_labels_dir):
                    stats['original'] += 1
                copied += 1
                if copied % max(1, len(all_files)//20) == 0:  # Report progress approximately every 5%
                    self._report_progress(copied, len(all_files), f"Menyalin file original ({copied}/{len(all_files)})")
    
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
        """Hitung jumlah target augmentasi untuk setiap kelas dengan one-liner style."""
        targets = {}
        
        # Jika tidak ada parameter, gunakan target_factor=2.0 sebagai default
        if not target_count and not target_factor and not target_balance:
            target_factor = 2.0
            
        # Jika target balancing, gunakan kelas dengan jumlah sampel terbanyak
        if target_balance and class_distribution: 
            targets = {cls: max(class_distribution.values()) for cls in class_distribution}
        # Jika ada target count, gunakan itu
        elif target_count: 
            targets = {cls: target_count for cls in class_distribution or class_to_files.keys()}
        # Jika ada target factor, kalikan jumlah current
        elif target_factor: 
            targets = {cls: int(len(files) * target_factor) for cls, files in class_to_files.items()}
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
    
    def _load_image(self, img_path: Path) -> np.ndarray:
        """Load gambar dengan berbagai format yang didukung."""
        import cv2
        try:
            img = cv2.imread(str(img_path))
            if img is None: return None
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            self.logger.debug(f"⚠️ Error saat load gambar {img_path}: {str(e)}")
            return None
    
    def _save_image(self, img: np.ndarray, output_path: Path) -> bool:
        """Simpan gambar ke file dengan kompresi optimal."""
        import cv2
        try:
            # Konversi ke BGR untuk OpenCV
            bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            # Simpan dengan kompresi JPEG
            cv2.imwrite(str(output_path), bgr_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            return True
        except Exception as e:
            self.logger.debug(f"⚠️ Error saat menyimpan gambar {output_path}: {str(e)}")
            return False
    
    def get_pipeline(self, augmentation_types: List[str]):
        """Dapatkan pipeline augmentasi untuk digunakan di luar service."""
        return self._get_augmentation_pipeline(augmentation_types)