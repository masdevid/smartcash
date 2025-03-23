"""
File: smartcash/dataset/services/augmentor/class_balancer.py
Deskripsi: Kelas untuk balancing class dengan prioritas sampel dan tracking multi-kelas
"""

from typing import Dict, List, Tuple, Any, Optional, Callable, Set
from collections import defaultdict, Counter
import os
import logging
import random
from pathlib import Path

class ClassBalancer:
    """Kelas untuk balancing class dengan prioritas pengambilan sampel optimal"""
    
    def __init__(self, config: Dict = None, logger=None):
        """
        Inisialisasi ClassBalancer.
        
        Args:
            config: Konfigurasi aplikasi
            logger: Logger untuk logging
        """
        self.config = config or {}
        self.logger = logger or logging.getLogger(__name__)
        self.current_class_counts = defaultdict(int)  # Tracker untuk jumlah instance kelas saat ini
    
    def prepare_balanced_dataset(
        self,
        image_files: List[str],
        labels_dir: str,
        target_count: int = 1000,
        progress_callback: Optional[Callable] = None,
        filter_unknown: bool = True
    ) -> Dict[str, Any]:
        """
        Persiapkan dataset yang dibalance dengan prioritas sampling optimal.
        
        Args:
            image_files: List path file gambar
            labels_dir: Path direktori label
            target_count: Target jumlah instance per kelas
            progress_callback: Callback untuk melaporkan progres
            filter_unknown: Filter kelas unknown
            
        Returns:
            Dictionary hasil balancing
        """
        # Logging analysis start
        self.logger.info(f"🔍 Menganalisis distribusi kelas untuk balancing (target: {target_count}/kelas)")
        
        # Reset tracker jumlah instance kelas
        self.current_class_counts = defaultdict(int)
        
        # Map semua file ke kelas-kelasnya dengan informasi jumlah dan distribusi
        files_metadata, class_counts = self._map_files_with_metadata(
            image_files, labels_dir, progress_callback, filter_unknown
        )
        
        # Update current class counts dengan nilai awal
        self.current_class_counts.update(class_counts)
        
        # Hitung kebutuhan augmentasi untuk setiap kelas
        augmentation_needs = self._calculate_augmentation_needs(
            class_counts, target_count, progress_callback
        )
        
        # Pilih file untuk augmentasi dengan prioritas
        selected_files = self._select_prioritized_files(
            files_metadata, augmentation_needs, target_count, progress_callback
        )
        
        # Generate hasil
        result = {
            'class_counts': dict(class_counts),
            'files_metadata': files_metadata,
            'augmentation_needs': augmentation_needs,
            'selected_files': selected_files,
            'target_count': target_count,
            'total_classes': len(class_counts),
            'classes_to_augment': sum(1 for v in augmentation_needs.values() if v > 0),
            'total_needed': sum(augmentation_needs.values()),
            'current_class_counts': dict(self.current_class_counts)
        }
        
        # Log ringkasan balancing
        self.logger.info(f"📊 Ringkasan balancing: {result['classes_to_augment']}/{result['total_classes']} kelas perlu ditambah {result['total_needed']} instance")
        self.logger.info(f"🎯 Jumlah file terpilih untuk augmentasi: {len(selected_files)}")
        
        return result
    
    def _map_files_with_metadata(
        self, 
        image_files: List[str], 
        labels_dir: str, 
        progress_callback: Optional[Callable] = None,
        filter_unknown: bool = True
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, int]]:
        """
        Map file ke kelas dengan metadata lengkap untuk prioritisasi.
        
        Args:
            image_files: List path file gambar
            labels_dir: Path direktori label
            progress_callback: Callback untuk melaporkan progres
            filter_unknown: Filter kelas unknown
            
        Returns:
            Tuple (files_metadata, class_counts)
        """
        # Dictionary untuk menyimpan metadata file
        files_metadata = {}
        
        # Counter untuk jumlah instance per kelas
        class_counts = defaultdict(int)
        
        # Notifikasi mulai pemrosesan
        if progress_callback:
            progress_callback(
                progress=0, total=len(image_files),
                message=f"Menganalisis {len(image_files)} file untuk metadata kelas",
                status="info"
            )
        
        # Proses setiap file gambar
        for i, img_path in enumerate(image_files):
            # Dapatkan path label
            img_name = Path(img_path).stem
            label_path = os.path.join(labels_dir, f"{img_name}.txt")
            
            # Hanya proses file yang memiliki label
            if not os.path.exists(label_path):
                continue
                
            # Proses file label untuk mendapatkan semua kelas dan hitungannya
            try:
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                
                # Count kelas dalam file
                class_counter = Counter()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:  # Format YOLOv5: class_id x y width height
                        class_id = parts[0]
                        # Filter kelas "unknown" jika diminta
                        if filter_unknown and (class_id == "unknown" or class_id == "-1"):
                            continue
                        class_counter[class_id] += 1
                
                # Hanya tambahkan file ke metadata jika ada kelas valid
                if class_counter:
                    # Sortir kelas berdasarkan jumlah instance (descending)
                    sorted_classes = sorted(class_counter.items(), key=lambda x: x[1], reverse=True)
                    class_distribution = {cls: count for cls, count in sorted_classes}
                    
                    # Metadata untuk file ini
                    files_metadata[img_path] = {
                        'classes': set(class_counter.keys()),
                        'class_counts': class_distribution,
                        'num_classes': len(class_counter),
                        'total_instances': sum(class_counter.values()),
                        'primary_class': sorted_classes[0][0] if sorted_classes else None,
                    }
                    
                    # Update class_counts global
                    for cls, count in class_counter.items():
                        class_counts[cls] += count
            except Exception as e:
                self.logger.warning(f"⚠️ Error saat membaca label {label_path}: {str(e)}")
                continue
                
            # Report progress
            if progress_callback and (i % max(1, len(image_files) // 10) == 0 or i == len(image_files) - 1):
                progress_callback(
                    progress=i+1, total=len(image_files),
                    message=f"Analisis file ({i+1}/{len(image_files)}): {len(class_counts)} kelas teridentifikasi",
                    status="info"
                )
        
        # Log hasil mapping
        self.logger.info(f"✅ Identifikasi metadata selesai: {len(files_metadata)} file valid dengan {len(class_counts)} kelas teridentifikasi")
        
        return files_metadata, class_counts
    
    def _calculate_augmentation_needs(
        self, 
        class_counts: Dict[str, int], 
        target_count: int,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, int]:
        """
        Hitung kebutuhan augmentasi untuk setiap kelas.
        
        Args:
            class_counts: Dictionary jumlah instance per kelas
            target_count: Target jumlah instance per kelas
            progress_callback: Callback untuk melaporkan progres
            
        Returns:
            Dictionary kebutuhan augmentasi per kelas
        """
        # Hitung kebutuhan augmentasi
        augmentation_needs = {}
        
        # Untuk setiap kelas, hitung berapa banyak instance yang perlu ditambahkan
        for cls_id, count in class_counts.items():
            # Jika jumlah instance kurang dari target, maka perlu ditambahkan
            # Jika sudah memenuhi target, tidak perlu augmentasi
            augmentation_needs[cls_id] = max(0, target_count - count)
        
        # Log ringkasan
        classes_needing = sum(1 for needed in augmentation_needs.values() if needed > 0)
        total_needed = sum(augmentation_needs.values())
        
        if progress_callback:
            progress_callback(
                message=f"📊 Hasil analisis: {classes_needing}/{len(class_counts)} kelas perlu ditambah {total_needed} sampel",
                status="info"
            )
            
        return augmentation_needs
    
    def _select_prioritized_files(
        self, 
        files_metadata: Dict[str, Dict[str, Any]], 
        augmentation_needs: Dict[str, int],
        target_count: int,
        progress_callback: Optional[Callable] = None
    ) -> List[str]:
        """
        Pilih file dengan prioritisasi berdasarkan kelas yang perlu diaugmentasi.
        
        Args:
            files_metadata: Dictionary metadata file
            augmentation_needs: Dictionary kebutuhan augmentasi per kelas
            target_count: Target jumlah instance per kelas
            progress_callback: Callback untuk melaporkan progres
            
        Returns:
            List file yang akan diaugmentasi
        """
        if progress_callback:
            progress_callback(
                message="🔄 Memulai seleksi file dengan prioritisasi optimal",
                status="info"
            )
            
        # Dapatkan kelas yang perlu diaugmentasi
        classes_to_augment = [cls for cls, need in augmentation_needs.items() if need > 0]
        
        if not classes_to_augment:
            self.logger.info("✅ Semua kelas sudah memenuhi target, tidak perlu augmentasi")
            return []
            
        # List untuk menyimpan file yang akan diaugmentasi
        selected_files = []
        
        # Pelacakan perubahan kebutuhan selama augmentasi dinamis
        current_needs = dict(augmentation_needs)
        fulfilled_classes = set()  # Set kelas yang sudah terpenuhi target
        
        # Grup file berdasarkan kelas utamanya
        files_by_class = defaultdict(list)
        for file_path, metadata in files_metadata.items():
            primary_class = metadata.get('primary_class')
            if primary_class and primary_class in classes_to_augment:
                files_by_class[primary_class].append((file_path, metadata))
        
        # Proses kelas berdasarkan prioritas kebutuhan (terbanyak dulu)
        sorted_classes = sorted(classes_to_augment, key=lambda cls: current_needs.get(cls, 0), reverse=True)
        
        # Iterasi untuk setiap kelas yang membutuhkan augmentasi
        for i, cls_id in enumerate(sorted_classes):
            needed = current_needs.get(cls_id, 0)
            if needed <= 0 or cls_id in fulfilled_classes:
                continue
                
            # Log
            self.logger.info(f"🎯 Memproses kelas {cls_id} ({i+1}/{len(sorted_classes)}) - butuh tambahan {needed} instances")
            
            if progress_callback:
                progress_callback(
                    progress=i, total=len(sorted_classes),
                    message=f"Memproses kelas {cls_id} ({i+1}/{len(sorted_classes)}) - butuh {needed} instances",
                    status="info"
                )
            
            # 1. Cari file yang memiliki kelas ini dan tidak memiliki kelas yang sudah terpenuhi target
            high_priority_files = []
            medium_priority_files = []
            low_priority_files = []
            
            # Kategorikan file berdasarkan prioritas
            for file_path, metadata in files_metadata.items():
                if cls_id not in metadata['classes']:
                    continue
                    
                # Cek apakah file ini memiliki kelas yang sudah terpenuhi target
                has_fulfilled = any(cls in fulfilled_classes for cls in metadata['classes'])
                
                # Prioritas Atas: Tidak memiliki kelas terpenuhi dan jumlah kelas terkecil
                if not has_fulfilled:
                    high_priority_files.append((file_path, metadata))
                
                # Prioritas Menengah: Jumlah kelas terkecil
                elif metadata['num_classes'] <= 2:  # Asumsi 2 atau kurang kelas masih prioritas menengah
                    medium_priority_files.append((file_path, metadata))
                    
                # Prioritas Bawah: Sisanya
                else:
                    low_priority_files.append((file_path, metadata))
            
            # Sortir file berdasarkan jumlah kelas (terkecil dulu)
            high_priority_files.sort(key=lambda x: x[1]['num_classes'])
            medium_priority_files.sort(key=lambda x: x[1]['num_classes'])
            
            # Gabungkan semua file berdasarkan prioritas
            prioritized_files = high_priority_files + medium_priority_files + low_priority_files
            
            # Pilih file berdasarkan prioritas dan kebutuhan
            files_for_class = []
            remaining_need = needed
            
            for file_path, metadata in prioritized_files:
                if remaining_need <= 0:
                    break
                    
                # Tambahkan file ke daftar terpilih
                files_for_class.append(file_path)
                
                # Kurangi kebutuhan berdasarkan kontribusi file ini untuk kelas saat ini
                contribution = metadata['class_counts'].get(cls_id, 0)
                remaining_need -= contribution
                
                # Update kebutuhan untuk semua kelas dalam file ini
                for file_cls, count in metadata['class_counts'].items():
                    if file_cls in current_needs:
                        current_needs[file_cls] = max(0, current_needs[file_cls] - count)
                        
                        # Cek apakah kelas ini sudah terpenuhi target
                        self.current_class_counts[file_cls] += count
                        if self.current_class_counts[file_cls] >= target_count:
                            fulfilled_classes.add(file_cls)
                            self.logger.info(f"✅ Kelas {file_cls} telah mencapai target {target_count}")
            
            # Tambahkan semua file untuk kelas ini ke daftar terpilih
            selected_files.extend(files_for_class)
            
            # Cek apakah kelas saat ini sudah terpenuhi
            if remaining_need <= 0 or self.current_class_counts[cls_id] >= target_count:
                fulfilled_classes.add(cls_id)
                self.logger.info(f"✅ Kelas {cls_id} telah mencapai target {target_count}")
            else:
                self.logger.info(f"⚠️ Kelas {cls_id} masih membutuhkan {remaining_need} instances")
                
            # Report progress
            if progress_callback:
                progress_callback(
                    progress=i+1, total=len(sorted_classes),
                    message=f"Kelas {cls_id}: Dipilih {len(files_for_class)} file (Kelas terpenuhi: {len(fulfilled_classes)}/{len(classes_to_augment)})",
                    status="info"
                )
        
        # Deduplikasi file terpilih
        selected_files = list(set(selected_files))
        
        # Log hasil
        self.logger.info(f"✅ Pemilihan file selesai: {len(selected_files)} file unik terpilih untuk augmentasi")
        self.logger.info(f"✅ Status kelas: {len(fulfilled_classes)}/{len(classes_to_augment)} kelas diperkirakan akan terpenuhi")
        
        if progress_callback:
            progress_callback(
                message=f"✅ Pemilihan file selesai: {len(selected_files)} file terpilih untuk augmentasi",
                status="success"
            )
            
        return selected_files
    
    def update_class_counts(self, augmentation_results: List[Dict[str, Any]], target_count: int) -> Dict[str, int]:
        """
        Update jumlah instance kelas berdasarkan hasil augmentasi.
        
        Args:
            augmentation_results: Hasil augmentasi dari proses sebelumnya
            target_count: Target jumlah instance per kelas
            
        Returns:
            Dictionary kelas yang sudah terpenuhi target
        """
        fulfilled_classes = {}
        
        # Update count untuk setiap kelas yang diaugmentasi
        for result in augmentation_results:
            if result.get('status') != 'success':
                continue
                
            # Ambil informasi kelas
            class_id = result.get('class_id')
            all_classes = result.get('all_classes', [class_id] if class_id else [])
            generated = result.get('generated', 0)
            
            # Update count untuk semua kelas yang ada dalam file
            for cls in all_classes:
                if not cls:
                    continue
                    
                # Update count
                self.current_class_counts[cls] += generated
                
                # Cek jika sudah mencapai target
                if self.current_class_counts[cls] >= target_count:
                    fulfilled_classes[cls] = self.current_class_counts[cls]
        
        # Log kelas yang sudah terpenuhi
        # if fulfilled_classes:
        #     self.logger.info(f"✅ Kelas yang mencapai target setelah augmentasi: {', '.join(fulfilled_classes.keys())}")
        
        return fulfilled_classes