"""
File: smartcash/dataset/utils/augmentor_utils.py
Deskripsi: Fungsi helper untuk augmentasi dataset dengan pendekatan DRY dan one-liner
"""

import os, glob, shutil, random
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Callable, Set, Union
from collections import defaultdict

def get_class_from_label(label_path: str) -> Optional[str]:
   """Ekstrak ID kelas utama dari file label YOLOv5."""
   try:
       if not os.path.exists(label_path): return None
       # Baca file dan ekstrak class_ids dengan one-liner
       with open(label_path, 'r') as f:
           class_ids = [parts[0] for line in f.readlines() 
                     if len(parts := line.strip().split()) >= 5]
       return min(class_ids) if class_ids else None
   except Exception:
       return None

def process_label_file(label_path: str, collect_all_classes: bool = False) -> Tuple[Optional[str], Set[str]]:
   """Proses file label untuk mendapatkan kelas utama dan semua kelas."""
   try:
       if not os.path.exists(label_path): return None, set()
       # Baca file dan ekstrak class_ids dengan one-liner
       with open(label_path, 'r') as f:
           class_ids = [parts[0] for line in f.readlines() 
                     if len(parts := line.strip().split()) >= 5]
       return (min(class_ids) if class_ids else None, 
               set(class_ids) if collect_all_classes and class_ids else set())
   except Exception:
       return None, set()

def move_files_to_preprocessed(images_output_dir: str, labels_output_dir: str, 
                            output_prefix: str, final_output_dir: str,
                            split: str, logger=None) -> bool:
   """Pindahkan file augmentasi ke direktori preprocessed."""
   try:
       # Buat direktori target dan dapatkan file dengan one-liner
       [os.makedirs(os.path.join(final_output_dir, split, subdir), exist_ok=True) 
        for subdir in ['images', 'labels']]
       augmented_files = glob.glob(os.path.join(images_output_dir, f"{output_prefix}_*.jpg"))
       
       if logger: logger.info(f"📦 Memindahkan {len(augmented_files)} file augmentasi ke {final_output_dir}/{split}")
       
       # Pindahkan file dengan one-liner looping
       for img_file in augmented_files:
           img_name = os.path.basename(img_file)
           label_name = f"{os.path.splitext(img_name)[0]}.txt"
           
           # Define target paths dengan one-liner
           img_target, label_target = [os.path.join(final_output_dir, split, subdir, file_name) 
                                    for subdir, file_name in [('images', img_name), ('labels', label_name)]]
           label_file = os.path.join(labels_output_dir, label_name)
           
           # Move files dengan one-liner conditional
           for src, dst in [(img_file, img_target), (label_file, label_target)]:
               if os.path.exists(src): shutil.copy2(src, dst); os.remove(src)
       
       return True
   except Exception as e:
       if logger: logger.error(f"❌ Error saat memindahkan file: {str(e)}")
       return False

def process_augmentation_results(results: List[Dict], logger=None) -> Dict[str, Any]:
   """Proses hasil augmentasi untuk statistik konsolidasian."""
   try:
       # Statistik dasar dengan one-liner
       stats = {
           'successful': [r for r in results if r.get('status') == 'success'],
           'failed': [r for r in results if r.get('status') != 'success'],
           'total_generated': sum(r.get('generated', 0) for r in results),
           'class_stats': defaultdict(lambda: {'files': 0, 'generated': 0})
       }
       
       # Populate class stats dengan one-liner
       [stats['class_stats'][result.get('class_id', 'unknown')].update({
           'files': stats['class_stats'][result.get('class_id', 'unknown')]['files'] + 1,
           'generated': stats['class_stats'][result.get('class_id', 'unknown')]['generated'] + result.get('generated', 0)
       }) for result in stats['successful']]
       
       # Tambahan statistik dengan one-liner
       stats.update({
           'total_files': len(results),
           'success_rate': len(stats['successful']) / max(1, len(results)),
           'generated_per_file': stats['total_generated'] / max(1, len(stats['successful'])),
           'unique_classes': len(stats['class_stats'])
       })
       
       return stats
   except Exception as e:
       if logger: logger.error(f"❌ Error saat memproses hasil augmentasi: {str(e)}")
       return {
           'total_files': len(results),
           'total_generated': sum(r.get('generated', 0) for r in results if isinstance(r, dict)),
           'error': str(e)
       }

def map_and_analyze_files(
   image_files: List[str], 
   labels_dir: str,
   target_count: int = 1000,
   progress_callback: Optional[Callable] = None
) -> Tuple[Dict[str, List[str]], Dict[str, int], Dict[str, int], List[str]]:
   """
   Konsolidasi fungsi mapping, needs calculation, dan selection dengan pendekatan DRY.
   
   Returns:
       Tuple (files_by_class, class_counts, augmentation_needs, selected_files)
   """
   files_by_class, class_counts = defaultdict(list), defaultdict(int)
   
   # Notifikasi mulai pemrosesan
   if progress_callback:
       progress_callback(
           progress=0, total=len(image_files),
           message=f"Menganalisis {len(image_files)} file dataset untuk balancing",
           status="info", step=0
       )
   
   # Proses semua file sekaligus dengan progress reporting
   for i, img_path in enumerate(image_files):
       img_name = Path(img_path).stem
       label_path = str(Path(labels_dir) / f"{img_name}.txt")
       
       # Proses label file untuk mendapatkan kelas
       main_class, all_classes = process_label_file(label_path, True)
       if not main_class: continue
       
       # Update tracking kelas dengan one-liner
       files_by_class[main_class].append(img_path)
       [class_counts.update({cls: class_counts[cls] + 1}) for cls in all_classes]
       
       # Report progres dengan throttling
       if progress_callback and (i % max(1, len(image_files) // 10) == 0 or i == len(image_files) - 1):
           progress_callback(
               progress=i+1, total=len(image_files),
               message=f"Analisis file ({int((i+1)/len(image_files)*100)}%): {i+1}/{len(image_files)}",
               status="info", step=0
           )
   
   # Calculate augmentation needs dengan one-liner
   augmentation_needs = {cls_id: max(0, target_count - count) 
                      for cls_id, count in class_counts.items()}
   
   # Pilih file untuk augmentasi
   classes_to_augment = [cls_id for cls_id, needed in augmentation_needs.items() if needed > 0]
   selected_files = []
   
   # Proses file selection untuk setiap kelas dengan one-liner where possible
   for i, class_id in enumerate(classes_to_augment):
       needed = augmentation_needs.get(class_id, 0)
       available_files = files_by_class.get(class_id, [])
       
       if available_files and needed > 0:
           # Pilih file secara efisien dengan one-liner
           files_to_augment = (random.sample(available_files, min(len(available_files), needed)) 
                            if len(available_files) > needed else available_files)
           selected_files.extend(files_to_augment)
       
       # Progress reporting untuk class processing
       if progress_callback and (i % max(1, len(classes_to_augment) // 5) == 0 or i == len(classes_to_augment) - 1):
           progress_callback(
               progress=i+1, total=len(classes_to_augment),
               message=f"Pemilihan file ({int((i+1)/len(classes_to_augment)*100)}%): {i+1}/{len(classes_to_augment)} kelas",
               status="info", step=0
           )
   
   # Final progress reporting
   if progress_callback:
       progress_callback(
           message=f"✅ Analisis selesai: {len(selected_files)} file dipilih dari {len(class_counts)} kelas",
           status="info", step=0
       )
   
   return files_by_class, class_counts, augmentation_needs, selected_files

# Backward compatibility untuk API lama
def map_files_to_classes(image_files: List[str], labels_dir: str, progress_callback: Optional[Callable] = None) -> Tuple[Dict[str, List[str]], Dict[str, int]]:
   """Backward compatibility untuk map_files_to_classes."""
   files_by_class, class_counts, _, _ = map_and_analyze_files(image_files, labels_dir, progress_callback=progress_callback)
   return files_by_class, class_counts

def calculate_augmentation_needs(class_counts: Dict[str, int], target_count: int, progress_callback: Optional[Callable] = None) -> Dict[str, int]:
   """Backward compatibility untuk calculate_augmentation_needs."""
   augmentation_needs = {cls_id: max(0, target_count - count) for cls_id, count in class_counts.items()}
   
   # Log ringkasan hasil kebutuhan augmentasi untuk backward compatibility
   if progress_callback:
       classes_needing = sum(1 for needed in augmentation_needs.values() if needed > 0)
       total_needed = sum(augmentation_needs.values())
       progress_callback(
           message=f"📊 Hasil analisis: {classes_needing}/{len(class_counts)} kelas perlu ditambah {total_needed} sampel",
           status="info", step=0
       )
   
   return augmentation_needs

def select_files_for_augmentation(files_by_class: Dict[str, List[str]], augmentation_needs: Dict[str, int], progress_callback: Optional[Callable] = None) -> List[str]:
   """Backward compatibility untuk select_files_for_augmentation."""
   classes_to_augment = [cls_id for cls_id, needed in augmentation_needs.items() if needed > 0]
   selected_files = []
   
   for class_id in classes_to_augment:
       needed = augmentation_needs.get(class_id, 0)
       available_files = files_by_class.get(class_id, [])
       
       if available_files and needed > 0:
           num_files_to_select = min(len(available_files), needed)
           files_to_augment = random.sample(available_files, num_files_to_select) if len(available_files) > num_files_to_select else available_files
           selected_files.extend(files_to_augment)
   
   # Log ringkasan untuk backward compatibility
   if progress_callback:
       progress_callback(
           message=f"✅ Pemilihan selesai: {len(selected_files)} file dipilih untuk augmentasi",
           status="info", step=0
       )
   
   return selected_files