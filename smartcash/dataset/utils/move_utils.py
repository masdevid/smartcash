"""
File: smartcash/dataset/utils/move_utils.py
Deskripsi: Utilitas untuk pemindahan dan pengelolaan file dataset
"""

import os
import shutil
import glob
from pathlib import Path
from typing import Dict, Any, List, Optional
from tqdm.notebook import tqdm
from smartcash.dataset.utils.dataset_constants import DEFAULT_SPLITS

# Definisi ekstensi gambar yang didukung
IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']

def move_files_to_preprocessed(
    images_output_dir: str, 
    labels_output_dir: str, 
    output_prefix: str, 
    final_output_dir: str,
    split: str, 
    logger=None
) -> bool:
    """
    Pindahkan file augmentasi ke direktori preprocessed.
    
    Args:
        images_output_dir: Direktori output gambar
        labels_output_dir: Direktori output label
        output_prefix: Prefix output file
        final_output_dir: Direktori tujuan akhir
        split: Nama split dataset
        logger: Logger untuk mencatat aktivitas
        
    Returns:
        Boolean yang menunjukkan keberhasilan
    """
    try:
        # Buat direktori target dan dapatkan file dengan one-liner
        [os.makedirs(os.path.join(final_output_dir, split, subdir), exist_ok=True) 
         for subdir in ['images', 'labels']]
        
        # Cari semua file augmentasi dengan pattern yang lebih fleksibel
        augmented_files = []
        for ext in IMG_EXTENSIONS:
            pattern = os.path.join(images_output_dir, f"{output_prefix}_*{ext}")
            augmented_files.extend(glob.glob(pattern))
        
        # Hanya log jumlah file yang ditemukan
        if logger:
            logger.info(f"📦 Ditemukan {len(augmented_files)} file augmentasi")
        
        # Jika tidak ada file yang ditemukan, coba cari semua file di direktori (tanpa log detail)
        if not augmented_files:
            all_files = []
            for ext in IMG_EXTENSIONS:
                pattern = os.path.join(images_output_dir, f"*{ext}")
                all_files.extend(glob.glob(pattern))
        
        # Pindahkan file yang ditemukan dengan progress bar
        moved_count = 0
        total_files = len(augmented_files)
        
        # Buat progress bar dengan deskripsi yang jelas
        with tqdm(total=total_files, desc=f"📦 Menyalin file ke {split}", unit="file", colour="green") as pbar:
            for img_file in augmented_files:
                img_name = os.path.basename(img_file)
                label_name = f"{os.path.splitext(img_name)[0]}.txt"
                
                # Define target paths
                img_target = os.path.join(final_output_dir, split, 'images', img_name)
                label_target = os.path.join(final_output_dir, split, 'labels', label_name)
                label_file = os.path.join(labels_output_dir, label_name)
                
                # Copy file dengan progress bar
                for src, dst in [(img_file, img_target), (label_file, label_target)]:
                    if os.path.exists(src):
                        # Copy file tanpa menghapus aslinya
                        shutil.copy2(src, dst)
                        moved_count += 1
                
                # Update progress bar setiap gambar (bukan setiap file)
                pbar.update(1)
        
        if logger:
            logger.info(f"✅ Berhasil memindahkan {moved_count} file ke {os.path.join(final_output_dir, split)}")
        
        return moved_count > 0
    except Exception as e:
        if logger: 
            logger.error(f"❌ Error saat memindahkan file: {str(e)}")
        return False

def get_source_dir(split: str, config: Dict) -> str:
    """
    Dapatkan direktori sumber data split.
    
    Args:
        split: Nama split ('train', 'valid', 'test')
        config: Konfigurasi aplikasi
        
    Returns:
        Path direktori sumber sebagai string
    """
    # Dapatkan direktori data dari config atau gunakan default
    data_dir = config.get('dataset_dir', config.get('data', {}).get('dir', 'data'))
    
    # Pastikan data_dir adalah string
    if isinstance(data_dir, Path):
        data_dir = str(data_dir)
    
    # Jika punya local.split, gunakan itu
    if 'local' in config.get('data', {}) and split in config.get('data', {}).get('local', {}):
        path = config['data']['local'][split]
        # Pastikan path adalah string
        return str(path) if isinstance(path, Path) else path
    
    # Fallback ke direktori default
    return os.path.join(data_dir, split)

def calculate_total_images(splits: List[str], config: Dict) -> Dict[str, int]:
    """
    Hitung total gambar di setiap split.
    
    Args:
        splits: List nama split yang akan dihitung
        config: Konfigurasi aplikasi
        
    Returns:
        Dictionary jumlah gambar per split
    """
    total_by_split = {}
    
    for split in splits:
        # Konversi path ke string untuk menghindari masalah dengan PosixPath
        source_dir = str(get_source_dir(split, config))
        images_dir = os.path.join(source_dir, 'images')
        
        # Periksa apakah direktori ada
        if os.path.exists(images_dir) and os.path.isdir(images_dir):
            # Hitung jumlah gambar dengan ekstensi yang didukung
            count = 0
            for ext in IMG_EXTENSIONS:
                pattern = os.path.join(images_dir, f'*{ext}')
                count += len(glob.glob(pattern))
            
            total_by_split[split] = count
        else:
            # Jika direktori tidak ada, set count ke 0
            total_by_split[split] = 0
    
    # Hitung total semua split
    total_by_split['all'] = sum(total_by_split.values())
    return total_by_split

def resolve_splits(split: Optional[str]) -> List[str]:
    """
    Resolve split parameter menjadi list split yang akan diproses.
    
    Args:
        split: Nama split ('train', 'valid', 'test', 'all', None)
        
    Returns:
        List nama split yang akan diproses
    """
    if not split or split.lower() == 'all': 
        return DEFAULT_SPLITS
    elif split.lower() == 'val': 
        return ['valid']
    else: 
        return [split]