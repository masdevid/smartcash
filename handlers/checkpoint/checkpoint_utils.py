# File: smartcash/handlers/checkpoint/checkpoint_utils.py
# Author: Alfrida Sabar
# Deskripsi: Utilitas umum untuk pengelolaan checkpoint (Diringkas)

import os
from datetime import datetime
from typing import Dict, Optional
from pathlib import Path

def generate_checkpoint_name(
    config: Dict, 
    run_type: str = 'default',
    epoch: Optional[int] = None
) -> str:
    """
    Generate nama checkpoint dengan struktur yang konsisten
    
    Args:
        config: Konfigurasi model/training
        run_type: Tipe run ('best', 'latest', 'epoch')
        epoch: Nomor epoch (untuk run_type='epoch')
    
    Returns:
        String nama checkpoint unik
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Ekstrak informasi konfigurasi
    backbone = config.get('model', {}).get('backbone', 'default')
    dataset = config.get('data', {}).get('source', 'default')
    
    # Format epoch jika disediakan
    epoch_str = f"_epoch_{epoch}" if epoch is not None and run_type == 'epoch' else ""
    
    return f"smartcash_{backbone}_{dataset}_{run_type}{epoch_str}_{timestamp}.pth"

def get_checkpoint_path(
    checkpoint_path: Optional[str], 
    output_dir: Path,
    default_type: str = 'best'
) -> Optional[str]:
    """
    Dapatkan path checkpoint yang valid.
    
    Args:
        checkpoint_path: Path yang diberikan user (bisa None)
        output_dir: Direktori utama checkpoint
        default_type: Tipe default ('best' atau 'latest')
        
    Returns:
        Path checkpoint yang valid atau None jika tidak ditemukan
    """
    # Jika path sudah diberikan dan valid
    if checkpoint_path and os.path.exists(checkpoint_path):
        return checkpoint_path
        
    # Cari checkpoint terbaik jika diminta
    if default_type == 'best':
        # Cari file dengan pola *_best_*.pth
        best_files = list(output_dir.glob('*_best_*.pth'))
        if best_files:
            # Ambil yang paling baru berdasarkan waktu modifikasi
            return str(max(best_files, key=os.path.getmtime))
    
    # Cari checkpoint latest jika diminta atau best tidak ditemukan
    latest_files = list(output_dir.glob('*_latest_*.pth'))
    if latest_files:
        return str(max(latest_files, key=os.path.getmtime))
    
    # Cari checkpoint epoch terakhir jika tidak ada best/latest
    epoch_files = list(output_dir.glob('*_epoch_*.pth'))
    if epoch_files:
        return str(max(epoch_files, key=os.path.getmtime))
    
    # Tidak ada checkpoint yang ditemukan
    return None