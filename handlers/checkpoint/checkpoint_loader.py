# File: smartcash/handlers/checkpoint/checkpoint_loader.py
# Author: Alfrida Sabar
# Deskripsi: Komponen untuk loading model checkpoint (Diringkas)

import torch
from datetime import datetime
from typing import Dict, Optional, Any
from pathlib import Path

from smartcash.utils.logger import get_logger

class CheckpointLoader:
    """Komponen untuk loading model checkpoint dengan penanganan error yang robust."""
    
    def __init__(self, output_dir: Path, logger=None):
        """
        Inisialisasi loader.
        
        Args:
            output_dir: Direktori yang berisi checkpoint
            logger: Logger kustom (opsional)
        """
        self.output_dir = output_dir
        self.logger = logger or get_logger("checkpoint_loader")
        
        # Dapatkan finder dan history dari sibling modules
        from smartcash.handlers.checkpoint.checkpoint_finder import CheckpointFinder
        from smartcash.handlers.checkpoint.checkpoint_history import CheckpointHistory
        
        self.finder = CheckpointFinder(output_dir, logger)
        self.history = CheckpointHistory(output_dir, logger)
    
    def load_checkpoint(
        self, 
        checkpoint_path=None,
        device=None,
        model=None,
        optimizer=None,
        scheduler=None
    ) -> Dict:
        """
        Muat checkpoint dengan dukungan berbagai opsi dan resume training.
        
        Args:
            checkpoint_path: Path ke checkpoint (jika None, akan mengambil checkpoint terbaik)
            device: Perangkat untuk memuat model
            model: Model yang akan dimuat dengan weights (opsional)
            optimizer: Optimizer yang akan dimuat dengan state (opsional)
            scheduler: Scheduler yang akan dimuat dengan state (opsional)
            
        Returns:
            Dict berisi metadata dan state_dict
        """
        device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        try:
            # Cari checkpoint terbaik jika path tidak diberikan
            if checkpoint_path is None:
                checkpoint_path = self.finder.find_best_checkpoint()
                if checkpoint_path is None:
                    checkpoint_path = self.finder.find_latest_checkpoint()
                    if checkpoint_path is None:
                        raise FileNotFoundError("Tidak ditemukan checkpoint yang valid")
            
            # Muat checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # Muat state ke model jika disediakan
            if model is not None and 'model_state_dict' in checkpoint:
                try:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    self.logger.info(f"✅ Model state berhasil dimuat")
                except Exception as e:
                    self.logger.warning(f"⚠️ Mencoba memuat dengan strict=False...")
                    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            
            # Muat state ke optimizer
            if optimizer is not None and 'optimizer_state_dict' in checkpoint:
                try: optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                except Exception as e: self.logger.warning(f"⚠️ Gagal memuat state optimizer: {str(e)}")
            
            # Muat state ke scheduler
            if scheduler is not None and 'scheduler_state_dict' in checkpoint:
                try: scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                except Exception as e: self.logger.warning(f"⚠️ Gagal memuat state scheduler: {str(e)}")
            
            # Update riwayat resume
            self.history.update_resume_history(checkpoint_path)
            
            self.logger.info(f"📂 Checkpoint berhasil dimuat dari: {checkpoint_path}")
            
            # Tambahkan metadata berguna ke hasil
            checkpoint['loaded_from'] = checkpoint_path
            checkpoint['load_time'] = datetime.now().isoformat()
            
            return checkpoint
            
        except Exception as e:
            self.logger.error(f"❌ Gagal memuat checkpoint: {str(e)}")
            return {
                'epoch': 0, 'model_state_dict': None, 'optimizer_state_dict': None,
                'scheduler_state_dict': None, 'metrics': {}, 'config': {},
                'timestamp': datetime.now().isoformat(), 'error': str(e)
            }