# File: smartcash/handlers/checkpoint/checkpoint_finder.py
# Author: Alfrida Sabar
# Deskripsi: Pencarian checkpoint model berdasarkan berbagai kriteria (Diringkas)

from typing import Dict, List, Optional
from pathlib import Path

from smartcash.utils.logger import get_logger

class CheckpointFinder:
    """Pencarian checkpoint model berdasarkan berbagai kriteria."""
    
    def __init__(self, output_dir: Path, logger=None):
        """
        Inisialisasi pencari checkpoint.
        
        Args:
            output_dir: Direktori yang berisi checkpoint
            logger: Logger kustom (opsional)
        """
        self.output_dir = output_dir
        self.logger = logger or get_logger("checkpoint_finder")
        
        # Dapatkan history manager
        from smartcash.handlers.checkpoint.checkpoint_history import CheckpointHistory
        self.history = CheckpointHistory(output_dir, logger)
    
    def find_best_checkpoint(self) -> Optional[str]:
        """
        Temukan checkpoint terbaik berdasarkan riwayat training.
        
        Returns:
            Path ke checkpoint terbaik, atau None jika tidak ada
        """
        try:
            history = self.history.get_history()
            
            # Urutkan berdasarkan is_best flag
            best_runs = [
                run for run in history.get('runs', [])
                if run.get('is_best', False)
            ]
            
            if best_runs:
                # Ambil checkpoint terbaik terakhir
                best_run = max(best_runs, key=lambda x: x.get('timestamp', ''))
                best_checkpoint_name = best_run['checkpoint_name']
                best_checkpoint_path = self.output_dir / best_checkpoint_name
                
                # Pastikan file ada
                if best_checkpoint_path.exists():
                    return str(best_checkpoint_path)
            
            # Jika tidak ada checkpoint terbaik valid, cari checkpoint terakhir
            latest_checkpoint = self.find_latest_checkpoint()
            return latest_checkpoint
            
        except Exception as e:
            self.logger.error(f"❌ Gagal mencari checkpoint terbaik: {str(e)}")
            return None
    
    def find_latest_checkpoint(self) -> Optional[str]:
        """
        Temukan checkpoint terakhir berdasarkan waktu modifikasi.
        
        Returns:
            Path ke checkpoint terakhir, atau None jika tidak ada
        """
        try:
            # Cari semua file checkpoint
            checkpoint_files = list(self.output_dir.glob('*.pth'))
            
            if not checkpoint_files:
                return None
            
            # Urutkan berdasarkan waktu modifikasi (terbaru dulu)
            latest_checkpoint = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
            return str(latest_checkpoint)
            
        except Exception as e:
            self.logger.error(f"❌ Gagal mencari checkpoint terakhir: {str(e)}")
            return None
    
    def find_checkpoint_by_epoch(self, epoch: int) -> Optional[str]:
        """
        Temukan checkpoint untuk epoch tertentu.
        
        Args:
            epoch: Nomor epoch yang dicari
            
        Returns:
            Path ke checkpoint epoch, atau None jika tidak ada
        """
        try:
            # Cari file dengan pola *_epoch_{epoch}_*.pth
            epoch_files = list(self.output_dir.glob(f'*_epoch_{epoch}_*.pth'))
            
            if not epoch_files:
                return None
            
            # Ambil yang paling baru jika ada beberapa
            epoch_checkpoint = max(epoch_files, key=lambda x: x.stat().st_mtime)
            return str(epoch_checkpoint)
            
        except Exception as e:
            self.logger.error(f"❌ Gagal mencari checkpoint untuk epoch {epoch}: {str(e)}")
            return None
    
    def list_checkpoints(self) -> Dict[str, List[Path]]:
        """
        Dapatkan daftar semua checkpoint yang tersedia.
        
        Returns:
            Dict berisi list checkpoint berdasarkan tipe
        """
        try:
            checkpoints = {
                'best': list(self.output_dir.glob('*_best_*.pth')),
                'latest': list(self.output_dir.glob('*_latest_*.pth')),
                'epoch': list(self.output_dir.glob('*_epoch_*.pth')),
                'emergency': list(self.output_dir.glob('*emergency*.pth'))
            }
            
            # Sort berdasarkan waktu modifikasi (terbaru dulu)
            for category, files in checkpoints.items():
                checkpoints[category] = sorted(
                    files, 
                    key=lambda x: x.stat().st_mtime, 
                    reverse=True
                )
            
            return checkpoints
            
        except Exception as e:
            self.logger.error(f"❌ Gagal mendapatkan daftar checkpoint: {str(e)}")
            return {'best': [], 'latest': [], 'epoch': [], 'emergency': []}