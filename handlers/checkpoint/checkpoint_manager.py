# File: smartcash/handlers/checkpoint/checkpoint_manager.py
# Author: Alfrida Sabar
# Deskripsi: Manager utama untuk pengelolaan checkpoint model dengan API terpadu

import os
import torch
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path

from smartcash.utils.logger import SmartCashLogger
from smartcash.handlers.checkpoint.checkpoint_loader import CheckpointLoader
from smartcash.handlers.checkpoint.checkpoint_saver import CheckpointSaver
from smartcash.handlers.checkpoint.checkpoint_finder import CheckpointFinder
from smartcash.handlers.checkpoint.checkpoint_history import CheckpointHistory
from smartcash.handlers.checkpoint.checkpoint_utils import get_checkpoint_path

class CheckpointManager:
    """
    Manager utama untuk pengelolaan checkpoint model SmartCash.
    
    Mengimplementasikan pola Facade untuk menyediakan API terpadu ke
    semua komponen terkait checkpoint seperti loading, saving, pencarian,
    dan pengelolaan history.
    """
    
    def __init__(
        self,
        output_dir: str = 'runs/train/weights', 
        logger: Optional[SmartCashLogger] = None
    ):
        """
        Inisialisasi CheckpointManager
        
        Args:
            output_dir: Direktori untuk menyimpan checkpoint
            logger: Logger kustom (opsional)
        """
        self.output_dir = Path(output_dir)
        self.logger = logger or SmartCashLogger(__name__)
        
        # Buat direktori jika belum ada
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Inisialisasi komponen
        self.loader = CheckpointLoader(self.output_dir, self.logger)
        self.saver = CheckpointSaver(self.output_dir, self.logger)
        self.finder = CheckpointFinder(self.output_dir, self.logger)
        self.history = CheckpointHistory(self.output_dir, self.logger)
        
        self.logger.info(f"🔧 CheckpointManager diinisialisasi: {self.output_dir}")
    
    # ===== Metode untuk loading checkpoint =====
    
    def load_checkpoint(
        self, 
        checkpoint_path: Optional[str] = None,
        device: Optional[torch.device] = None,
        model: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None
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
        return self.loader.load_checkpoint(
            checkpoint_path=checkpoint_path,
            device=device,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler
        )
    
    def load_state_without_model(
        self, 
        checkpoint_path: Optional[str] = None,
        device: Optional[torch.device] = None
    ) -> Dict:
        """
        Muat checkpoint metadata, tanpa memasukkan state ke model.
        
        Args:
            checkpoint_path: Path ke checkpoint (jika None, akan mengambil checkpoint terbaik)
            device: Perangkat untuk memuat state
            
        Returns:
            Dict berisi metadata dan state_dict
        """
        return self.loader.load_state_without_model(checkpoint_path, device)
    
    def validate_checkpoint(self, checkpoint_path: str) -> Tuple[bool, Dict]:
        """
        Validasi apakah checkpoint valid dan lengkap.
        
        Args:
            checkpoint_path: Path ke checkpoint
            
        Returns:
            Tuple (is_valid, message_dict)
        """
        return self.loader.validate_checkpoint(checkpoint_path)
    
    def extract_metadata(self, checkpoint_path: str) -> Dict:
        """
        Ekstrak metadata dari checkpoint tanpa memuat model state.
        
        Args:
            checkpoint_path: Path ke checkpoint
            
        Returns:
            Dict metadata
        """
        return self.loader.extract_metadata(checkpoint_path)
    
    # ===== Metode untuk penyimpanan checkpoint =====
    
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        config: Dict,
        epoch: int,
        metrics: Dict,
        is_best: bool = False,
        save_optimizer: bool = True
    ) -> Dict[str, str]:
        """
        Simpan checkpoint model dengan metadata komprehensif.
        
        Args:
            model: Model PyTorch
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            config: Konfigurasi training
            epoch: Epoch saat ini
            metrics: Metrik training
            is_best: Apakah ini model terbaik
            save_optimizer: Apakah menyimpan state optimizer
        
        Returns:
            Dict berisi path checkpoint yang disimpan
        """
        return self.saver.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            epoch=epoch,
            metrics=metrics,
            is_best=is_best,
            save_optimizer=save_optimizer
        )
    
    def save_multiple_checkpoints(
        self,
        state_dict: Dict[str, Any],
        config: Dict,
        outputs: List[str] = ['best', 'latest']
    ) -> Dict[str, str]:
        """
        Simpan state model ke multiple output files (best, latest, etc).
        
        Args:
            state_dict: State model dan metadata
            config: Konfigurasi model
            outputs: List tipe output ('best', 'latest', 'epoch')
            
        Returns:
            Dict berisi path checkpoint untuk setiap tipe
        """
        return self.saver.save_multiple_checkpoints(state_dict, config, outputs)
    
    def cleanup_checkpoints(
        self, 
        max_checkpoints: int = 5,
        keep_best: bool = True,
        keep_latest: bool = True,
        max_epochs: int = 5
    ) -> List[str]:
        """
        Bersihkan checkpoint lama dengan opsi fleksibel
        
        Args:
            max_checkpoints: Jumlah maksimal checkpoint per kategori
            keep_best: Pertahankan semua checkpoint terbaik
            keep_latest: Pertahankan checkpoint latest terakhir
            max_epochs: Jumlah maksimal checkpoint epoch yang disimpan
            
        Returns:
            List path checkpoint yang dihapus
        """
        return self.saver.cleanup_checkpoints(
            max_checkpoints=max_checkpoints,
            keep_best=keep_best,
            keep_latest=keep_latest,
            max_epochs=max_epochs
        )
    
    def copy_to_drive(
        self, 
        drive_dir: str = '/content/drive/MyDrive/SmartCash/checkpoints',
        best_only: bool = False
    ) -> List[str]:
        """
        Salin checkpoint ke Google Drive
        
        Args:
            drive_dir: Direktori tujuan di Google Drive
            best_only: Hanya salin checkpoint terbaik
            
        Returns:
            List path checkpoint yang disalin
        """
        return self.saver.copy_to_drive(drive_dir, best_only)
    
    # ===== Metode untuk pencarian checkpoint =====
    
    def find_best_checkpoint(self) -> Optional[str]:
        """
        Temukan checkpoint terbaik berdasarkan riwayat training.
        
        Returns:
            Path ke checkpoint terbaik, atau None jika tidak ada
        """
        return self.finder.find_best_checkpoint()
    
    def find_latest_checkpoint(self) -> Optional[str]:
        """
        Temukan checkpoint terakhir berdasarkan waktu modifikasi.
        
        Returns:
            Path ke checkpoint terakhir, atau None jika tidak ada
        """
        return self.finder.find_latest_checkpoint()
    
    def find_checkpoint_by_epoch(self, epoch: int) -> Optional[str]:
        """
        Temukan checkpoint untuk epoch tertentu.
        
        Args:
            epoch: Nomor epoch yang dicari
            
        Returns:
            Path ke checkpoint epoch, atau None jika tidak ada
        """
        return self.finder.find_checkpoint_by_epoch(epoch)
    
    def list_checkpoints(self) -> Dict[str, List[Path]]:
        """
        Dapatkan daftar semua checkpoint yang tersedia.
        
        Returns:
            Dict berisi list checkpoint berdasarkan tipe
        """
        return self.finder.list_checkpoints()
    
    def filter_checkpoints(
        self, 
        backbone: Optional[str] = None, 
        dataset: Optional[str] = None,
        min_epoch: Optional[int] = None
    ) -> List[Path]:
        """
        Filter checkpoint berdasarkan kriteria.
        
        Args:
            backbone: Filter berdasarkan backbone (opsional)
            dataset: Filter berdasarkan dataset (opsional)
            min_epoch: Filter berdasarkan epoch minimal (opsional)
            
        Returns:
            List checkpoint yang sesuai kriteria
        """
        return self.finder.filter_checkpoints(backbone, dataset, min_epoch)
    
    # ===== Metode untuk pengelolaan riwayat =====
    
    def get_training_history(self) -> Dict[str, Any]:
        """
        Ambil seluruh riwayat training.
        
        Returns:
            Dict riwayat training
        """
        return self.history.get_history()
    
    def export_history_to_json(self, output_path: Optional[str] = None) -> str:
        """
        Export riwayat training ke JSON.
        
        Args:
            output_path: Path output (opsional)
            
        Returns:
            Path file JSON
        """
        return self.history.export_to_json(output_path)
    
    # ===== Metode utilitas tambahan =====
    
    def get_checkpoint_path(self, checkpoint_path: Optional[str] = None) -> Optional[str]:
        """
        Dapatkan path checkpoint yang valid.
        
        Args:
            checkpoint_path: Path checkpoint (jika None, cari checkpoint terbaik)
            
        Returns:
            Path checkpoint valid atau None jika tidak ditemukan
        """
        return get_checkpoint_path(checkpoint_path, self.output_dir)
    
    def display_checkpoints(self) -> None:
        """
        Tampilkan daftar checkpoint yang tersedia dalam format yang mudah dibaca.
        """
        checkpoints = self.list_checkpoints()
        
        # Tampilkan hasil
        for category, files in checkpoints.items():
            if files:
                print(f"📦 {category.capitalize()} Checkpoints:")
                for f in files:
                    size_mb = f.stat().st_size / (1024 * 1024)
                    print(f"- {f.name} ({size_mb:.2f} MB, terakhir diubah: {datetime.fromtimestamp(f.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')})")
                print()
        
        # Tampilkan riwayat resume terakhir
        try:
            history = self.get_training_history()
            last_resume = history.get('last_resume')
            if last_resume:
                print(f"🔄 Resume terakhir: {last_resume['checkpoint']} pada {last_resume['timestamp']}")
        except:
            pass