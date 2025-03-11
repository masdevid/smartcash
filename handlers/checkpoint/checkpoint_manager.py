# File: smartcash/handlers/checkpoint/checkpoint_manager.py
# Author: Alfrida Sabar
# Deskripsi: Manager utama untuk pengelolaan checkpoint model dengan API terpadu (Diringkas)

import torch
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

from smartcash.utils.logger import get_logger
from smartcash.handlers.checkpoint.checkpoint_loader import CheckpointLoader
from smartcash.handlers.checkpoint.checkpoint_saver import CheckpointSaver
from smartcash.handlers.checkpoint.checkpoint_finder import CheckpointFinder
from smartcash.handlers.checkpoint.checkpoint_history import CheckpointHistory

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
        logger = None
    ):
        """
        Inisialisasi CheckpointManager
        
        Args:
            output_dir: Direktori untuk menyimpan checkpoint
            logger: Logger kustom (opsional)
        """
        self.output_dir = Path(output_dir)
        self.logger = logger or get_logger("checkpoint_manager")
        
        # Buat direktori jika belum ada
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Inisialisasi komponen
        self.loader = CheckpointLoader(self.output_dir, self.logger)
        self.saver = CheckpointSaver(self.output_dir, self.logger)
        self.finder = CheckpointFinder(self.output_dir, self.logger)
        self.history = CheckpointHistory(self.output_dir, self.logger)
        
    
    # API untuk loading checkpoint
    def load_checkpoint(self, checkpoint_path=None, device=None, model=None, optimizer=None, scheduler=None):
        """Muat checkpoint model"""
        return self.loader.load_checkpoint(checkpoint_path, device, model, optimizer, scheduler)
    
    # API untuk penyimpanan checkpoint
    def save_checkpoint(self, model, optimizer, scheduler, config, epoch, metrics, is_best=False, save_optimizer=True):
        """Simpan checkpoint model"""
        return self.saver.save_checkpoint(model, optimizer, scheduler, config, epoch, metrics, is_best, save_optimizer)
    
    # API untuk pencarian checkpoint
    def find_best_checkpoint(self):
        """Temukan checkpoint terbaik"""
        return self.finder.find_best_checkpoint()
    
    def find_latest_checkpoint(self):
        """Temukan checkpoint terakhir"""
        return self.finder.find_latest_checkpoint()
    
    def find_checkpoint_by_epoch(self, epoch):
        """Temukan checkpoint untuk epoch tertentu"""
        return self.finder.find_checkpoint_by_epoch(epoch)
    
    def list_checkpoints(self):
        """Dapatkan daftar checkpoint yang tersedia"""
        return self.finder.list_checkpoints()
    
    # API untuk pengelolaan history
    def get_training_history(self):
        """Dapatkan riwayat training"""
        return self.history.get_history()
    
    # Metode utilitas
    def cleanup_checkpoints(self, max_checkpoints=5, keep_best=True, keep_latest=True, max_epochs=5):
        """Bersihkan checkpoint lama"""
        return self.saver.cleanup_checkpoints(max_checkpoints, keep_best, keep_latest, max_epochs)
    
    def copy_to_drive(self, drive_dir='/content/drive/MyDrive/SmartCash/checkpoints', best_only=False):
        """Salin checkpoint ke Google Drive"""
        return self.saver.copy_to_drive(drive_dir, best_only)