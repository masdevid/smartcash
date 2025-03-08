# File: smartcash/handlers/checkpoint/checkpoint_finder.py
# Author: Alfrida Sabar
# Deskripsi: Pencarian checkpoint model berdasarkan berbagai kriteria

import os
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from smartcash.utils.logger import SmartCashLogger
from smartcash.handlers.checkpoint.checkpoint_history import CheckpointHistory

class CheckpointFinder:
    """
    Pencarian checkpoint model berdasarkan berbagai kriteria.
    """
    
    def __init__(
        self,
        output_dir: Path,
        logger: Optional[SmartCashLogger] = None
    ):
        """
        Inisialisasi pencari checkpoint.
        
        Args:
            output_dir: Direktori yang berisi checkpoint
            logger: Logger kustom (opsional)
        """
        self.output_dir = output_dir
        self.logger = logger or SmartCashLogger(__name__)
        self.history = CheckpointHistory(output_dir, logger)
    
    def find_best_checkpoint(self) -> Optional[str]:
        """
        Temukan checkpoint terbaik berdasarkan riwayat training.
        
        Returns:
            Path ke checkpoint terbaik, atau None jika tidak ada
        """
        try:
            history = self.history.get_history()
            
            # Urutkan berdasarkan metrik yang relevan (is_best flag)
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
                    self.logger.info(f"🌟 Checkpoint terbaik ditemukan: {best_checkpoint_path}")
                    return str(best_checkpoint_path)
                else:
                    self.logger.warning(f"⚠️ File checkpoint terbaik tidak ditemukan: {best_checkpoint_path}")
            
            # Jika tidak ada checkpoint terbaik valid, cari checkpoint terakhir
            latest_checkpoint = self.find_latest_checkpoint()
            if latest_checkpoint:
                self.logger.info(f"ℹ️ Menggunakan checkpoint terakhir sebagai alternatif: {latest_checkpoint}")
                return latest_checkpoint
                
            # Tidak ada checkpoint yang ditemukan
            self.logger.warning("⚠️ Tidak ditemukan checkpoint terbaik")
            return None
            
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
                self.logger.warning("⚠️ Tidak ditemukan file checkpoint")
                return None
            
            # Urutkan berdasarkan waktu modifikasi (terbaru dulu)
            latest_checkpoint = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
            
            self.logger.info(f"🕒 Checkpoint terakhir ditemukan: {latest_checkpoint}")
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
                self.logger.warning(f"⚠️ Tidak ditemukan checkpoint untuk epoch {epoch}")
                return None
            
            # Jika ada beberapa, ambil yang paling baru
            if len(epoch_files) > 1:
                epoch_checkpoint = max(epoch_files, key=lambda x: x.stat().st_mtime)
            else:
                epoch_checkpoint = epoch_files[0]
            
            self.logger.info(f"🔢 Checkpoint untuk epoch {epoch} ditemukan: {epoch_checkpoint}")
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
            
            # Log jumlah checkpoint yang ditemukan
            total_checkpoints = sum(len(files) for files in checkpoints.values())
            self.logger.info(f"🔍 Ditemukan {total_checkpoints} checkpoint dalam {len(checkpoints)} kategori")
            
            return checkpoints
            
        except Exception as e:
            self.logger.error(f"❌ Gagal mendapatkan daftar checkpoint: {str(e)}")
            return {'best': [], 'latest': [], 'epoch': [], 'emergency': []}
    
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
        try:
            # Dapatkan semua checkpoint
            all_checkpoints = []
            for files in self.list_checkpoints().values():
                all_checkpoints.extend(files)
            
            # Filter berdasarkan kriteria
            filtered_checkpoints = all_checkpoints
            
            if backbone:
                filtered_checkpoints = [
                    cp for cp in filtered_checkpoints
                    if backbone in cp.name
                ]
                
            if dataset:
                filtered_checkpoints = [
                    cp for cp in filtered_checkpoints
                    if dataset in cp.name
                ]
                
            if min_epoch is not None:
                # Extract epoch dari nama file
                def get_epoch(checkpoint_path: Path) -> Optional[int]:
                    import re
                    match = re.search(r'_epoch_(\d+)_', checkpoint_path.name)
                    if match:
                        return int(match.group(1))
                    return None
                
                filtered_checkpoints = [
                    cp for cp in filtered_checkpoints
                    if get_epoch(cp) is not None and get_epoch(cp) >= min_epoch
                ]
            
            self.logger.info(f"🔍 Ditemukan {len(filtered_checkpoints)} checkpoint yang sesuai kriteria")
            return filtered_checkpoints
            
        except Exception as e:
            self.logger.error(f"❌ Gagal memfilter checkpoint: {str(e)}")
            return []