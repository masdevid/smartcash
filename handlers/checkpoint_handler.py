# File: smartcash/utils/checkpoint_handler.py
# Author: Alfrida Sabar
# Deskripsi: Utilitas komprehensif untuk mengelola checkpoint model dengan dukungan berbagai operasi dan failsafe

import os
import torch
import shutil
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Callable

from smartcash.utils.logger import SmartCashLogger

class CheckpointHandler:
    """
    Manajemen checkpoint model yang komprehensif untuk SmartCash.
    
    Fitur:
    - Penyimpanan checkpoint dengan nama yang terstruktur
    - Pencarian, pembersihan, dan pemulihan checkpoint
    - Dukungan untuk berbagai jenis checkpoint
    - Logging dan pelacakan riwayat training
    """
    
    def __init__(
        self, 
        checkpoints_dir: str = 'runs/train/weights', 
        logger: Optional[SmartCashLogger] = None
    ):
        """
        Inisialisasi CheckpointHandler
        
        Args:
            checkpoints_dir: Direktori untuk menyimpan checkpoint
            logger: Logger kustom (opsional)
        """
        self.checkpoints_dir = Path(checkpoints_dir)
        self.history_file = self.checkpoints_dir / 'training_history.yaml'
        
        # Buat direktori jika belum ada
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logger
        self.logger = logger or SmartCashLogger(__name__)
        
        # Inisialisasi riwayat training
        self._init_training_history()
    
    def _init_training_history(self):
        """Inisialisasi file riwayat training jika belum ada"""
        if not self.history_file.exists():
            with open(self.history_file, 'w') as f:
                yaml.safe_dump({
                    'total_runs': 0,
                    'runs': []
                }, f)
    
    def generate_checkpoint_name(
        self, 
        config: Dict, 
        run_type: str = 'default'
    ) -> str:
        """
        Generate nama checkpoint dengan struktur yang konsisten
        
        Args:
            config: Konfigurasi model/training
            run_type: Tipe run (default, best, latest)
        
        Returns:
            String nama checkpoint unik
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Ekstrak informasi konfigurasi
        backbone = config.get('model', {}).get('backbone', 'default')
        dataset = config.get('data', {}).get('source', 'default')
        
        return f"smartcash_{backbone}_{dataset}_{run_type}_{timestamp}.pth"
    
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        config: Dict,
        epoch: int,
        metrics: Dict,
        is_best: bool = False
    ) -> Dict[str, str]:
        """
        Simpan checkpoint model dengan metadata komprehensif
        
        Args:
            model: Model PyTorch
            config: Konfigurasi training
            epoch: Epoch saat ini
            metrics: Metrik training
            is_best: Apakah ini model terbaik
        
        Returns:
            Dict berisi path checkpoint yang disimpan
        """
        try:
            # Pastikan direktori checkpoint ada
            self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate nama file checkpoint
            run_type = 'best' if is_best else 'latest'
            checkpoint_name = self.generate_checkpoint_name(config, run_type)
            checkpoint_path = self.checkpoints_dir / checkpoint_name
            
            # Prepare checkpoint data
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'metrics': metrics,
                'config': config,
                'timestamp': datetime.now().isoformat()
            }
            
            # Simpan checkpoint
            torch.save(checkpoint, checkpoint_path)
            
            # Update riwayat training
            self._update_training_history(checkpoint_name, metrics, is_best)
            
            self.logger.success(f"💾 Checkpoint disimpan: {checkpoint_name}")
            
            return {
                'path': str(checkpoint_path),
                'type': run_type,
                'size': f"{checkpoint_path.stat().st_size / (1024*1024):.2f} MB"
            }
        
        except Exception as e:
            self.logger.error(f"❌ Gagal menyimpan checkpoint: {str(e)}")
            raise
    
    def _update_training_history(
        self, 
        checkpoint_name: str, 
        metrics: Dict, 
        is_best: bool
    ):
        """
        Update riwayat training dalam file YAML
        
        Args:
            checkpoint_name: Nama file checkpoint
            metrics: Metrik training
            is_best: Apakah checkpoint terbaik
        """
        try:
            with open(self.history_file, 'r') as f:
                history = yaml.safe_load(f) or {'total_runs': 0, 'runs': []}
            
            history['total_runs'] += 1
            history['runs'].append({
                'checkpoint_name': checkpoint_name,
                'timestamp': datetime.now().isoformat(),
                'is_best': is_best,
                'metrics': metrics
            })
            
            # Batasi jumlah riwayat
            if len(history['runs']) > 50:
                history['runs'] = history['runs'][-50:]
            
            with open(self.history_file, 'w') as f:
                yaml.safe_dump(history, f)
        
        except Exception as e:
            self.logger.warning(f"⚠️ Gagal memperbarui riwayat training: {str(e)}")
    
    def load_checkpoint(
        self, 
        checkpoint_path: Optional[str] = None,
        device: Optional[torch.device] = None
    ) -> Dict:
        """
        Muat checkpoint dengan dukungan berbagai opsi
        
        Args:
            checkpoint_path: Path ke checkpoint (jika None, akan mengambil checkpoint terbaik)
            device: Perangkat untuk memuat model
        
        Returns:
            Dict berisi metadata dan state_dict
        """
        device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        try:
            # Cari checkpoint terbaik jika path tidak diberikan
            if checkpoint_path is None:
                checkpoint_path = self.find_best_checkpoint()
            
            # Muat checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            self.logger.success(f"📂 Checkpoint dimuat dari: {checkpoint_path}")
            return checkpoint
        
        except Exception as e:
            self.logger.error(f"❌ Gagal memuat checkpoint: {str(e)}")
            raise
    
    def find_best_checkpoint(self) -> Optional[str]:
        """
        Temukan checkpoint terbaik berdasarkan riwayat training
        
        Returns:
            Path ke checkpoint terbaik, atau None jika tidak ada
        """
        try:
            with open(self.history_file, 'r') as f:
                history = yaml.safe_load(f) or {'runs': []}
            
            # Urutkan berdasarkan metrik yang relevan (misalnya val_loss)
            best_runs = [
                run for run in history['runs'] 
                if run.get('is_best', False)
            ]
            
            if best_runs:
                # Ambil checkpoint terbaik terakhir
                best_run = max(best_runs, key=lambda x: x.get('timestamp', ''))
                best_checkpoint_name = best_run['checkpoint_name']
                best_checkpoint_path = self.checkpoints_dir / best_checkpoint_name
                
                return str(best_checkpoint_path)
            
            # Jika tidak ada checkpoint terbaik, ambil checkpoint terakhir
            if history['runs']:
                last_run = history['runs'][-1]
                last_checkpoint_name = last_run['checkpoint_name']
                last_checkpoint_path = self.checkpoints_dir / last_checkpoint_name
                
                return str(last_checkpoint_path)
            
            return None
        
        except Exception as e:
            self.logger.error(f"❌ Gagal mencari checkpoint terbaik: {str(e)}")
            return None
    
    def delete_old_checkpoints(
        self, 
        max_checkpoints: int = 10, 
        keep_best: bool = True
    ) -> List[str]:
        """
        Hapus checkpoint lama
        
        Args:
            max_checkpoints: Jumlah maksimal checkpoint yang disimpan
            keep_best: Pertahankan checkpoint terbaik
        
        Returns:
            List checkpoint yang dihapus
        """
        try:
            # Dapatkan semua checkpoint
            checkpoints = sorted(
                self.checkpoints_dir.glob('*.pth'),
                key=os.path.getctime
            )
            
            # Ambil checkpoint terbaik untuk dikecualikan
            best_checkpoint = None
            if keep_best:
                best_checkpoint_path = self.find_best_checkpoint()
                if best_checkpoint_path:
                    best_checkpoint = Path(best_checkpoint_path)
            
            # Tentukan checkpoint yang akan dihapus
            to_delete = checkpoints[:-max_checkpoints]
            if best_checkpoint and best_checkpoint in to_delete:
                to_delete.remove(best_checkpoint)
            
            # Hapus checkpoint
            deleted_paths = []
            for checkpoint in to_delete:
                try:
                    checkpoint.unlink()
                    deleted_paths.append(str(checkpoint))
                    self.logger.info(f"🗑️ Checkpoint dihapus: {checkpoint.name}")
                except Exception as e:
                    self.logger.warning(f"⚠️ Gagal menghapus {checkpoint.name}: {str(e)}")
            
            return deleted_paths
        
        except Exception as e:
            self.logger.error(f"❌ Gagal menghapus checkpoint lama: {str(e)}")
            return []
    
    def copy_to_drive(
        self, 
        drive_dir: str = '/content/drive/MyDrive/SmartCash/checkpoints'
    ) -> List[str]:
        """
        Salin checkpoint ke Google Drive
        
        Args:
            drive_dir: Direktori tujuan di Google Drive
        
        Returns:
            List path checkpoint yang disalin
        """
        try:
            # Buat direktori jika belum ada
            os.makedirs(drive_dir, exist_ok=True)
            
            # Cari checkpoint
            checkpoint_paths = list(self.checkpoints_dir.glob('*.pth'))
            
            # Salin checkpoint
            copied_paths = []
            for checkpoint in checkpoint_paths:
                dest_path = os.path.join(drive_dir, checkpoint.name)
                shutil.copy2(checkpoint, dest_path)
                copied_paths.append(dest_path)
                self.logger.info(f"📂 Checkpoint disalin: {checkpoint.name}")
            
            return copied_paths
        
        except Exception as e:
            self.logger.error(f"❌ Gagal menyalin checkpoint ke Drive: {str(e)}")
            return []