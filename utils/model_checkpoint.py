# File: utils/model_checkpoint.py
# Author: Alfrida Sabar
# Deskripsi: Handler untuk menyimpan dan memuat model checkpoint
# dengan dukungan versioning dan pemilihan checkpoint terbaik

from typing import Dict, Optional
from pathlib import Path
import torch
import json
from datetime import datetime

from utils.logger import SmartCashLogger

class ModelCheckpoint:
    """
    Handler untuk manajemen model checkpoint dengan dukungan:
    - Multiple metrics monitoring
    - Top-K checkpoints saving
    - Versioning dan metadata
    """
    
    def __init__(
        self,
        dirpath: str,
        filename: str = 'checkpoint_{epoch:02d}_{val_loss:.3f}',
        monitor: str = 'val_loss',
        mode: str = 'min',
        save_top_k: int = 3,
        save_last: bool = True,
        logger: Optional[SmartCashLogger] = None
    ):
        self.dirpath = Path(dirpath)
        self.filename = filename
        self.monitor = monitor
        self.mode = mode
        self.save_top_k = save_top_k
        self.save_last = save_last
        self.logger = logger or SmartCashLogger(__name__)
        
        # Buat direktori
        self.dirpath.mkdir(parents=True, exist_ok=True)
        
        # Setup tracking
        self.best_score = float('inf') if mode == 'min' else float('-inf')
        self.checkpoints = []
        
        # Load metadata jika ada
        self.metadata_path = self.dirpath / 'checkpoint_metadata.json'
        self._load_metadata()
        
    def _load_metadata(self) -> None:
        """Load metadata checkpoint dari file"""
        if self.metadata_path.exists():
            with open(self.metadata_path, 'r') as f:
                self.metadata = json.load(f)
                
            self.logger.info(
                f"📂 Metadata checkpoint dimuat: {len(self.metadata)} entries"
            )
        else:
            self.metadata = {}
            
    def _save_metadata(self) -> None:
        """Simpan metadata checkpoint ke file"""
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
            
    def _format_filename(
        self,
        epoch: int,
        metrics: Dict[str, float]
    ) -> str:
        """Format nama file checkpoint dengan metrik"""
        formatted = self.filename
        # Replace epoch
        formatted = formatted.replace('{epoch:02d}', f'{epoch:02d}')
        
        # Replace metrics
        for k, v in metrics.items():
            placeholder = '{' + k + ':.3f}'
            if placeholder in formatted:
                formatted = formatted.replace(placeholder, f'{v:.3f}')
                
        return formatted + '.pt'
        
    def _is_better(self, current: float) -> bool:
        """Check apakah metrik saat ini lebih baik dari best"""
        if self.mode == 'min':
            return current < self.best_score
        return current > self.best_score
        
    def __call__(
        self,
        model: torch.nn.Module,
        epoch: int,
        metrics: Dict[str, float]
    ) -> None:
        """
        Save model checkpoint
        Args:
            model: Model PyTorch
            epoch: Epoch saat ini
            metrics: Dict metrik evaluasi
        """
        if self.monitor not in metrics:
            self.logger.warning(
                f"⚠️ Metrik {self.monitor} tidak ditemukan di {metrics.keys()}"
            )
            return
            
        # Format nama file
        filename = self._format_filename(epoch, metrics)
        filepath = self.dirpath / filename
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        torch.save(checkpoint, filepath)
        
        # Update metadata
        self.metadata[str(filepath)] = {
            'epoch': epoch,
            'metrics': metrics,
            'timestamp': checkpoint['timestamp']
        }
        
        # Track checkpoint
        self.checkpoints.append(str(filepath))
        
        # Update best score
        current = metrics[self.monitor]
        if self._is_better(current):
            self.best_score = current
            self.logger.success(
                f"✨ Model terbaik disimpan: {filename}\n"
                f"   {self.monitor}: {current:.4f}"
            )
        else:
            self.logger.info(f"💾 Checkpoint disimpan: {filename}")
            
        # Manage jumlah checkpoint
        self._cleanup_checkpoints()
        
        # Save metadata
        self._save_metadata()
        
    def _cleanup_checkpoints(self) -> None:
        """Cleanup checkpoint lama"""
        if self.save_top_k > 0 and len(self.checkpoints) > self.save_top_k:
            # Sort checkpoints berdasarkan metrik
            sorted_checkpoints = sorted(
                self.checkpoints,
                key=lambda x: self.metadata[x]['metrics'][self.monitor],
                reverse=self.mode == 'max'
            )
            
            # Remove checkpoint yang tidak diperlukan
            for checkpoint in sorted_checkpoints[self.save_top_k:]:
                try:
                    Path(checkpoint).unlink()
                    del self.metadata[checkpoint]
                    self.checkpoints.remove(checkpoint)
                    
                    self.logger.info(f"🗑️ Menghapus checkpoint: {checkpoint}")
                except Exception as e:
                    self.logger.warning(
                        f"⚠️ Gagal menghapus {checkpoint}: {str(e)}"
                    )
                    
    def get_best_checkpoint(self) -> Optional[str]:
        """Dapatkan path ke checkpoint terbaik"""
        if not self.checkpoints:
            return None
            
        return sorted(
            self.checkpoints,
            key=lambda x: self.metadata[x]['metrics'][self.monitor],
            reverse=self.mode == 'max'
        )[0]
        
    def load_checkpoint(
        self,
        model: torch.nn.Module,
        checkpoint_path: Optional[str] = None
    ) -> Dict:
        """
        Load model checkpoint
        Args:
            model: Model PyTorch
            checkpoint_path: Path spesifik atau None untuk load yang terbaik
        Returns:
            Dict checkpoint data
        """
        if checkpoint_path is None:
            checkpoint_path = self.get_best_checkpoint()
            
        if checkpoint_path is None:
            raise ValueError("Tidak ada checkpoint yang tersedia")
            
        self.logger.info(f"📂 Loading checkpoint: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return checkpoint