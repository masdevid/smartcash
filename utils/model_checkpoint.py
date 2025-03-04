# File: smartcash/utils/model_checkpoint.py
# Author: Alfrida Sabar
# Deskripsi: Utilitas untuk menyimpan model checkpoint dengan perbaikan masalah pickle

import os
import torch
from typing import Optional, Dict, Any, Callable
from datetime import datetime
from pathlib import Path

class ModelCheckpoint:
    """Model checkpoint handler dengan perbaikan masalah pickle."""
    
    def __init__(
        self,
        save_dir: str = 'checkpoints',
        logger = None
    ):
        """
        Inisialisasi Model Checkpoint handler.
        
        Args:
            save_dir: Direktori untuk menyimpan checkpoint
            logger: Logger (optional, tidak disimpan untuk menghindari pickle)
        """
        self.save_dir = Path(save_dir)
        self.best_loss = float('inf')
        
        # Gunakan callable untuk logging alih-alih menyimpan objek logger
        if logger:
            self._log_info = lambda msg: logger.info(msg)
            self._log_success = lambda msg: logger.success(msg)
            self._log_error = lambda msg: logger.error(msg)
        else:
            # Default to print jika tidak ada logger
            self._log_info = lambda msg: print(f"ℹ️ {msg}")
            self._log_success = lambda msg: print(f"✅ {msg}")
            self._log_error = lambda msg: print(f"❌ {msg}")
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
    
    def _get_model_name(self, config: Dict) -> str:
        """Generate a unique model name based on configuration."""
        # Get detection mode
        if isinstance(config.get('layers', []), list):
            mode = 'multilayer' if len(config.get('layers', [])) > 1 else 'single'
        else:
            mode = 'single'
            
        # Get backbone type
        backbone = config.get('backbone', 'default')
        
        # Get data source
        data_source = config.get('data_source', 'default')
        
        # Get timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Construct name
        name = f"smartcash_{mode}_{backbone}_{data_source}_{timestamp}"
        
        return name
    
    def save(
        self,
        model: torch.nn.Module,
        config: Dict,
        epoch: int,
        loss: float,
        is_best: bool = False
    ) -> None:
        """
        Save model checkpoint secara aman tanpa masalah pickle.
        
        Args:
            model: Model yang akan disimpan
            config: Konfigurasi training
            epoch: Epoch saat ini
            loss: Nilai loss saat ini
            is_best: Apakah ini model terbaik sejauh ini
        """
        try:
            # Generate model name
            model_name = self._get_model_name(config)
            
            # Save model state
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'loss': loss,
                'config': config
            }
            
            # Save latest checkpoint
            latest_path = os.path.join(self.save_dir, f"{model_name}_latest.pth")
            torch.save(checkpoint, latest_path)
            self._log_info(f"💾 Saved latest checkpoint: {latest_path}")
            
            # Save epoch checkpoint
            epoch_path = os.path.join(self.save_dir, f"{model_name}_epoch_{epoch}.pth")
            torch.save(checkpoint, epoch_path)
            self._log_info(f"💾 Saved epoch checkpoint: {epoch_path}")
            
            # Update and save best model if needed
            if loss < self.best_loss:
                self.best_loss = loss
                best_path = os.path.join(self.save_dir, f"{model_name}_best.pth")
                torch.save(checkpoint, best_path)
                self._log_success(f"🏆 Saved best model: {best_path}")
                
        except Exception as e:
            self._log_error(f"❌ Failed to save checkpoint: {str(e)}")
            raise
    
    def load(self, model_path: str) -> Dict:
        """
        Load model checkpoint.
        
        Args:
            model_path: Path ke file checkpoint
            
        Returns:
            Dictionary containing model state and metadata
        """
        try:
            checkpoint = torch.load(model_path)
            self._log_info(f"📂 Loaded checkpoint: {model_path}")
            return checkpoint
            
        except Exception as e:
            self._log_error(f"❌ Failed to load checkpoint: {str(e)}")
            raise

class StatelessCheckpointSaver:
    """Fungsi untuk menyimpan checkpoint tanpa state/class untuk menghindari masalah pickle."""
    
    @staticmethod
    def save_checkpoint(
        model: torch.nn.Module,
        config: Dict,
        epoch: int,
        loss: float,
        checkpoint_dir: str,
        is_best: bool = False,
        log_fn: Optional[Callable] = None
    ) -> Dict[str, str]:
        """
        Simpan checkpoint model secara stateless (tanpa referensi ke self).
        
        Args:
            model: Model untuk disimpan
            config: Konfigurasi training
            epoch: Epoch saat ini
            loss: Nilai loss
            checkpoint_dir: Direktori untuk menyimpan checkpoint
            is_best: Apakah ini model terbaik
            log_fn: Fungsi logging opsional
            
        Returns:
            Dict dengan path checkpoint yang disimpan
        """
        # Gunakan print jika tidak ada log_fn
        log = log_fn or print
        
        # Buat direktori jika belum ada
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Generate timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Buat nama file dasar
        if 'backbone' in config and 'data_source' in config:
            base_name = f"smartcash_{config['backbone']}_{config['data_source']}"
        else:
            base_name = f"smartcash_model_{timestamp}"
        
        # Siapkan data checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'loss': loss,
            'config': config
        }
        
        # Simpan epoch checkpoint
        epoch_path = os.path.join(checkpoint_dir, f"{base_name}_epoch_{epoch}.pth")
        torch.save(checkpoint, epoch_path)
        log(f"💾 Saved epoch checkpoint: {epoch_path}")
        
        # Simpan latest
        latest_path = os.path.join(checkpoint_dir, f"{base_name}_latest.pth")
        torch.save(checkpoint, latest_path)
        
        # Simpan best jika perlu
        best_path = None
        if is_best:
            best_path = os.path.join(checkpoint_dir, f"{base_name}_best.pth")
            torch.save(checkpoint, best_path)
            log(f"🏆 Saved best model: {best_path}")
        
        return {
            'epoch': epoch_path,
            'latest': latest_path,
            'best': best_path
        }