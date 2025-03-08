# File: smartcash/handlers/checkpoint/checkpoint_loader.py
# Author: Alfrida Sabar
# Deskripsi: Komponen untuk loading model checkpoint

import torch
import traceback
from datetime import datetime
from typing import Dict, Optional, Tuple, Any, Union
from pathlib import Path

from smartcash.utils.logger import SmartCashLogger
from smartcash.handlers.checkpoint.checkpoint_finder import CheckpointFinder
from smartcash.handlers.checkpoint.checkpoint_history import CheckpointHistory

class CheckpointLoader:
    """
    Komponen untuk loading model checkpoint dengan penanganan error yang robust.
    """
    
    def __init__(
        self,
        output_dir: Path,
        logger: Optional[SmartCashLogger] = None
    ):
        """
        Inisialisasi loader.
        
        Args:
            output_dir: Direktori yang berisi checkpoint
            logger: Logger kustom (opsional)
        """
        self.output_dir = output_dir
        self.logger = logger or SmartCashLogger(__name__)
        
        # Inisialisasi finder dan history
        self.finder = CheckpointFinder(output_dir, logger)
        self.history = CheckpointHistory(output_dir, logger)
    
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
        device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        try:
            # Cari checkpoint terbaik jika path tidak diberikan
            if checkpoint_path is None:
                checkpoint_path = self.finder.find_best_checkpoint()
                if checkpoint_path is None:
                    self.logger.warning("⚠️ Tidak ditemukan checkpoint terbaik")
                    
                    # Cari checkpoint latest sebagai fallback
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
                    self.logger.error(f"❌ Gagal memuat state model: {str(e)}")
                    # Coba load dengan strict=False
                    self.logger.warning("⚠️ Mencoba memuat dengan strict=False...")
                    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                    self.logger.warning("✅ Model dimuat dengan strict=False (beberapa layer mungkin tidak termuat)")
            
            # Muat state ke optimizer jika disediakan
            if optimizer is not None and 'optimizer_state_dict' in checkpoint:
                try:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    self.logger.info(f"✅ Optimizer state berhasil dimuat")
                except Exception as e:
                    self.logger.warning(f"⚠️ Gagal memuat state optimizer: {str(e)}")
            
            # Muat state ke scheduler jika disediakan
            if scheduler is not None and 'scheduler_state_dict' in checkpoint:
                try:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    self.logger.info(f"✅ Scheduler state berhasil dimuat")
                except Exception as e:
                    self.logger.warning(f"⚠️ Gagal memuat state scheduler: {str(e)}")
            
            # Update riwayat resume
            self.history.update_resume_history(checkpoint_path)
            
            self.logger.success(f"📂 Checkpoint berhasil dimuat dari: {checkpoint_path}")
            
            # Tambahkan metadata berguna ke hasil
            checkpoint['loaded_from'] = checkpoint_path
            checkpoint['load_time'] = datetime.now().isoformat()
            
            if 'epoch' not in checkpoint:
                checkpoint['epoch'] = 0
            
            if 'metrics' not in checkpoint:
                checkpoint['metrics'] = {}
                
            return checkpoint
        
        except Exception as e:
            self.logger.error(f"❌ Gagal memuat checkpoint: {str(e)}")
            self.logger.error(f"📋 Detail stack trace: {traceback.format_exc()}")
            
            # Return default state yang tidak akan menyebabkan crash
            return {
                'epoch': 0,
                'model_state_dict': None,
                'optimizer_state_dict': None,
                'scheduler_state_dict': None,
                'metrics': {},
                'config': {},
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
    
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
        device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        try:
            # Cari checkpoint terbaik jika path tidak diberikan
            if checkpoint_path is None:
                checkpoint_path = self.finder.find_best_checkpoint()
                if checkpoint_path is None:
                    raise FileNotFoundError("Tidak ditemukan checkpoint yang valid")
            
            # Muat checkpoint tanpa menyalin state ke model
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            self.logger.info(f"📂 Checkpoint metadata dimuat dari: {checkpoint_path}")
            
            # Tambahkan metadata berguna
            checkpoint['loaded_from'] = checkpoint_path
            checkpoint['load_time'] = datetime.now().isoformat()
            
            return checkpoint
        
        except Exception as e:
            self.logger.error(f"❌ Gagal memuat checkpoint metadata: {str(e)}")
            return {
                'epoch': 0,
                'metrics': {},
                'config': {},
                'error': str(e)
            }
    
    def validate_checkpoint(self, checkpoint_path: str) -> Tuple[bool, Dict]:
        """
        Validasi apakah checkpoint valid dan lengkap.
        
        Args:
            checkpoint_path: Path ke checkpoint
            
        Returns:
            Tuple (is_valid, message_dict)
        """
        try:
            # Coba memuat checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Cek komponen kunci
            required_keys = ['model_state_dict', 'epoch', 'config']
            missing_keys = [key for key in required_keys if key not in checkpoint]
            
            if missing_keys:
                return False, {
                    'valid': False,
                    'missing_keys': missing_keys,
                    'message': f"Checkpoint tidak valid: Key {missing_keys} tidak ditemukan"
                }
            
            # Cek struktur state_dict
            if not isinstance(checkpoint['model_state_dict'], dict) or len(checkpoint['model_state_dict']) == 0:
                return False, {
                    'valid': False,
                    'message': "Checkpoint tidak valid: model_state_dict kosong atau bukan dictionary"
                }
                
            # Cek data konfigurasi tambahan
            if not isinstance(checkpoint.get('config', {}), dict):
                return False, {
                    'valid': False,
                    'message': "Checkpoint tidak valid: config bukan dictionary"
                }
            
            # Checkpoint valid
            return True, {
                'valid': True,
                'epoch': checkpoint.get('epoch', 0),
                'timestamp': checkpoint.get('timestamp', ''),
                'message': "Checkpoint valid dan lengkap"
            }
            
        except Exception as e:
            return False, {
                'valid': False,
                'message': f"Gagal memvalidasi checkpoint: {str(e)}",
                'error': str(e)
            }
    
    def extract_metadata(self, checkpoint_path: str) -> Dict:
        """
        Ekstrak metadata dari checkpoint tanpa memuat model state.
        
        Args:
            checkpoint_path: Path ke checkpoint
            
        Returns:
            Dict metadata
        """
        try:
            # Gunakan torch load dengan map_location='cpu'
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Ekstrak metadata tanpa state_dict dan optimizer_state
            metadata = {
                'epoch': checkpoint.get('epoch', 0),
                'metrics': checkpoint.get('metrics', {}),
                'config': checkpoint.get('config', {}),
                'timestamp': checkpoint.get('timestamp', ''),
                'file_path': checkpoint_path,
                'file_size_mb': Path(checkpoint_path).stat().st_size / (1024 * 1024)
            }
            
            # Jika model_state_dict ada, tambahkan info tentang layer
            if 'model_state_dict' in checkpoint and isinstance(checkpoint['model_state_dict'], dict):
                metadata['layers'] = list(checkpoint['model_state_dict'].keys())
                metadata['num_layers'] = len(metadata['layers'])
            
            return metadata
            
        except Exception as e:
            self.logger.error(f"❌ Gagal mengekstrak metadata: {str(e)}")
            return {
                'error': str(e),
                'file_path': checkpoint_path
            }