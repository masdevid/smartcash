# File: smartcash/handlers/checkpoint/checkpoint_history.py
# Author: Alfrida Sabar
# Deskripsi: Pengelolaan riwayat checkpoint training

import yaml
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

from smartcash.utils.logger import SmartCashLogger

class CheckpointHistory:
    """
    Pengelolaan riwayat checkpoint dan training.
    """
    
    def __init__(
        self,
        output_dir: Path,
        logger: Optional[SmartCashLogger] = None
    ):
        """
        Inisialisasi pengelolaan riwayat checkpoint.
        
        Args:
            output_dir: Direktori untuk menyimpan riwayat
            logger: Logger kustom (opsional)
        """
        self.output_dir = output_dir
        self.logger = logger or SmartCashLogger(__name__)
        self.history_file = self.output_dir / 'training_history.yaml'
        
        # Inisialisasi file riwayat jika belum ada
        self._initialize_history_file()
    
    def _initialize_history_file(self) -> None:
        """Inisialisasi file riwayat jika belum ada."""
        try:
            if not self.history_file.exists():
                with open(self.history_file, 'w') as f:
                    yaml.safe_dump({
                        'total_runs': 0,
                        'runs': [],
                        'last_resume': None
                    }, f)
                self.logger.info(f"📝 File riwayat training diinisialisasi: {self.history_file}")
        except Exception as e:
            self.logger.error(f"❌ Gagal menginisialisasi riwayat training: {str(e)}")
            # Buat file kosong jika gagal untuk mencegah error berikutnya
            try:
                with open(self.history_file, 'w') as f:
                    f.write('{}')
            except:
                pass
    
    def update_training_history(
        self, 
        checkpoint_name: str, 
        metrics: Dict[str, Any], 
        is_best: bool,
        epoch: int
    ) -> None:
        """
        Update riwayat training dalam file YAML.
        
        Args:
            checkpoint_name: Nama file checkpoint
            metrics: Metrik training
            is_best: Apakah checkpoint terbaik
            epoch: Nomor epoch
        """
        try:
            with open(self.history_file, 'r') as f:
                history = yaml.safe_load(f) or {'total_runs': 0, 'runs': [], 'last_resume': None}
            
            history['total_runs'] += 1
            history['runs'].append({
                'checkpoint_name': checkpoint_name,
                'timestamp': datetime.now().isoformat(),
                'is_best': is_best,
                'epoch': epoch,
                'metrics': metrics
            })
            
            # Batasi jumlah riwayat
            if len(history['runs']) > 50:
                history['runs'] = history['runs'][-50:]
            
            with open(self.history_file, 'w') as f:
                yaml.safe_dump(history, f)
                
            self.logger.debug(f"📝 Riwayat training diperbarui: {checkpoint_name}")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Gagal memperbarui riwayat training: {str(e)}")
    
    def update_resume_history(self, checkpoint_path: str) -> None:
        """
        Update riwayat resume training.
        
        Args:
            checkpoint_path: Path checkpoint yang digunakan untuk resume
        """
        try:
            with open(self.history_file, 'r') as f:
                history = yaml.safe_load(f) or {'total_runs': 0, 'runs': [], 'last_resume': None}
            
            history['last_resume'] = {
                'checkpoint': Path(checkpoint_path).name,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(self.history_file, 'w') as f:
                yaml.safe_dump(history, f)
                
            self.logger.info(f"🔄 Riwayat resume training diperbarui: {Path(checkpoint_path).name}")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Gagal memperbarui riwayat resume: {str(e)}")
    
    def get_history(self) -> Dict[str, Any]:
        """
        Ambil seluruh riwayat training.
        
        Returns:
            Dict riwayat training
        """
        try:
            with open(self.history_file, 'r') as f:
                history = yaml.safe_load(f) or {'total_runs': 0, 'runs': [], 'last_resume': None}
            return history
        except Exception as e:
            self.logger.warning(f"⚠️ Gagal membaca riwayat training: {str(e)}")
            return {'total_runs': 0, 'runs': [], 'last_resume': None}
    
    def export_to_json(self, output_path: Optional[str] = None) -> str:
        """
        Export riwayat training ke JSON.
        
        Args:
            output_path: Path output (opsional)
            
        Returns:
            Path file JSON
        """
        if output_path is None:
            output_path = str(self.output_dir / 'training_history.json')
            
        try:
            history = self.get_history()
            
            with open(output_path, 'w') as f:
                json.dump(history, f, indent=2)
                
            self.logger.info(f"📊 Riwayat training diekspor ke JSON: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"❌ Gagal mengekspor riwayat training: {str(e)}")
            return ""