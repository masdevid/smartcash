# File: smartcash/handlers/evaluation/integration/checkpoint_manager_adapter.py
# Author: Alfrida Sabar
# Deskripsi: Adapter untuk pengelolaan checkpoint model

import os
import glob
import torch
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import re

from smartcash.utils.logger import SmartCashLogger, get_logger

class CheckpointManagerAdapter:
    """
    Adapter untuk CheckpointManager.
    Menyediakan antarmuka untuk pencarian dan validasi checkpoint model.
    """
    
    def __init__(
        self, 
        config: Dict,
        logger: Optional[SmartCashLogger] = None
    ):
        """
        Inisialisasi CheckpointManagerAdapter.
        
        Args:
            config: Konfigurasi untuk evaluasi
            logger: Logger kustom (opsional)
        """
        self.config = config
        self.logger = logger or get_logger("checkpoint_adapter")
        
        # Setup direktori checkpoint
        self.checkpoints_dir = Path(self.config.get('checkpoints_dir', 'runs/train/weights'))
        
        # Buat direktori jika belum ada
        if not self.checkpoints_dir.exists():
            self.logger.warning(f"⚠️ Direktori checkpoint tidak ditemukan: {self.checkpoints_dir}")
            self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"✅ Direktori checkpoint dibuat: {self.checkpoints_dir}")
        
        self.logger.debug(f"🔧 CheckpointManagerAdapter diinisialisasi (dir={self.checkpoints_dir})")
    
    def get_latest_checkpoint(self, pattern: str = None) -> str:
        """
        Dapatkan checkpoint terbaru.
        
        Args:
            pattern: Pola glob untuk memfilter file (opsional)
            
        Returns:
            Path ke checkpoint terbaru
            
        Raises:
            FileNotFoundError: Jika tidak ada checkpoint yang ditemukan
        """
        # Tentukan pola pencarian
        search_pattern = pattern or "*.pt"
        
        # Dapatkan semua checkpoint
        checkpoints = list(self.checkpoints_dir.glob(search_pattern))
        
        if not checkpoints:
            raise FileNotFoundError(f"❌ Tidak ada checkpoint yang ditemukan dengan pola: {search_pattern}")
        
        # Ambil yang terbaru berdasarkan waktu modifikasi
        latest = max(checkpoints, key=os.path.getmtime)
        
        self.logger.info(f"🔍 Checkpoint terbaru: {latest}")
        return str(latest)
    
    def get_best_checkpoint(self, metric: str = 'mAP') -> str:
        """
        Dapatkan checkpoint terbaik berdasarkan metrik.
        
        Args:
            metric: Metrik untuk pemilihan ('mAP', 'f1', dll)
            
        Returns:
            Path ke checkpoint terbaik
            
        Raises:
            FileNotFoundError: Jika tidak ada checkpoint best yang ditemukan
        """
        # Cari file *_best.pt
        best_checkpoints = list(self.checkpoints_dir.glob("*_best.pt"))
        
        if best_checkpoints:
            # Jika ada file dengan suffix _best, gunakan yang terbaru
            best = max(best_checkpoints, key=os.path.getmtime)
            self.logger.info(f"🏆 Checkpoint terbaik: {best}")
            return str(best)
        
        # Jika tidak ada file best, analisis metadata untuk mencari terbaik
        try:
            # Dapatkan semua checkpoint dengan metadata
            all_checkpoints = self.list_checkpoints()
            
            if not all_checkpoints:
                raise FileNotFoundError("❌ Tidak ada checkpoint yang ditemukan")
            
            # Cari checkpoint dengan nilai metric tertinggi
            best_score = -1
            best_checkpoint = None
            
            for cp in all_checkpoints:
                cp_info = self.get_checkpoint_info(cp)
                if cp_info and metric in cp_info.get('metrics', {}):
                    score = cp_info['metrics'][metric]
                    if score > best_score:
                        best_score = score
                        best_checkpoint = cp
            
            if best_checkpoint:
                self.logger.info(f"🏆 Checkpoint terbaik ({metric}={best_score:.4f}): {best_checkpoint}")
                return best_checkpoint
            
            # Jika tidak ada metadata, gunakan yang terbaru
            self.logger.warning(f"⚠️ Tidak ada checkpoint dengan metadata {metric}, menggunakan yang terbaru")
            return self.get_latest_checkpoint()
            
        except Exception as e:
            self.logger.warning(f"⚠️ Gagal mencari checkpoint terbaik: {str(e)}")
            # Fallback ke yang terbaru
            return self.get_latest_checkpoint()
    
    def list_checkpoints(self, pattern: str = None) -> List[str]:
        """
        Dapatkan daftar checkpoint yang tersedia.
        
        Args:
            pattern: Pola glob untuk memfilter file (opsional)
            
        Returns:
            List path checkpoint
        """
        # Tentukan pola pencarian
        search_pattern = pattern or "*.pt"
        
        # Dapatkan semua checkpoint
        checkpoints = list(self.checkpoints_dir.glob(search_pattern))
        
        # Konversi ke string dan urutkan berdasarkan waktu modifikasi (terbaru dulu)
        checkpoints_str = [str(cp) for cp in checkpoints]
        checkpoints_str.sort(key=os.path.getmtime, reverse=True)
        
        self.logger.info(f"📋 Ditemukan {len(checkpoints_str)} checkpoint")
        return checkpoints_str
    
    def validate_checkpoint(self, checkpoint_path: str) -> bool:
        """
        Validasi integritas checkpoint.
        
        Args:
            checkpoint_path: Path ke checkpoint
            
        Returns:
            True jika checkpoint valid
            
        Raises:
            FileNotFoundError: Jika file tidak ditemukan
            RuntimeError: Jika checkpoint tidak valid
        """
        try:
            # Cek keberadaan file
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"❌ File checkpoint tidak ditemukan: {checkpoint_path}")
            
            # Coba load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Verifikasi konten
            required_keys = ['model_state_dict', 'optimizer_state_dict', 'epoch']
            missing_keys = [key for key in required_keys if key not in checkpoint]
            
            if missing_keys:
                self.logger.warning(f"⚠️ Checkpoint {os.path.basename(checkpoint_path)} tidak memiliki kunci: {missing_keys}")
                # Masih dianggap valid meskipun ada key yang hilang
            
            self.logger.debug(f"✅ Checkpoint valid: {os.path.basename(checkpoint_path)}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Checkpoint tidak valid: {str(e)}")
            raise RuntimeError(f"Checkpoint tidak valid: {str(e)}")
    
    def get_checkpoint_info(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Dapatkan informasi dari checkpoint.
        
        Args:
            checkpoint_path: Path ke checkpoint
            
        Returns:
            Dictionary berisi informasi checkpoint
        """
        try:
            # Cek keberadaan file
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"❌ File checkpoint tidak ditemukan: {checkpoint_path}")
            
            # Coba load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Ekstrak info
            info = {
                'filename': os.path.basename(checkpoint_path),
                'path': checkpoint_path,
                'size': os.path.getsize(checkpoint_path),
                'last_modified': os.path.getmtime(checkpoint_path),
                'epoch': checkpoint.get('epoch', 0),
                'metrics': checkpoint.get('metrics', {}),
                'config': checkpoint.get('config', {}),
                'backbone': self._extract_backbone_info(checkpoint_path, checkpoint)
            }
            
            return info
            
        except Exception as e:
            self.logger.warning(f"⚠️ Gagal mendapatkan info checkpoint: {str(e)}")
            # Return info minimal
            return {
                'filename': os.path.basename(checkpoint_path),
                'path': checkpoint_path,
                'size': os.path.getsize(checkpoint_path) if os.path.exists(checkpoint_path) else 0,
                'last_modified': os.path.getmtime(checkpoint_path) if os.path.exists(checkpoint_path) else 0,
            }
    
    def _extract_backbone_info(self, checkpoint_path: str, checkpoint: Dict) -> str:
        """
        Ekstrak informasi backbone dari checkpoint.
        
        Args:
            checkpoint_path: Path ke checkpoint
            checkpoint: Data checkpoint yang sudah dimuat
            
        Returns:
            String informasi backbone
        """
        # Cek dari config
        if 'config' in checkpoint and 'backbone' in checkpoint['config']:
            return checkpoint['config']['backbone']
        
        # Cek dari nama file
        filename = os.path.basename(checkpoint_path).lower()
        if 'efficientnet' in filename:
            return 'efficientnet'
        elif 'darknet' in filename or 'csp' in filename:
            return 'cspdarknet'
        
        # Default
        return 'unknown'