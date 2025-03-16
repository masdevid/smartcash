"""
File: smartcash/model/manager_checkpoint.py
Deskripsi: Integrasi checkpoint service dengan model manager
"""

from typing import Dict, List, Optional, Union, Any
from pathlib import Path

import torch

from smartcash.common.logger import get_logger
from smartcash.common.interfaces.checkpoint_interface import ICheckpointService
from smartcash.model.exceptions import ModelCheckpointError
from smartcash.model.services.checkpoint.checkpoint_service import CheckpointService


class ModelCheckpointManager(ICheckpointService):
    """
    Manager untuk integrasi model manager dengan checkpoint service.
    Menyediakan fungsi-fungsi untuk menyimpan, memuat, dan mengelola 
    checkpoint model.
    """
    
    def __init__(
        self,
        model_manager,
        checkpoint_dir: str = "runs/train/checkpoints",
        max_checkpoints: int = 5,
        logger = None
    ):
        """
        Inisialisasi Model Checkpoint Manager.
        
        Args:
            model_manager: Instance ModelManager yang akan diintegrasikan
            checkpoint_dir: Direktori untuk menyimpan checkpoint
            max_checkpoints: Jumlah maksimum checkpoint yang disimpan
            logger: Logger untuk mencatat aktivitas
        """
        self.model_manager = model_manager
        self.logger = logger or get_logger("model_checkpoint_manager")
        
        # Inisialisasi checkpoint service
        self.checkpoint_service = CheckpointService(
            checkpoint_dir=checkpoint_dir,
            max_checkpoints=max_checkpoints,
            logger=self.logger
        )
        
        # Buat referensi ke checkpoint service di model manager
        self.model_manager.set_checkpoint_service(self)
        
        self.logger.info(f"✨ ModelCheckpointManager diinisialisasi dengan checkpoint dir: {checkpoint_dir}")
    
    def save_checkpoint(
        self,
        model: Optional[torch.nn.Module] = None,
        path: str = "model_checkpoint.pt",
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
        is_best: bool = False
    ) -> str:
        """
        Simpan checkpoint model.
        
        Args:
            model: Model yang akan disimpan (default: model dari model_manager)
            path: Path untuk menyimpan checkpoint
            optimizer: Optimizer untuk disimpan
            epoch: Nomor epoch saat ini
            metadata: Metadata tambahan untuk disimpan
            is_best: Flag untuk menandai checkpoint terbaik
            
        Returns:
            Path ke checkpoint yang disimpan
            
        Raises:
            ModelCheckpointError: Jika gagal menyimpan checkpoint
        """
        try:
            # Gunakan model dari model_manager jika tidak disediakan
            if model is None:
                if self.model_manager.model is None:
                    self.model_manager.build_model()
                model = self.model_manager.model
            
            # Gabungkan metadata dengan informasi model
            merged_metadata = {
                'model_type': self.model_manager.model_type,
                'backbone': self.model_manager.config.get('backbone', 'unknown'),
                'img_size': self.model_manager.config.get('img_size', [640, 640]),
                'detection_layers': self.model_manager.config.get('detection_layers', [])
            }
            
            # Tambahkan metadata custom jika disediakan
            if metadata:
                merged_metadata.update(metadata)
            
            # Simpan checkpoint
            return self.checkpoint_service.save_checkpoint(
                model=model,
                path=path,
                optimizer=optimizer,
                epoch=epoch,
                metadata=merged_metadata,
                is_best=is_best
            )
            
        except Exception as e:
            error_msg = f"❌ Gagal menyimpan checkpoint model: {str(e)}"
            self.logger.error(error_msg)
            raise ModelCheckpointError(error_msg) from e
    
    def load_checkpoint(
        self,
        path: str,
        model: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        map_location: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Load checkpoint ke model.
        
        Args:
            path: Path ke file checkpoint
            model: Model untuk load state (default: model dari model_manager)
            optimizer: Optimizer untuk load state
            map_location: Device untuk load checkpoint
            
        Returns:
            Model yang sudah diload dengan state dari checkpoint
            
        Raises:
            ModelCheckpointError: Jika gagal load checkpoint
        """
        try:
            # Gunakan model dari model_manager jika tidak disediakan
            if model is None:
                if self.model_manager.model is None:
                    self.model_manager.build_model()
                model = self.model_manager.model
            
            # Load device dari model_manager jika tidak disediakan
            if map_location is None:
                map_location = self.model_manager.config.get('device', 'cpu')
            
            # Load checkpoint ke model
            loaded_checkpoint = self.checkpoint_service.load_checkpoint(
                path=path,
                model=model,
                optimizer=optimizer,
                map_location=map_location
            )
            
            # Update model di model_manager
            self.model_manager.model = model
            
            return loaded_checkpoint
            
        except Exception as e:
            error_msg = f"❌ Gagal load checkpoint model: {str(e)}"
            self.logger.error(error_msg)
            raise ModelCheckpointError(error_msg) from e
    
    def get_best_checkpoint(self) -> Optional[str]:
        """
        Dapatkan path ke checkpoint terbaik.
        
        Returns:
            Path ke checkpoint terbaik atau None jika tidak ada
        """
        return self.checkpoint_service.get_best_checkpoint()
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """
        Dapatkan path ke checkpoint terbaru.
        
        Returns:
            Path ke checkpoint terbaru atau None jika tidak ada
        """
        return self.checkpoint_service.get_latest_checkpoint()
    
    def list_checkpoints(self, sort_by: str = 'time') -> List[Dict[str, Any]]:
        """
        Daftar semua checkpoint yang tersedia.
        
        Args:
            sort_by: Kriteria pengurutan ('time', 'name', atau 'epoch')
            
        Returns:
            List dictionary dengan informasi checkpoint
        """
        return self.checkpoint_service.list_checkpoints(sort_by=sort_by)
    
    def export_to_onnx(
        self,
        output_path: str,
        input_shape: List[int] = None,
        opset_version: int = 12,
        dynamic_axes: Optional[Dict] = None
    ) -> str:
        """
        Export model ke format ONNX.
        
        Args:
            output_path: Path untuk menyimpan model ONNX
            input_shape: Bentuk input tensor (default dari config model)
            opset_version: Versi ONNX opset
            dynamic_axes: Dictionary axes dinamis untuk input/output
            
        Returns:
            Path ke file ONNX yang disimpan
            
        Raises:
            ModelCheckpointError: Jika gagal mengekspor model
        """
        try:
            # Pastikan model sudah dibangun
            if self.model_manager.model is None:
                self.model_manager.build_model()
            
            # Gunakan input shape dari config jika tidak disediakan
            if input_shape is None:
                img_size = self.model_manager.config.get('img_size', [640, 640])
                input_shape = [1, 3, img_size[0], img_size[1]]
            
            # Export model ke ONNX
            return self.checkpoint_service.export_to_onnx(
                model=self.model_manager.model,
                output_path=output_path,
                input_shape=input_shape,
                opset_version=opset_version,
                dynamic_axes=dynamic_axes
            )
            
        except Exception as e:
            error_msg = f"❌ Gagal mengekspor model ke ONNX: {str(e)}"
            self.logger.error(error_msg)
            raise ModelCheckpointError(error_msg) from e