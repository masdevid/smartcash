# File: smartcash/handlers/model/integration/checkpoint_adapter.py
# Author: Alfrida Sabar
# Deskripsi: Adapter untuk integrasi dengan CheckpointManager

import torch
from typing import Dict, Optional, Any, List, Tuple
from pathlib import Path

from smartcash.utils.logger import get_logger, SmartCashLogger
from smartcash.exceptions.base import ModelError

class CheckpointAdapter:
    """
    Adapter untuk integrasi dengan CheckpointManager.
    Menyediakan antarmuka yang konsisten untuk manajemen checkpoint.
    """
    
    def __init__(
        self,
        config: Dict,
        logger: Optional[SmartCashLogger] = None,
        output_dir: Optional[str] = None
    ):
        """
        Inisialisasi checkpoint adapter.
        
        Args:
            config: Konfigurasi aplikasi
            logger: Custom logger (opsional)
            output_dir: Direktori output untuk checkpoint (opsional)
        """
        self.config = config
        self.logger = logger or get_logger("checkpoint_adapter")
        
        # Setup output directory
        if output_dir is None:
            self.output_dir = Path(config.get('output_dir', 'runs/train')) / "weights"
        else:
            self.output_dir = Path(output_dir)
        
        # Import CheckpointManager (lazy import)
        self._checkpoint_manager = None
    
    @property
    def checkpoint_manager(self):
        """Lazy initialization of checkpoint manager."""
        if self._checkpoint_manager is None:
            from smartcash.handlers.checkpoint import CheckpointManager
            self._checkpoint_manager = CheckpointManager(
                output_dir=str(self.output_dir),
                logger=self.logger
            )
        return self._checkpoint_manager
    
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict[str, Any],
        scheduler: Optional[Any] = None,
        is_best: bool = False,
        filename: Optional[str] = None
    ) -> str:
        """
        Simpan checkpoint model.
        
        Args:
            model: Model PyTorch
            optimizer: Optimizer
            epoch: Nomor epoch saat ini
            metrics: Metrik evaluasi
            scheduler: Learning rate scheduler (opsional)
            is_best: Flag apakah ini checkpoint terbaik
            filename: Nama file checkpoint (opsional)
            
        Returns:
            Path ke checkpoint yang disimpan
        """
        try:
            # Buat direktori output jika belum ada
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Simpan checkpoint
            checkpoint_path = self.checkpoint_manager.save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics=metrics,
                config=self.config,
                is_best=is_best,
                filename=filename
            )
            
            self.logger.info(
                f"💾 Checkpoint disimpan:\n"
                f"   • Path: {checkpoint_path}\n"
                f"   • Epoch: {epoch}\n"
                f"   • Is Best: {is_best}"
            )
            
            return checkpoint_path
            
        except Exception as e:
            self.logger.error(f"❌ Gagal menyimpan checkpoint: {str(e)}")
            raise ModelError(f"Gagal menyimpan checkpoint: {str(e)}")
    
    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: Optional[torch.device] = None,
        only_weights: bool = False
    ) -> Tuple[torch.nn.Module, Dict[str, Any]]:
        """
        Muat checkpoint model.
        
        Args:
            checkpoint_path: Path ke file checkpoint
            model: Model PyTorch (opsional, jika None akan error)
            optimizer: Optimizer (opsional)
            scheduler: Learning rate scheduler (opsional)
            device: Device untuk model
            only_weights: Hanya muat weights, bukan state optimizer/scheduler
            
        Returns:
            Tuple (Model, Metadata checkpoint)
        """
        try:
            # Muat checkpoint
            checkpoint = self.checkpoint_manager.load_checkpoint(checkpoint_path)
            
            if model is None:
                raise ModelError("Model harus diberikan untuk memuat checkpoint")
            
            # Muat state dict model
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Pindahkan model ke device jika ada
            if device is not None:
                model = model.to(device)
            
            # Muat state optimizer jika tidak only_weights
            if optimizer is not None and not only_weights and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Muat state scheduler jika tidak only_weights
            if scheduler is not None and not only_weights and 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            self.logger.success(
                f"✅ Checkpoint berhasil dimuat:\n"
                f"   • Path: {checkpoint_path}\n"
                f"   • Epoch: {checkpoint.get('epoch', 'unknown')}\n"
                f"   • Loss: {checkpoint.get('metrics', {}).get('val_loss', 'unknown')}"
            )
            
            return model, checkpoint
            
        except Exception as e:
            self.logger.error(f"❌ Gagal memuat checkpoint: {str(e)}")
            raise ModelError(f"Gagal memuat checkpoint: {str(e)}")
    
    def find_best_checkpoint(self, metric: str = "val_loss") -> Optional[str]:
        """
        Temukan checkpoint terbaik berdasarkan metrik.
        
        Args:
            metric: Nama metrik untuk menentukan checkpoint terbaik
            
        Returns:
            Path ke checkpoint terbaik, atau None jika tidak ada
        """
        try:
            # Gunakan checkpoint manager untuk menemukan checkpoint terbaik
            best_checkpoint = self.checkpoint_manager.find_best_checkpoint(metric)
            
            if best_checkpoint:
                self.logger.info(f"🏆 Checkpoint terbaik ditemukan: {best_checkpoint}")
            else:
                self.logger.warning("⚠️ Tidak ada checkpoint terbaik yang ditemukan")
                
            return best_checkpoint
            
        except Exception as e:
            self.logger.error(f"❌ Gagal menemukan checkpoint terbaik: {str(e)}")
            return None
    
    def list_checkpoints(self, pattern: Optional[str] = None) -> List[str]:
        """
        Dapatkan daftar semua checkpoint yang tersedia.
        
        Args:
            pattern: Pola glob untuk memfilter file
            
        Returns:
            List path checkpoint
        """
        try:
            checkpoints = self.checkpoint_manager.list_checkpoints(pattern)
            
            self.logger.info(f"📋 {len(checkpoints)} checkpoint ditemukan")
            
            return checkpoints
            
        except Exception as e:
            self.logger.error(f"❌ Gagal mendapatkan daftar checkpoint: {str(e)}")
            return []
    
    def export_best_model(
        self, 
        format: str = 'torchscript',
        input_shape: Optional[List[int]] = None,
        metric: str = 'val_loss',
        output_path: Optional[str] = None
    ) -> Optional[str]:
        """
        Export model terbaik ke format deployment.
        
        Args:
            format: Format export ('torchscript', 'onnx')
            input_shape: Bentuk input tensor [B, C, H, W]
            metric: Metrik untuk memilih model terbaik
            output_path: Path output (opsional)
            
        Returns:
            Path ke model yang diexport, atau None jika gagal
        """
        try:
            # Temukan checkpoint terbaik
            best_checkpoint = self.find_best_checkpoint(metric)
            
            if not best_checkpoint:
                self.logger.warning("⚠️ Tidak ada checkpoint terbaik untuk diexport")
                return None
            
            # Load checkpoint
            checkpoint = self.checkpoint_manager.load_checkpoint(best_checkpoint)
            
            # Import ModelManager (lazy import)
            from smartcash.handlers.model.model_manager import ModelManager
            
            # Buat model manager
            model_manager = ModelManager(self.config, self.logger)
            
            # Muat model
            model, _ = model_manager.load_model(best_checkpoint)
            
            # Setup output path
            if output_path is None:
                export_dir = Path(self.config.get('output_dir', 'runs/export'))
                export_dir.mkdir(parents=True, exist_ok=True)
                
                # Nama file output
                filename = f"model_{format}_{metric}.{format}"
                output_path = str(export_dir / filename)
            
            # Set model ke mode evaluasi
            model.eval()
            
            # Export model sesuai format
            if format.lower() == 'torchscript':
                # Default input shape
                if input_shape is None:
                    input_shape = [1, 3, 640, 640]
                
                # Buat example input
                example_input = torch.rand(*input_shape)
                
                # Trace model
                traced_model = torch.jit.trace(model, example_input)
                
                # Simpan model
                traced_model.save(output_path)
                
                self.logger.success(f"✅ Model berhasil diexport ke TorchScript: {output_path}")
                
            elif format.lower() == 'onnx':
                # Import untuk ONNX
                try:
                    import onnx
                    import onnxruntime
                except ImportError:
                    self.logger.error("❌ Gagal mengimport onnx dan onnxruntime. Pastikan package terinstall")
                    return None
                
                # Default input shape
                if input_shape is None:
                    input_shape = [1, 3, 640, 640]
                
                # Buat example input
                example_input = torch.rand(*input_shape)
                
                # Export ke ONNX
                torch.onnx.export(
                    model,                      # Model PyTorch
                    example_input,              # Input example
                    output_path,                # Output path
                    export_params=True,         # Export parameter
                    opset_version=12,           # ONNX opset version
                    do_constant_folding=True,   # Constant folding untuk optimasi
                    input_names=['input'],      # Nama input
                    output_names=['output'],    # Nama output
                    dynamic_axes={              # Dimensi dinamis
                        'input': {0: 'batch_size'},
                        'output': {0: 'batch_size'}
                    }
                )
                
                # Verifikasi model ONNX
                onnx_model = onnx.load(output_path)
                onnx.checker.check_model(onnx_model)
                
                self.logger.success(f"✅ Model berhasil diexport ke ONNX: {output_path}")
                
            else:
                self.logger.error(f"❌ Format export '{format}' tidak didukung")
                return None
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"❌ Gagal mengexport model: {str(e)}")
            return None