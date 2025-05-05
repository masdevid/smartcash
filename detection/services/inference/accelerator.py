"""
File: smartcash/detection/services/inference/accelerator.py
Deskripsi: Abstraksi hardware untuk akselerasi inferensi (CPU/GPU/TPU).
"""

import os
import platform
from enum import Enum
from typing import Dict, Optional, Union, Any

from smartcash.common.logger import SmartCashLogger, get_logger


class AcceleratorType(Enum):
    """Tipe akselerator hardware"""
    CPU = 'cpu'
    CUDA = 'cuda'
    MPS = 'mps'
    TPU = 'tpu'
    ROCm = 'rocm'
    AUTO = 'auto'


class HardwareAccelerator:
    """Abstraksi hardware untuk akselerasi inferensi"""
    
    def __init__(self, 
                accelerator_type: AcceleratorType = AcceleratorType.AUTO,
                device_id: int = 0,
                use_fp16: bool = True,
                logger: Optional[SmartCashLogger] = None):
        """
        Inisialisasi Hardware Accelerator
        
        Args:
            accelerator_type: Tipe akselerator yang akan digunakan
            device_id: ID device untuk akselerator
            use_fp16: Flag untuk menggunakan precision FP16
            logger: Logger untuk mencatat aktivitas (opsional)
        """
        self.accelerator_type = accelerator_type
        self.device_id = device_id
        self.use_fp16 = use_fp16
        self.logger = logger or get_logger("HardwareAccelerator")
        self.device = None
        
        # Deteksi otomatis hardware terbaik jika AUTO
        if self.accelerator_type == AcceleratorType.AUTO:
            self.accelerator_type = self._auto_detect_hardware()
    
    def setup(self) -> bool:
        """
        Setup akselerator untuk inferensi
        
        Returns:
            Boolean yang menunjukkan keberhasilan setup
        """
        try:
            # Setup berdasarkan tipe akselerator
            if self.accelerator_type == AcceleratorType.CUDA:
                return self._setup_cuda()
                
            elif self.accelerator_type == AcceleratorType.MPS:
                return self._setup_mps()
                
            elif self.accelerator_type == AcceleratorType.TPU:
                return self._setup_tpu()
                
            elif self.accelerator_type == AcceleratorType.ROCm:
                return self._setup_rocm()
                
            elif self.accelerator_type == AcceleratorType.CPU:
                return self._setup_cpu()
                
            else:
                self.logger.warning(f"⚠️ Tipe akselerator tidak dikenal: {self.accelerator_type}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Error saat setup akselerator: {str(e)}")
            # Fallback ke CPU
            self.accelerator_type = AcceleratorType.CPU
            return self._setup_cpu()
    
    def get_device(self) -> Any:
        """
        Dapatkan device yang dikonfigurasi
        
        Returns:
            Device untuk akselerasi inferensi
        """
        if self.device is None:
            self.setup()
        return self.device
    
    def get_device_info(self) -> Dict:
        """
        Dapatkan informasi device
        
        Returns:
            Dictionary berisi informasi device
        """
        info = {
            "accelerator_type": self.accelerator_type.value,
            "device_id": self.device_id,
            "use_fp16": self.use_fp16,
            "platform": platform.platform(),
            "system": platform.system(),
            "processor": platform.processor()
        }
        
        # Tambahkan informasi CUDA jika tersedia
        if self.accelerator_type == AcceleratorType.CUDA:
            try:
                import torch
                info["cuda_available"] = torch.cuda.is_available()
                if info["cuda_available"]:
                    info["cuda_device_count"] = torch.cuda.device_count()
                    info["cuda_device_name"] = torch.cuda.get_device_name(self.device_id)
                    info["cuda_version"] = torch.version.cuda
                    info["cudnn_version"] = torch.backends.cudnn.version()
                    info["cudnn_enabled"] = torch.backends.cudnn.enabled
            except ImportError:
                info["cuda_available"] = False
        
        # Tambahkan informasi MPS jika tersedia
        elif self.accelerator_type == AcceleratorType.MPS:
            try:
                import torch
                info["mps_available"] = torch.backends.mps.is_available()
            except (ImportError, AttributeError):
                info["mps_available"] = False
        
        return info
    
    def _auto_detect_hardware(self) -> AcceleratorType:
        """
        Deteksi otomatis hardware terbaik
        
        Returns:
            Tipe akselerator terbaik yang tersedia
        """
        try:
            import torch
            
            # Cek ketersediaan CUDA
            if torch.cuda.is_available():
                self.logger.info(f"🔍 Terdeteksi CUDA dengan {torch.cuda.device_count()} device")
                return AcceleratorType.CUDA
            
            # Cek ketersediaan MPS (Apple Metal)
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.logger.info(f"🔍 Terdeteksi MPS (Apple Silicon)")
                return AcceleratorType.MPS
            
            # Cek ketersediaan ROCm
            if hasattr(torch, 'hip') and torch.hip.is_available():
                self.logger.info(f"🔍 Terdeteksi ROCm")
                return AcceleratorType.ROCm
            
            # Fallback ke CPU
            self.logger.info(f"🔍 Tidak terdeteksi akselerator GPU, menggunakan CPU")
            return AcceleratorType.CPU
            
        except ImportError:
            self.logger.warning(f"⚠️ PyTorch tidak terinstal, menggunakan CPU")
            return AcceleratorType.CPU
    
    def _setup_cuda(self) -> bool:
        """
        Setup akselerator CUDA
        
        Returns:
            Boolean yang menunjukkan keberhasilan setup
        """
        try:
            import torch
            
            if not torch.cuda.is_available():
                self.logger.warning(f"⚠️ CUDA tidak tersedia, fallback ke CPU")
                self.accelerator_type = AcceleratorType.CPU
                return self._setup_cpu()
            
            # Validasi device ID
            device_count = torch.cuda.device_count()
            if self.device_id >= device_count:
                self.logger.warning(f"⚠️ ID device CUDA {self.device_id} tidak valid (tersedia {device_count} device). Menggunakan device 0.")
                self.device_id = 0
            
            # Set device
            device_name = torch.cuda.get_device_name(self.device_id)
            self.device = torch.device(f'cuda:{self.device_id}')
            torch.cuda.set_device(self.device_id)
            
            # Optimasi CUDA
            torch.backends.cudnn.benchmark = True
            
            # Setup FP16
            if self.use_fp16:
                self.logger.info(f"✓ Menggunakan FP16 precision")
                torch.set_default_dtype(torch.float16)
                # CUDNN auto-tuner
                torch.backends.cudnn.benchmark = True
            
            self.logger.info(f"✅ CUDA diaktifkan pada {device_name} (ID: {self.device_id})")
            return True
            
        except ImportError:
            self.logger.error(f"❌ PyTorch atau CUDA tidak terinstal")
            self.accelerator_type = AcceleratorType.CPU
            return self._setup_cpu()
            
        except Exception as e:
            self.logger.error(f"❌ Error saat setup CUDA: {str(e)}")
            self.accelerator_type = AcceleratorType.CPU
            return self._setup_cpu()
    
    def _setup_mps(self) -> bool:
        """
        Setup akselerator MPS (Apple Metal)
        
        Returns:
            Boolean yang menunjukkan keberhasilan setup
        """
        try:
            import torch
            
            if not hasattr(torch.backends, 'mps') or not torch.backends.mps.is_available():
                self.logger.warning(f"⚠️ MPS tidak tersedia, fallback ke CPU")
                self.accelerator_type = AcceleratorType.CPU
                return self._setup_cpu()
            
            # Set device
            self.device = torch.device('mps')
            
            # FP16 tidak didukung penuh oleh MPS
            if self.use_fp16:
                self.logger.warning(f"⚠️ FP16 mungkin tidak didukung penuh oleh MPS, menggunakan FP32")
                self.use_fp16 = False
            
            self.logger.info(f"✅ MPS (Apple Metal) diaktifkan")
            return True
            
        except ImportError:
            self.logger.error(f"❌ PyTorch tidak terinstal atau tidak mendukung MPS")
            self.accelerator_type = AcceleratorType.CPU
            return self._setup_cpu()
            
        except Exception as e:
            self.logger.error(f"❌ Error saat setup MPS: {str(e)}")
            self.accelerator_type = AcceleratorType.CPU
            return self._setup_cpu()
    
    def _setup_tpu(self) -> bool:
        """
        Setup akselerator TPU
        
        Returns:
            Boolean yang menunjukkan keberhasilan setup
        """
        try:
            # Coba impor torch_xla untuk TPU
            import torch_xla
            import torch_xla.core.xla_model as xm
            
            # Set device
            self.device = xm.xla_device()
            
            self.logger.info(f"✅ TPU diaktifkan")
            return True
            
        except ImportError:
            self.logger.error(f"❌ torch_xla tidak terinstal")
            self.accelerator_type = AcceleratorType.CPU
            return self._setup_cpu()
            
        except Exception as e:
            self.logger.error(f"❌ Error saat setup TPU: {str(e)}")
            self.accelerator_type = AcceleratorType.CPU
            return self._setup_cpu()
    
    def _setup_rocm(self) -> bool:
        """
        Setup akselerator ROCm
        
        Returns:
            Boolean yang menunjukkan keberhasilan setup
        """
        try:
            import torch
            
            if not hasattr(torch, 'hip') or not torch.hip.is_available():
                self.logger.warning(f"⚠️ ROCm tidak tersedia, fallback ke CPU")
                self.accelerator_type = AcceleratorType.CPU
                return self._setup_cpu()
            
            # Set device
            self.device = torch.device('cuda')  # ROCm menggunakan 'cuda' sebagai identifier
            
            self.logger.info(f"✅ ROCm diaktifkan")
            return True
            
        except ImportError:
            self.logger.error(f"❌ PyTorch atau ROCm tidak terinstal")
            self.accelerator_type = AcceleratorType.CPU
            return self._setup_cpu()
            
        except Exception as e:
            self.logger.error(f"❌ Error saat setup ROCm: {str(e)}")
            self.accelerator_type = AcceleratorType.CPU
            return self._setup_cpu()
    
    def _setup_cpu(self) -> bool:
        """
        Setup akselerator CPU
        
        Returns:
            Boolean yang menunjukkan keberhasilan setup
        """
        try:
            import torch
            
            # Set device
            self.device = torch.device('cpu')
            
            # Set thread count untuk optimasi
            if hasattr(torch, 'set_num_threads'):
                import multiprocessing
                num_cores = multiprocessing.cpu_count()
                recommended_threads = max(1, num_cores - 1)  # Gunakan semua core kecuali 1
                torch.set_num_threads(recommended_threads)
                self.logger.info(f"✓ Menggunakan {recommended_threads} thread dari {num_cores} CPU core")
            
            self.logger.info(f"✅ CPU diaktifkan")
            return True
            
        except ImportError:
            self.logger.warning(f"⚠️ PyTorch tidak terinstal, menggunakan device standard")
            self.device = "cpu"
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error saat setup CPU: {str(e)}")
            self.device = "cpu"
            return True