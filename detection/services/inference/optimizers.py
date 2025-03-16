"""
File: smartcash/detection/services/inference/optimizers.py
Deskripsi: Fungsi optimasi untuk inferensi model deteksi objek.
"""

import os
from typing import Dict, Optional, Union, Any

from smartcash.common.logger import SmartCashLogger, get_logger
from smartcash.common.constants import ModelFormat, MODEL_EXTENSIONS


class ModelOptimizer:
    """Kelas utilitas untuk optimasi model inferensi"""
    
    def __init__(self, logger: Optional[SmartCashLogger] = None):
        """
        Inisialisasi ModelOptimizer
        
        Args:
            logger: Logger untuk mencatat aktivitas (opsional)
        """
        self.logger = logger or get_logger("ModelOptimizer")
    
    def optimize_to_onnx(self, 
                        model, 
                        output_path: str,
                        input_shape: tuple = (1, 3, 640, 640),
                        dynamic_axes: Dict = None,
                        opset_version: int = 12,
                        simplify: bool = True) -> bool:
        """
        Optimasi model ke format ONNX
        
        Args:
            model: Model PyTorch yang akan dioptimasi
            output_path: Path untuk menyimpan model ONNX
            input_shape: Bentuk input model
            dynamic_axes: Axes dinamis untuk input/output model
            opset_version: Versi ONNX opset
            simplify: Flag untuk menyederhanakan model ONNX
            
        Returns:
            Boolean yang menunjukkan keberhasilan optimasi
        """
        try:
            import torch
            
            # Default dynamic axes jika tidak ditentukan
            if dynamic_axes is None:
                dynamic_axes = {
                    "input": {0: "batch_size"},
                    "output": {0: "batch_size"}
                }
            
            # Pastikan direktori output ada
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Dapatkan nama input dan output model jika tersedia
            input_names = ["input"]
            output_names = ["output"]
            
            if hasattr(model, 'input_names'):
                input_names = model.input_names
            if hasattr(model, 'output_names'):
                output_names = model.output_names
            
            # Buat dummy input
            x = torch.randn(input_shape, requires_grad=True)
            
            # Export ke ONNX
            self.logger.info(f"🔄 Mengekspor model ke ONNX (opset {opset_version})")
            torch.onnx.export(
                model,
                x,
                output_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes
            )
            
            # Simplifikasi model ONNX jika diminta
            if simplify:
                try:
                    import onnx
                    from onnxsim import simplify as onnx_simplify
                    
                    self.logger.info(f"🔄 Menyederhanakan model ONNX")
                    
                    # Load model ONNX
                    onnx_model = onnx.load(output_path)
                    
                    # Check model
                    onnx.checker.check_model(onnx_model)
                    
                    # Simplify
                    simplified_model, check = onnx_simplify(onnx_model)
                    
                    if check:
                        # Save simplified model
                        onnx.save(simplified_model, output_path)
                        self.logger.info(f"✅ Model ONNX berhasil disederhanakan")
                    else:
                        self.logger.warning(f"⚠️ Simplifikasi model ONNX tidak berhasil divalidasi")
                        
                except ImportError:
                    self.logger.warning(f"⚠️ Package onnx-simplifier tidak terinstal, melewati simplifikasi")
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ Error saat menyederhanakan model ONNX: {str(e)}")
            
            self.logger.info(f"✅ Model berhasil diekspor ke {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error saat mengoptimasi model ke ONNX: {str(e)}")
            return False
    
    def optimize_to_torchscript(self, 
                              model, 
                              output_path: str,
                              input_shape: tuple = (1, 3, 640, 640),
                              method: str = 'trace') -> bool:
        """
        Optimasi model ke format TorchScript
        
        Args:
            model: Model PyTorch yang akan dioptimasi
            output_path: Path untuk menyimpan model TorchScript
            input_shape: Bentuk input model
            method: Metode konversi ('trace' atau 'script')
            
        Returns:
            Boolean yang menunjukkan keberhasilan optimasi
        """
        try:
            import torch
            
            # Pastikan direktori output ada
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Konversi model ke TorchScript
            self.logger.info(f"🔄 Mengkonversi model ke TorchScript dengan metode '{method}'")
            
            # Pilih metode konversi
            if method == 'trace':
                # Buat dummy input
                x = torch.randn(input_shape)
                
                # Trace model
                scripted_model = torch.jit.trace(model, x)
                
            elif method == 'script':
                # Script model
                scripted_model = torch.jit.script(model)
                
            else:
                self.logger.error(f"❌ Metode konversi tidak valid: {method}")
                return False
            
            # Simpan model
            scripted_model.save(output_path)
            
            self.logger.info(f"✅ Model berhasil dikonversi ke TorchScript ({output_path})")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error saat mengoptimasi model ke TorchScript: {str(e)}")
            return False
    
    def optimize_to_tensorrt(self, 
                           onnx_path: str, 
                           output_path: str,
                           fp16_mode: bool = True,
                           int8_mode: bool = False,
                           workspace_size: int = 1 << 30) -> bool:
        """
        Optimasi model ONNX ke format TensorRT
        
        Args:
            onnx_path: Path ke model ONNX yang akan dioptimasi
            output_path: Path untuk menyimpan model TensorRT
            fp16_mode: Flag untuk mengaktifkan mode FP16
            int8_mode: Flag untuk mengaktifkan mode INT8
            workspace_size: Ukuran workspace TensorRT dalam bytes
            
        Returns:
            Boolean yang menunjukkan keberhasilan optimasi
        """
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit
            import numpy as np
            
            # Pastikan direktori output ada
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Buat TensorRT logger
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            
            # Buat builder dan network
            self.logger.info(f"🔄 Membuat TensorRT engine dari {onnx_path}")
            builder = trt.Builder(TRT_LOGGER)
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            parser = trt.OnnxParser(network, TRT_LOGGER)
            
            # Parse model ONNX
            with open(onnx_path, 'rb') as model:
                if not parser.parse(model.read()):
                    self.logger.error(f"❌ Error saat parsing model ONNX")
                    for error in range(parser.num_errors):
                        self.logger.error(f"    - {parser.get_error(error)}")
                    return False
            
            # Set ukuran batch maksimum
            builder.max_batch_size = 1
            
            # Konfigurasi builder
            config = builder.create_builder_config()
            config.max_workspace_size = workspace_size
            
            # Aktifkan mode FP16 jika diminta
            if fp16_mode and builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                self.logger.info(f"✓ Mode FP16 diaktifkan")
            
            # Aktifkan mode INT8 jika diminta
            if int8_mode and builder.platform_has_fast_int8:
                config.set_flag(trt.BuilderFlag.INT8)
                self.logger.info(f"✓ Mode INT8 diaktifkan")
                # Catatan: Untuk INT8 sepenuhnya, perlu tambahan kalibrasi
            
            # Buat engine
            engine = builder.build_engine(network, config)
            
            # Simpan engine
            with open(output_path, 'wb') as f:
                f.write(engine.serialize())
            
            self.logger.info(f"✅ Model berhasil dikonversi ke TensorRT ({output_path})")
            return True
            
        except ImportError as e:
            self.logger.error(f"❌ Package TensorRT tidak terinstal: {str(e)}")
            return False
            
        except Exception as e:
            self.logger.error(f"❌ Error saat mengoptimasi model ke TensorRT: {str(e)}")
            return False
    
    def optimize_to_tflite(self, 
                         model_path: str, 
                         output_path: str,
                         quantize: bool = False,
                         input_shape: tuple = (1, 640, 640, 3)) -> bool:
        """
        Optimasi model ke format TensorFlow Lite
        
        Args:
            model_path: Path ke model TensorFlow yang akan dioptimasi
            output_path: Path untuk menyimpan model TFLite
            quantize: Flag untuk mengaktifkan quantization
            input_shape: Bentuk input model
            
        Returns:
            Boolean yang menunjukkan keberhasilan optimasi
        """
        try:
            import tensorflow as tf
            
            # Pastikan direktori output ada
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Load model TensorFlow
            self.logger.info(f"🔄 Memuat model TensorFlow dari {model_path}")
            model = tf.saved_model.load(model_path)
            
            # Buat TFLite converter
            converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
            
            # Set opsi optimasi
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            # Aktifkan quantization jika diminta
            if quantize:
                self.logger.info(f"✓ Quantization diaktifkan")
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_types = [tf.float16]
            
            # Konversi model
            tflite_model = converter.convert()
            
            # Simpan model
            with open(output_path, 'wb') as f:
                f.write(tflite_model)
            
            self.logger.info(f"✅ Model berhasil dikonversi ke TFLite ({output_path})")
            return True
            
        except ImportError as e:
            self.logger.error(f"❌ Package TensorFlow tidak terinstal: {str(e)}")
            return False
            
        except Exception as e:
            self.logger.error(f"❌ Error saat mengoptimasi model ke TFLite: {str(e)}")
            return False
    
    def optimize_model(self, model, model_format: ModelFormat, output_path: str, **kwargs) -> bool:
        """
        Optimasi model ke format yang ditentukan
        
        Args:
            model: Model yang akan dioptimasi
            model_format: Format target optimasi
            output_path: Path untuk menyimpan model hasil optimasi
            **kwargs: Parameter tambahan untuk optimasi
            
        Returns:
            Boolean yang menunjukkan keberhasilan optimasi
        """
        # Pastikan ekstensi file output sesuai dengan format model
        if not output_path.endswith(MODEL_EXTENSIONS.get(model_format, "")):
            output_path = f"{output_path}{MODEL_EXTENSIONS.get(model_format, '')}"
            
        # Optimasi model berdasarkan format
        if model_format == ModelFormat.ONNX:
            return self.optimize_to_onnx(model, output_path, **kwargs)
            
        elif model_format == ModelFormat.TORCHSCRIPT:
            return self.optimize_to_torchscript(model, output_path, **kwargs)
            
        elif model_format == ModelFormat.TENSORRT:
            # Untuk TensorRT, perlu model ONNX terlebih dahulu
            if kwargs.get('onnx_path') is None:
                onnx_path = f"{output_path}.onnx"
                if not os.path.exists(onnx_path):
                    if not self.optimize_to_onnx(model, onnx_path, **kwargs):
                        return False
                kwargs['onnx_path'] = onnx_path
                
            return self.optimize_to_tensorrt(kwargs['onnx_path'], output_path, **kwargs)
            
        elif model_format == ModelFormat.TFLITE:
            # Untuk TFLite, perlu model TensorFlow terlebih dahulu
            return self.optimize_to_tflite(kwargs.get('model_path'), output_path, **kwargs)
            
        else:
            self.logger.error(f"❌ Format model tidak didukung: {model_format}")
            return False