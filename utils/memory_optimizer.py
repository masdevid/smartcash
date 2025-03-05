# File: smartcash/utils/memory_optimizer.py
# Author: Alfrida Sabar
# Deskripsi: Utilitas untuk mengoptimalkan penggunaan memori, terutama untuk GPU

import torch
import gc
from typing import Optional, Dict
from smartcash.utils.logger import SmartCashLogger

class MemoryOptimizer:
    """Kelas untuk mengoptimalkan penggunaan memori di lingkungan GPU terbatas"""
    
    def __init__(self, logger=None):
        self.logger = logger or SmartCashLogger("memory_optimizer")
        
    def check_gpu_status(self):
        """Cek status GPU dan RAM"""
        try:
            # Cek apakah GPU tersedia
            if torch.cuda.is_available():
                # Info GPU
                gpu_name = torch.cuda.get_device_name(0)
                memory_allocated = torch.cuda.memory_allocated(0) / (1024**2)
                memory_reserved = torch.cuda.memory_reserved(0) / (1024**2)
                
                print(f"🖥️ GPU: {gpu_name}")
                print(f"💾 GPU Memory Terpakai: {memory_allocated:.2f} MB")
                print(f"💾 GPU Memory Dicadangkan: {memory_reserved:.2f} MB")
                
                # Cek persentase penggunaan
                if torch.cuda.is_available():
                    try:
                        # Pendekatan 1: Menggunakan pynvml
                        import pynvml
                        pynvml.nvmlInit()
                        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        total_mem = info.total / (1024**2)
                        used_mem = info.used / (1024**2)
                        free_mem = info.free / (1024**2)
                        used_percent = (used_mem / total_mem) * 100
                        
                        print(f"💾 Total GPU Memory: {total_mem:.2f} MB")
                        print(f"💾 GPU Memory Bebas: {free_mem:.2f} MB")
                        print(f"📊 Penggunaan GPU: {used_percent:.2f}%")
                    except:
                        # Pendekatan 2: Menggunakan subprocess untuk nvidia-smi
                        try:
                            import subprocess
                            result = subprocess.check_output(
                                ['nvidia-smi', '--query-gpu=memory.total,memory.used,memory.free', '--format=csv,nounits,noheader'],
                                encoding='utf-8'
                            )
                            total_mem, used_mem, free_mem = map(int, result.strip().split(','))
                            used_percent = (used_mem / total_mem) * 100
                            
                            print(f"💾 Total GPU Memory: {total_mem} MB")
                            print(f"💾 GPU Memory Bebas: {free_mem} MB")
                            print(f"📊 Penggunaan GPU: {used_percent:.2f}%")
                        except:
                            pass
            else:
                print("❌ GPU tidak tersedia, menggunakan CPU")
            
            # Cek RAM
            try:
                import psutil
                ram = psutil.virtual_memory()
                ram_total = ram.total / (1024**3)  # GB
                ram_used = ram.used / (1024**3)    # GB
                ram_free = ram.available / (1024**3)  # GB
                ram_percent = ram.percent
                
                print(f"🖥️ Total RAM: {ram_total:.2f} GB")
                print(f"💾 RAM Terpakai: {ram_used:.2f} GB")
                print(f"💾 RAM Bebas: {ram_free:.2f} GB")
                print(f"📊 Penggunaan RAM: {ram_percent:.2f}%")
            except:
                pass
                
        except Exception as e:
            self.logger.error(f"❌ Gagal memeriksa status GPU: {str(e)}")
    
    def clear_gpu_memory(self):
        """Bersihkan memori GPU"""
        try:
            if torch.cuda.is_available():
                # Catat penggunaan awal
                memory_before = torch.cuda.memory_allocated(0) / (1024**2)
                
                # Bersihkan cache dan memori tidak terpakai
                torch.cuda.empty_cache()
                gc.collect()
                
                # Catat penggunaan setelah
                memory_after = torch.cuda.memory_allocated(0) / (1024**2)
                memory_freed = memory_before - memory_after
                
                self.logger.success(f"✅ Memori GPU dibersihkan")
                self.logger.info(f"📊 Memori dibebaskan: {memory_freed:.2f} MB")
                
                return memory_freed
            else:
                self.logger.info("ℹ️ GPU tidak tersedia, tidak perlu membersihkan")
                return 0
        except Exception as e:
            self.logger.error(f"❌ Gagal membersihkan memori GPU: {str(e)}")
            return 0
    
    def optimize_for_inference(self, model):
        """Optimasi model untuk inferensi"""
        try:
            # Pastikan mode evaluasi
            model.eval()
            
            # Konversi ke half precision jika GPU tersedia
            if torch.cuda.is_available():
                model_fp16 = model.half()  # 16-bit floating-point
                self.logger.info("✅ Model dikonversi ke FP16 (half precision)")
                return model_fp16
            
            return model
        except Exception as e:
            self.logger.error(f"❌ Gagal mengoptimasi model: {str(e)}")
            return model
    
    def optimize_batch_size(self, model, target_memory_usage=0.7):
        """Cari batch size optimal menggunakan binary search"""
        if not torch.cuda.is_available():
            self.logger.info("ℹ️ GPU tidak tersedia, menggunakan batch size default (8)")
            return 8
        
        try:
            # Cari total memory GPU
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                total_mem = info.total / (1024**3)  # GB
            except:
                # Estimasi berdasarkan GPU Colab standar (~15GB)
                total_mem = 15.0
            
            # Target memory usage dalam GB
            target_mem = total_mem * target_memory_usage
            
            # Binary search untuk batch size optimal
            min_batch = 1
            max_batch = 128
            optimal_batch = 8  # Default
            
            # Bersihkan memori sebelum mulai
            self.clear_gpu_memory()
            
            while min_batch <= max_batch:
                mid_batch = (min_batch + max_batch) // 2
                
                try:
                    # Buat input dummy
                    dummy_input = torch.randn(mid_batch, 3, 640, 640, device='cuda')
                    
                    # Coba jalankan forward pass
                    with torch.no_grad():
                        model(dummy_input)
                    
                    # Berhasil, coba lebih besar
                    optimal_batch = mid_batch
                    min_batch = mid_batch + 1
                    
                    # Bersihkan setelah satu uji coba
                    del dummy_input
                    torch.cuda.empty_cache()
                except RuntimeError as e:
                    # Out of memory, coba lebih kecil
                    max_batch = mid_batch - 1
                    
                    # Bersihkan setelah error
                    torch.cuda.empty_cache()
                    gc.collect()
            
            self.logger.success(f"✅ Batch size optimal: {optimal_batch}")
            return optimal_batch
            
        except Exception as e:
            self.logger.error(f"❌ Gagal menentukan batch size optimal: {str(e)}")
            return 8  # Default batch size
            
    def get_optimization_stats(self) -> Dict:
        """Dapatkan statistik optimasi memori"""
        stats = {}
        
        # Cek penggunaan GPU
        if torch.cuda.is_available():
            stats['gpu_available'] = True
            stats['gpu_name'] = torch.cuda.get_device_name(0)
            stats['memory_allocated'] = torch.cuda.memory_allocated(0) / (1024**2)
            stats['memory_reserved'] = torch.cuda.memory_reserved(0) / (1024**2)
            
            # Coba dapatkan lebih banyak info
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                stats['total_memory'] = info.total / (1024**2)
                stats['used_memory'] = info.used / (1024**2)
                stats['free_memory'] = info.free / (1024**2)
                stats['memory_usage_percent'] = (info.used / info.total) * 100
            except:
                pass
        else:
            stats['gpu_available'] = False
            
        # Cek RAM
        try:
            import psutil
            ram = psutil.virtual_memory()
            stats['ram_total_gb'] = ram.total / (1024**3)
            stats['ram_used_gb'] = ram.used / (1024**3)
            stats['ram_free_gb'] = ram.available / (1024**3)
            stats['ram_percent'] = ram.percent
        except:
            pass
            
        return stats