# File: smartcash/utils/checkpoint_manager.py
# Author: Alfrida Sabar
# Deskripsi: Utilitas untuk mengelola checkpoint model dengan dukungan berbagai operasi

import os
import torch
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from smartcash.utils.logger import SmartCashLogger

class CheckpointManager:
    """Kelas untuk mengelola checkpoint model dengan berbagai operasi"""
    
    def __init__(self, checkpoints_dir='runs/train/weights', logger=None):
        self.checkpoints_dir = Path(checkpoints_dir)
        self.logger = logger or SmartCashLogger("checkpoint_manager")
        
        # Pastikan direktori checkpoint ada
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    def list_checkpoints(self):
        """Temukan semua checkpoint yang tersedia"""
        # Kategori checkpoint
        checkpoint_categories = {
            'best': list(self.checkpoints_dir.glob('*_best.pth')),
            'latest': list(self.checkpoints_dir.glob('*_latest.pth')),
            'epoch': list(self.checkpoints_dir.glob('*_epoch_*.pth'))
        }
        
        # Sort berdasarkan waktu modifikasi (terbaru dulu)
        for category, files in checkpoint_categories.items():
            checkpoint_categories[category] = sorted(
                files, 
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )
        
        return checkpoint_categories
    
    def get_checkpoint_info(self, checkpoint_path):
        """Dapatkan informasi dari checkpoint"""
        try:
            # Load checkpoint tanpa memuat model weights
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Ekstrak informasi
            info = {
                'filename': checkpoint_path.name,
                'path': str(checkpoint_path),
                'size': f"{checkpoint_path.stat().st_size / (1024*1024):.2f} MB",
                'modified': datetime.fromtimestamp(checkpoint_path.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
                'epoch': checkpoint.get('epoch', 'unknown'),
                'loss': checkpoint.get('loss', 'unknown'),
                'config': checkpoint.get('config', {})
            }
            
            # Ekstrak informasi backbone dan mode
            config = checkpoint.get('config', {})
            info['backbone'] = config.get('model', {}).get('backbone', 'unknown')
            info['detection_mode'] = config.get('detection_mode', 'unknown')
            info['layers'] = config.get('layers', ['unknown'])
            
            return info
            
        except Exception as e:
            self.logger.error(f"❌ Gagal mendapatkan informasi checkpoint {checkpoint_path.name}: {str(e)}")
            return {
                'filename': checkpoint_path.name,
                'path': str(checkpoint_path),
                'error': str(e)
            }
    
    def display_checkpoints(self):
        """Tampilkan daftar checkpoint dalam format yang mudah dibaca"""
        checkpoints = self.list_checkpoints()
        
        if sum(len(files) for files in checkpoints.values()) == 0:
            print("⚠️ Tidak ada checkpoint yang ditemukan")
            return
        
        # Tampilkan informasi untuk setiap kategori
        for category, files in checkpoints.items():
            if files:
                if category == 'best':
                    print(f"🏆 Checkpoint Terbaik:")
                elif category == 'latest':
                    print(f"🔄 Checkpoint Terakhir:")
                else:
                    print(f"📋 Checkpoint per Epoch:")
                
                # Tampilkan informasi untuk setiap file
                for i, file in enumerate(files):
                    info = self.get_checkpoint_info(file)
                    
                    if i < 5:  # Batasi tampilan hingga 5 file per kategori
                        print(f"  • {info['filename']}")
                        print(f"    - Epoch: {info['epoch']}")
                        print(f"    - Loss: {info.get('loss', 'unknown')}")
                        print(f"    - Backbone: {info.get('backbone', 'unknown')}")
                        print(f"    - Mode: {info.get('detection_mode', 'unknown')}")
                        print(f"    - Ukuran: {info['size']}")
                        print(f"    - Waktu: {info['modified']}")
                        print()
                
                # Tampilkan jumlah total jika ada lebih dari 5
                if len(files) > 5:
                    print(f"  ... dan {len(files) - 5} file lainnya\n")
    
    def delete_checkpoint(self, checkpoint_path):
        """Hapus checkpoint"""
        try:
            Path(checkpoint_path).unlink()
            self.logger.success(f"✅ Checkpoint {os.path.basename(checkpoint_path)} berhasil dihapus")
            return True
        except Exception as e:
            self.logger.error(f"❌ Gagal menghapus checkpoint: {str(e)}")
            return False
    
    def cleanup_checkpoints(self, keep_best=True, keep_latest=True, max_epochs=5):
        """Bersihkan checkpoint lama"""
        checkpoints = self.list_checkpoints()
        deleted_count = 0
        
        # Pertahankan checkpoint terbaik jika diminta
        if keep_best and checkpoints['best']:
            best_to_keep = checkpoints['best'][0]  # Pertahankan yang terbaru
            for cp in checkpoints['best'][1:]:  # Hapus sisanya
                self.delete_checkpoint(cp)
                deleted_count += 1
        
        # Pertahankan checkpoint terakhir jika diminta
        if keep_latest and checkpoints['latest']:
            latest_to_keep = checkpoints['latest'][0]  # Pertahankan yang terbaru
            for cp in checkpoints['latest'][1:]:  # Hapus sisanya
                self.delete_checkpoint(cp)
                deleted_count += 1
        
        # Batasi jumlah checkpoint epoch
        if len(checkpoints['epoch']) > max_epochs:
            # Pertahankan checkpoint epoch terbaru
            for cp in checkpoints['epoch'][max_epochs:]:
                self.delete_checkpoint(cp)
                deleted_count += 1
                
        self.logger.success(f"✅ Berhasil menghapus {deleted_count} checkpoint lama")
        return deleted_count
        
    def copy_to_drive(self, checkpoint_path, drive_dir):
        """Salin checkpoint ke Google Drive"""
        try:
            # Cek apakah drive_dir ada
            if not os.path.exists(drive_dir):
                os.makedirs(drive_dir, exist_ok=True)
                
            # Salin file
            import shutil
            dest_path = os.path.join(drive_dir, os.path.basename(checkpoint_path))
            shutil.copy2(checkpoint_path, dest_path)
            
            self.logger.success(f"✅ Checkpoint berhasil disalin ke {dest_path}")
            return dest_path
        except Exception as e:
            self.logger.error(f"❌ Gagal menyalin checkpoint ke Drive: {str(e)}")
            return None
    
    def find_best_checkpoint(self):
        """Temukan checkpoint terbaik berdasarkan nilai loss"""
        checkpoints = self.list_checkpoints()
        
        # Cek checkpoint best terlebih dahulu
        if checkpoints['best']:
            return checkpoints['best'][0]
        
        # Jika tidak ada best, cek checkpoint epoch dan ambil yang terbaik
        best_checkpoint = None
        best_loss = float('inf')
        
        for cp in checkpoints['epoch']:
            info = self.get_checkpoint_info(cp)
            try:
                loss = float(info.get('loss', float('inf')))
                if loss < best_loss:
                    best_loss = loss
                    best_checkpoint = cp
            except:
                continue
        
        # Jika tidak ada epoch atau best, gunakan latest
        if best_checkpoint is None and checkpoints['latest']:
            best_checkpoint = checkpoints['latest'][0]
            
        return best_checkpoint
    
    def load_checkpoint(self, model, checkpoint_path=None):
        """
        Muat checkpoint ke model
        
        Args:
            model: Model yang akan dimuat checkpoint
            checkpoint_path: Path ke checkpoint (jika None, gunakan best)
            
        Returns:
            Model yang sudah dimuat checkpoint
        """
        # Jika tidak ada path, gunakan best checkpoint
        if checkpoint_path is None:
            checkpoint_path = self.find_best_checkpoint()
            
        if checkpoint_path is None:
            self.logger.warning("⚠️ Tidak ada checkpoint yang ditemukan")
            return model
            
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            
            self.logger.success(f"✅ Model berhasil dimuat dari {checkpoint_path}")
            
            # Log informasi tambahan
            epoch = checkpoint.get('epoch', 'unknown')
            loss = checkpoint.get('loss', 'unknown')
            self.logger.info(f"📊 Epoch: {epoch}, Loss: {loss}")
            
            return model
        except Exception as e:
            self.logger.error(f"❌ Gagal memuat checkpoint: {str(e)}")
            return model