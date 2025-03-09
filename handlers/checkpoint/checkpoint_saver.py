# File: smartcash/handlers/checkpoint/checkpoint_saver.py
# Author: Alfrida Sabar
# Deskripsi: Komponen untuk penyimpanan model checkpoint (Diringkas)

import os
import torch
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

from smartcash.utils.logger import get_logger
from smartcash.handlers.checkpoint.checkpoint_utils import generate_checkpoint_name

class CheckpointSaver:
    """Komponen untuk penyimpanan model checkpoint dengan penanganan error yang robust."""
    
    def __init__(self, output_dir: Path, logger=None):
        """
        Inisialisasi saver.
        
        Args:
            output_dir: Direktori untuk menyimpan checkpoint
            logger: Logger kustom (opsional)
        """
        self.output_dir = output_dir
        self.logger = logger or get_logger("checkpoint_saver")
        
        # Buat direktori jika belum ada
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Dapatkan history manager
        from smartcash.handlers.checkpoint.checkpoint_history import CheckpointHistory
        self.history = CheckpointHistory(output_dir, logger)
    
    def save_checkpoint(
        self,
        model,
        optimizer,
        scheduler,
        config,
        epoch,
        metrics,
        is_best=False,
        save_optimizer=True
    ) -> Dict[str, str]:
        """
        Simpan checkpoint model dengan metadata komprehensif.
        """
        try:
            # Pastikan direktori checkpoint ada
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Tipe run berdasarkan is_best
            run_type = 'best' if is_best else 'latest'
                
            # Generate nama file checkpoint
            checkpoint_name = generate_checkpoint_name(config, run_type, epoch if run_type == 'epoch' else None)
            checkpoint_path = self.output_dir / checkpoint_name
            
            # Tentukan apakah perlu menyimpan checkpoint epoch
            save_epoch_checkpoint = (epoch % config.get('training', {}).get('save_every', 5) == 0)
            epoch_checkpoint_path = None
            
            if save_epoch_checkpoint and run_type != 'epoch':
                epoch_name = generate_checkpoint_name(config, 'epoch', epoch)
                epoch_checkpoint_path = self.output_dir / epoch_name
            
            # Prepare state untuk disimpan
            training_state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'metrics': metrics,
                'config': config,
                'timestamp': datetime.now().isoformat()
            }
            
            # Tambahkan optimizer dan scheduler state
            if save_optimizer:
                training_state['optimizer_state_dict'] = optimizer.state_dict()
                if scheduler is not None:
                    try:
                        training_state['scheduler_state_dict'] = scheduler.state_dict()
                    except AttributeError:
                        pass
            
            # Simpan checkpoint
            torch.save(training_state, checkpoint_path)
            
            # Simpan juga sebagai checkpoint epoch jika diperlukan
            if save_epoch_checkpoint and epoch_checkpoint_path:
                shutil.copy2(checkpoint_path, epoch_checkpoint_path)
            
            # Update riwayat training
            self.history.update_training_history(checkpoint_name, metrics, is_best, epoch)
            
            self.logger.info(f"💾 Checkpoint disimpan: {checkpoint_name}")
            
            return {
                'path': str(checkpoint_path),
                'type': run_type,
                'epoch_path': str(epoch_checkpoint_path) if save_epoch_checkpoint and epoch_checkpoint_path else None
            }
            
        except Exception as e:
            self.logger.error(f"❌ Gagal menyimpan checkpoint: {str(e)}")
            
            # Coba buat emergency checkpoint
            try:
                emergency_name = f"smartcash_emergency_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
                emergency_path = self.output_dir / emergency_name
                
                # Simpan hanya state model
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()}, emergency_path)
                self.logger.warning(f"⚠️ Emergency checkpoint disimpan di: {emergency_path}")
                
                return {'path': str(emergency_path), 'type': 'emergency'}
            except:
                return {'path': '', 'type': 'failed', 'error': str(e)}
    
    def cleanup_checkpoints(
        self, 
        max_checkpoints=5,
        keep_best=True,
        keep_latest=True,
        max_epochs=5
    ) -> List[str]:
        """
        Bersihkan checkpoint lama dengan opsi fleksibel
        """
        try:
            deleted_paths = []
            
            # Dapatkan semua checkpoint
            from smartcash.handlers.checkpoint.checkpoint_finder import CheckpointFinder
            finder = CheckpointFinder(self.output_dir, self.logger)
            checkpoints = finder.list_checkpoints()
            
            # Bersihkan checkpoint epoch
            if len(checkpoints['epoch']) > max_epochs:
                to_delete = checkpoints['epoch'][max_epochs:]
                for checkpoint in to_delete:
                    try:
                        checkpoint.unlink()
                        deleted_paths.append(str(checkpoint))
                    except Exception as e:
                        self.logger.warning(f"⚠️ Gagal menghapus {checkpoint.name}: {str(e)}")
            
            # Bersihkan checkpoint latest
            if keep_latest and len(checkpoints['latest']) > 1:
                to_delete = checkpoints['latest'][1:]
                for checkpoint in to_delete:
                    try:
                        checkpoint.unlink()
                        deleted_paths.append(str(checkpoint))
                    except Exception as e:
                        self.logger.warning(f"⚠️ Gagal menghapus {checkpoint.name}: {str(e)}")
            
            # Bersihkan checkpoint best
            if not keep_best and len(checkpoints['best']) > 1:
                to_delete = checkpoints['best'][1:]
                for checkpoint in to_delete:
                    try:
                        checkpoint.unlink()
                        deleted_paths.append(str(checkpoint))
                    except Exception as e:
                        self.logger.warning(f"⚠️ Gagal menghapus {checkpoint.name}: {str(e)}")
            
            self.logger.info(f"🧹 Pembersihan selesai: {len(deleted_paths)} checkpoint dihapus")
            return deleted_paths
            
        except Exception as e:
            self.logger.error(f"❌ Gagal membersihkan checkpoint: {str(e)}")
            return []
    
    def copy_to_drive(
        self, 
        drive_dir='/content/drive/MyDrive/SmartCash/checkpoints',
        best_only=False
    ) -> List[str]:
        """
        Salin checkpoint ke Google Drive
        """
        try:
            # Periksa apakah running di Colab dan Drive terpasang
            if not os.path.exists('/content/drive'):
                self.logger.warning("⚠️ Google Drive tidak terpasang")
                return []
                
            # Buat direktori jika belum ada
            os.makedirs(drive_dir, exist_ok=True)
            
            # Dapatkan checkpoint yang akan disalin
            copied_paths = []
            
            if best_only:
                from smartcash.handlers.checkpoint.checkpoint_finder import CheckpointFinder
                finder = CheckpointFinder(self.output_dir, self.logger)
                best_checkpoint = finder.find_best_checkpoint()
                
                if best_checkpoint:
                    checkpoint_paths = [Path(best_checkpoint)]
                else:
                    return []
            else:
                # Salin checkpoint best dan latest terbaru
                from smartcash.handlers.checkpoint.checkpoint_finder import CheckpointFinder
                finder = CheckpointFinder(self.output_dir, self.logger)
                checkpoints = finder.list_checkpoints()
                checkpoint_paths = checkpoints['best'][:1] + checkpoints['latest'][:1]
            
            # Salin checkpoint
            for checkpoint in checkpoint_paths:
                dest_path = os.path.join(drive_dir, checkpoint.name)
                shutil.copy2(checkpoint, dest_path)
                copied_paths.append(dest_path)
                
            self.logger.info(f"✅ {len(copied_paths)} checkpoint berhasil disalin ke {drive_dir}")
            return copied_paths
            
        except Exception as e:
            self.logger.error(f"❌ Gagal menyalin checkpoint ke Drive: {str(e)}")
            return []