# File: smartcash/handlers/checkpoint/checkpoint_saver.py
# Author: Alfrida Sabar
# Deskripsi: Komponen untuk penyimpanan model checkpoint

import os
import torch
import shutil
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List, Any, Union

from smartcash.utils.logger import SmartCashLogger
from smartcash.handlers.checkpoint.checkpoint_utils import generate_checkpoint_name
from smartcash.handlers.checkpoint.checkpoint_history import CheckpointHistory

class CheckpointSaver:
    """
    Komponen untuk penyimpanan model checkpoint dengan penanganan error yang robust.
    """
    
    def __init__(
        self,
        output_dir: Path,
        logger: Optional[SmartCashLogger] = None
    ):
        """
        Inisialisasi saver.
        
        Args:
            output_dir: Direktori untuk menyimpan checkpoint
            logger: Logger kustom (opsional)
        """
        self.output_dir = output_dir
        self.logger = logger or SmartCashLogger(__name__)
        
        # Buat direktori jika belum ada
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Inisialisasi history manager
        self.history = CheckpointHistory(output_dir, logger)
    
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        config: Dict,
        epoch: int,
        metrics: Dict,
        is_best: bool = False,
        save_optimizer: bool = True,
        run_type: str = None
    ) -> Dict[str, str]:
        """
        Simpan checkpoint model dengan metadata komprehensif.
        
        Args:
            model: Model PyTorch
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            config: Konfigurasi training
            epoch: Epoch saat ini
            metrics: Metrik training
            is_best: Apakah ini model terbaik
            save_optimizer: Apakah menyimpan state optimizer
            run_type: Tipe run (override 'best'/'latest')
            
        Returns:
            Dict berisi path checkpoint yang disimpan
        """
        try:
            # Pastikan direktori checkpoint ada
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Penentuan run_type berdasarkan is_best flag jika tidak diberikan
            if run_type is None:
                run_type = 'best' if is_best else 'latest'
                
            # Generate nama file checkpoint
            checkpoint_name = generate_checkpoint_name(config, run_type, epoch if run_type == 'epoch' else None)
            checkpoint_path = self.output_dir / checkpoint_name
            
            # Simpan checkpoint secara periodik berdasarkan epoch
            save_epoch_checkpoint = (epoch % config.get('training', {}).get('save_every', 5) == 0)
            epoch_checkpoint_path = None
            
            if save_epoch_checkpoint and run_type != 'epoch':
                epoch_name = generate_checkpoint_name(config, 'epoch', epoch)
                epoch_checkpoint_path = self.output_dir / epoch_name
            
            # Prepare state training untuk resume
            training_state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'metrics': metrics,
                'config': config,
                'timestamp': datetime.now().isoformat()
            }
            
            # Tambahkan state optimizer dan scheduler jika diminta
            if save_optimizer:
                training_state['optimizer_state_dict'] = optimizer.state_dict()
                # Simpan scheduler jika bukan None
                if scheduler is not None:
                    try:
                        training_state['scheduler_state_dict'] = scheduler.state_dict()
                    except AttributeError:
                        # Beberapa scheduler tidak memiliki state_dict()
                        pass
            
            # Simpan checkpoint
            torch.save(training_state, checkpoint_path)
            
            # Simpan juga sebagai checkpoint epoch jika diperlukan
            if save_epoch_checkpoint and epoch_checkpoint_path:
                shutil.copy2(checkpoint_path, epoch_checkpoint_path)
            
            # Update riwayat training
            self.history.update_training_history(checkpoint_name, metrics, is_best, epoch)
            
            self.logger.success(f"💾 Checkpoint disimpan: {checkpoint_name}")
            
            result = {
                'path': str(checkpoint_path),
                'type': run_type,
                'size': f"{checkpoint_path.stat().st_size / (1024*1024):.2f} MB"
            }
            
            if save_epoch_checkpoint and epoch_checkpoint_path:
                result['epoch_path'] = str(epoch_checkpoint_path)
            
            return result
        
        except Exception as e:
            self.logger.error(f"❌ Gagal menyimpan checkpoint: {str(e)}")
            self.logger.error(f"📋 Detail stack trace: {traceback.format_exc()}")
            
            # Fallback: coba simpan di temporary directory
            try:
                # Gunakan temporary file dengan extension yang benar
                with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp:
                    temp_checkpoint_path = tmp.name
                
                # Simpan hanya state model untuk emergency
                emergency_state = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'timestamp': datetime.now().isoformat()
                }
                
                torch.save(emergency_state, temp_checkpoint_path)
                self.logger.warning(f"⚠️ Emergency checkpoint disimpan di: {temp_checkpoint_path}")
                
                # Coba pindahkan ke direktori output
                try:
                    emergency_name = f"smartcash_emergency_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
                    emergency_path = self.output_dir / emergency_name
                    shutil.move(temp_checkpoint_path, emergency_path)
                    temp_checkpoint_path = str(emergency_path)
                    self.logger.info(f"✅ Emergency checkpoint dipindahkan ke: {emergency_path}")
                except:
                    # Jika gagal memindahkan, gunakan temporary file
                    pass
                
                return {
                    'path': temp_checkpoint_path,
                    'type': 'emergency',
                    'size': f"{Path(temp_checkpoint_path).stat().st_size / (1024*1024):.2f} MB"
                }
            except Exception as e2:
                self.logger.error(f"❌ Gagal membuat emergency checkpoint: {str(e2)}")
                return {
                    'path': '',
                    'type': 'failed',
                    'error': str(e)
                }
    
    def save_multiple_checkpoints(
        self,
        state_dict: Dict[str, Any],
        config: Dict,
        outputs: List[str] = ['best', 'latest']
    ) -> Dict[str, str]:
        """
        Simpan state model ke multiple output files (best, latest, etc).
        
        Args:
            state_dict: State model dan metadata
            config: Konfigurasi model
            outputs: List tipe output ('best', 'latest', 'epoch')
            
        Returns:
            Dict berisi path checkpoint untuk setiap tipe
        """
        results = {}
        
        try:
            for output_type in outputs:
                # Generate nama file
                if output_type == 'epoch' and 'epoch' in state_dict:
                    checkpoint_name = generate_checkpoint_name(config, output_type, state_dict['epoch'])
                else:
                    checkpoint_name = generate_checkpoint_name(config, output_type)
                    
                checkpoint_path = self.output_dir / checkpoint_name
                
                # Simpan checkpoint
                torch.save(state_dict, checkpoint_path)
                
                results[output_type] = str(checkpoint_path)
                
            self.logger.success(f"✅ Multiple checkpoints disimpan: {', '.join(outputs)}")
                
            return results
                
        except Exception as e:
            self.logger.error(f"❌ Gagal menyimpan multiple checkpoints: {str(e)}")
            return {'error': str(e)}
    
    def cleanup_checkpoints(
        self, 
        max_checkpoints: int = 5,
        keep_best: bool = True,
        keep_latest: bool = True,
        max_epochs: int = 5
    ) -> List[str]:
        """
        Bersihkan checkpoint lama dengan opsi fleksibel
        
        Args:
            max_checkpoints: Jumlah maksimal checkpoint per kategori
            keep_best: Pertahankan semua checkpoint terbaik
            keep_latest: Pertahankan checkpoint latest terakhir
            max_epochs: Jumlah maksimal checkpoint epoch yang disimpan
            
        Returns:
            List path checkpoint yang dihapus
        """
        try:
            deleted_paths = []
            
            # Dapatkan semua checkpoint
            from smartcash.checkpoint.checkpoint_finder import CheckpointFinder
            finder = CheckpointFinder(self.output_dir, self.logger)
            checkpoints = finder.list_checkpoints()
            
            # Bersihkan checkpoint epoch (simpan hanya max_epochs terakhir)
            if len(checkpoints['epoch']) > max_epochs:
                to_delete = checkpoints['epoch'][max_epochs:]
                for checkpoint in to_delete:
                    try:
                        checkpoint.unlink()
                        deleted_paths.append(str(checkpoint))
                        self.logger.info(f"🗑️ Epoch checkpoint dihapus: {checkpoint.name}")
                    except Exception as e:
                        self.logger.warning(f"⚠️ Gagal menghapus {checkpoint.name}: {str(e)}")
            
            # Bersihkan checkpoint latest (simpan hanya yang terbaru)
            if keep_latest and len(checkpoints['latest']) > 1:
                to_delete = checkpoints['latest'][1:]
                for checkpoint in to_delete:
                    try:
                        checkpoint.unlink()
                        deleted_paths.append(str(checkpoint))
                        self.logger.info(f"🗑️ Latest checkpoint dihapus: {checkpoint.name}")
                    except Exception as e:
                        self.logger.warning(f"⚠️ Gagal menghapus {checkpoint.name}: {str(e)}")
            
            # Bersihkan checkpoint best (simpan hanya yang terbaru jika tidak keep_best)
            if not keep_best and len(checkpoints['best']) > 1:
                to_delete = checkpoints['best'][1:]
                for checkpoint in to_delete:
                    try:
                        checkpoint.unlink()
                        deleted_paths.append(str(checkpoint))
                        self.logger.info(f"🗑️ Best checkpoint dihapus: {checkpoint.name}")
                    except Exception as e:
                        self.logger.warning(f"⚠️ Gagal menghapus {checkpoint.name}: {str(e)}")
            
            self.logger.info(f"🧹 Pembersihan selesai: {len(deleted_paths)} checkpoint dihapus")
            return deleted_paths
            
        except Exception as e:
            self.logger.error(f"❌ Gagal membersihkan checkpoint: {str(e)}")
            return []
    
    def copy_to_drive(
        self, 
        drive_dir: str = '/content/drive/MyDrive/SmartCash/checkpoints',
        best_only: bool = False
    ) -> List[str]:
        """
        Salin checkpoint ke Google Drive
        
        Args:
            drive_dir: Direktori tujuan di Google Drive
            best_only: Hanya salin checkpoint terbaik
            
        Returns:
            List path checkpoint yang disalin
        """
        try:
            # Periksa apakah running di Colab dan Drive terpasang
            if not os.path.exists('/content/drive'):
                self.logger.warning("⚠️ Google Drive tidak terpasang")
                return []
                
            # Buat direktori jika belum ada
            os.makedirs(drive_dir, exist_ok=True)
            
            # Dapatkan checkpoint yang akan disalin
            if best_only:
                from smartcash.checkpoint.checkpoint_finder import CheckpointFinder
                finder = CheckpointFinder(self.output_dir, self.logger)
                best_checkpoint = finder.find_best_checkpoint()
                
                if best_checkpoint:
                    checkpoint_paths = [Path(best_checkpoint)]
                else:
                    self.logger.warning("⚠️ Tidak ditemukan checkpoint terbaik")
                    return []
            else:
                # Salin semua checkpoint best dan latest terbaru
                from smartcash.checkpoint.checkpoint_finder import CheckpointFinder
                finder = CheckpointFinder(self.output_dir, self.logger)
                checkpoints = finder.list_checkpoints()
                checkpoint_paths = checkpoints['best'][:1] + checkpoints['latest'][:1]
            
            # Salin checkpoint
            copied_paths = []
            for checkpoint in checkpoint_paths:
                dest_path = os.path.join(drive_dir, checkpoint.name)
                try:
                    shutil.copy2(checkpoint, dest_path)
                    copied_paths.append(dest_path)
                    self.logger.info(f"📂 Checkpoint disalin: {checkpoint.name}")
                except Exception as e:
                    self.logger.warning(f"⚠️ Gagal menyalin {checkpoint.name}: {str(e)}")
            
            # Salin juga file riwayat
            try:
                history_dest = os.path.join(drive_dir, self.history.history_file.name)
                shutil.copy2(self.history.history_file, history_dest)
                self.logger.info(f"📂 Riwayat training disalin ke {history_dest}")
            except Exception as e:
                self.logger.warning(f"⚠️ Gagal menyalin riwayat training: {str(e)}")
            
            self.logger.success(f"✅ {len(copied_paths)} checkpoint berhasil disalin ke {drive_dir}")
            return copied_paths
            
        except Exception as e:
            self.logger.error(f"❌ Gagal menyalin checkpoint ke Drive: {str(e)}")
            return []