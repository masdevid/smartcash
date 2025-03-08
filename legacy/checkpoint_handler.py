# File: smartcash/handlers/checkpoint_handler.py
# Author: Alfrida Sabar
# Deskripsi: Handler untuk manajemen checkpoint model dengan penanganan error yang lebih baik

import os
import torch
import shutil
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import json
import traceback

from smartcash.utils.logger import SmartCashLogger

class CheckpointHandler:
    """
    Handler untuk manajemen checkpoint model dengan fitur:
    - Penyimpanan checkpoint secara robust dengan penanganan error
    - Checkpoint terbaik, terakhir, dan interval epoch
    - Metadata lengkap untuk resume training
    - Utility untuk melihat, membersihkan, dan memulihkan checkpoint
    """
    
    def __init__(
        self, 
        output_dir: str = 'runs/train/weights', 
        logger: Optional[SmartCashLogger] = None
    ):
        """
        Inisialisasi CheckpointHandler
        
        Args:
            output_dir: Direktori untuk menyimpan checkpoint
            logger: Logger kustom (opsional)
        """
        self.output_dir = Path(output_dir)
        self.history_file = self.output_dir / 'training_history.yaml'
        
        # Buat direktori jika belum ada
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logger
        self.logger = logger or SmartCashLogger(__name__)
        
        # Inisialisasi riwayat training
        self._init_training_history()
    
    def _init_training_history(self):
        """Inisialisasi file riwayat training jika belum ada"""
        try:
            if not self.history_file.exists():
                with open(self.history_file, 'w') as f:
                    yaml.safe_dump({
                        'total_runs': 0,
                        'runs': [],
                        'last_resume': None
                    }, f)
        except Exception as e:
            self.logger.error(f"❌ Gagal menginisialisasi riwayat training: {str(e)}")
            # Buat file kosong jika gagal untuk mencegah error berikutnya
            try:
                with open(self.history_file, 'w') as f:
                    f.write('{}')
            except:
                pass
    
    def generate_checkpoint_name(
        self, 
        config: Dict, 
        run_type: str = 'default',
        epoch: Optional[int] = None
    ) -> str:
        """
        Generate nama checkpoint dengan struktur yang konsisten
        
        Args:
            config: Konfigurasi model/training
            run_type: Tipe run ('best', 'latest', 'epoch')
            epoch: Nomor epoch (untuk run_type='epoch')
        
        Returns:
            String nama checkpoint unik
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Ekstrak informasi konfigurasi
        backbone = config.get('model', {}).get('backbone', 'default')
        dataset = config.get('data', {}).get('source', 'default')
        
        # Format epoch jika disediakan
        epoch_str = f"_epoch_{epoch}" if epoch is not None and run_type == 'epoch' else ""
        
        return f"smartcash_{backbone}_{dataset}_{run_type}{epoch_str}_{timestamp}.pth"
    
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        config: Dict,
        epoch: int,
        metrics: Dict,
        is_best: bool = False,
        save_optimizer: bool = True
    ) -> Dict[str, str]:
        """
        Simpan checkpoint model dengan metadata komprehensif
        
        Args:
            model: Model PyTorch
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            config: Konfigurasi training
            epoch: Epoch saat ini
            metrics: Metrik training
            is_best: Apakah ini model terbaik
            save_optimizer: Apakah menyimpan state optimizer
        
        Returns:
            Dict berisi path checkpoint yang disimpan
        """
        try:
            # Pastikan direktori checkpoint ada
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate nama file checkpoint
            run_type = 'best' if is_best else 'latest'
            checkpoint_name = self.generate_checkpoint_name(config, run_type)
            checkpoint_path = self.output_dir / checkpoint_name
            
            # Simpan checkpoint secara periodik berdasarkan epoch
            save_epoch_checkpoint = (epoch % config.get('training', {}).get('save_every', 5) == 0)
            epoch_checkpoint_path = None
            
            if save_epoch_checkpoint:
                epoch_name = self.generate_checkpoint_name(config, 'epoch', epoch)
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
            self._update_training_history(checkpoint_name, metrics, is_best, epoch)
            
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
                import tempfile
                temp_dir = tempfile.gettempdir()
                temp_checkpoint_path = Path(temp_dir) / f"smartcash_emergency_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
                
                # Simpan hanya state model untuk emergency
                emergency_state = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'timestamp': datetime.now().isoformat()
                }
                
                torch.save(emergency_state, temp_checkpoint_path)
                self.logger.warning(f"⚠️ Emergency checkpoint disimpan di: {temp_checkpoint_path}")
                
                return {
                    'path': str(temp_checkpoint_path),
                    'type': 'emergency',
                    'size': f"{temp_checkpoint_path.stat().st_size / (1024*1024):.2f} MB"
                }
            except Exception as e2:
                self.logger.error(f"❌ Gagal membuat emergency checkpoint: {str(e2)}")
                return {
                    'path': '',
                    'type': 'failed',
                    'error': str(e)
                }
    
    def _update_training_history(
        self, 
        checkpoint_name: str, 
        metrics: Dict, 
        is_best: bool,
        epoch: int
    ):
        """
        Update riwayat training dalam file YAML
        
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
        
        except Exception as e:
            self.logger.warning(f"⚠️ Gagal memperbarui riwayat training: {str(e)}")
    
    def load_checkpoint(
        self, 
        checkpoint_path: Optional[str] = None,
        device: Optional[torch.device] = None,
        model: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None
    ) -> Dict:
        """
        Muat checkpoint dengan dukungan berbagai opsi dan resume training
        
        Args:
            checkpoint_path: Path ke checkpoint (jika None, akan mengambil checkpoint terbaik)
            device: Perangkat untuk memuat model
            model: Model yang akan dimuat dengan weights (opsional)
            optimizer: Optimizer yang akan dimuat dengan state (opsional)
            scheduler: Scheduler yang akan dimuat dengan state (opsional)
            
        Returns:
            Dict berisi metadata dan state_dict
        """
        device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        try:
            # Cari checkpoint terbaik jika path tidak diberikan
            if checkpoint_path is None:
                checkpoint_path = self.find_best_checkpoint()
                if checkpoint_path is None:
                    self.logger.warning("⚠️ Tidak ditemukan checkpoint terbaik")
                    
                    # Cari checkpoint latest sebagai fallback
                    checkpoint_path = self.find_latest_checkpoint()
                    if checkpoint_path is None:
                        raise FileNotFoundError("Tidak ditemukan checkpoint yang valid")
            
            # Muat checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # Muat state ke model jika disediakan
            if model is not None and 'model_state_dict' in checkpoint:
                try:
                    model.load_state_dict(checkpoint['model_state_dict'])
                except Exception as e:
                    self.logger.error(f"❌ Gagal memuat state model: {str(e)}")
                    # Coba load dengan strict=False
                    self.logger.warning("⚠️ Mencoba memuat dengan strict=False...")
                    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            
            # Muat state ke optimizer jika disediakan
            if optimizer is not None and 'optimizer_state_dict' in checkpoint:
                try:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                except Exception as e:
                    self.logger.warning(f"⚠️ Gagal memuat state optimizer: {str(e)}")
            
            # Muat state ke scheduler jika disediakan
            if scheduler is not None and 'scheduler_state_dict' in checkpoint:
                try:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                except Exception as e:
                    self.logger.warning(f"⚠️ Gagal memuat state scheduler: {str(e)}")
            
            # Update riwayat resume
            self._update_resume_history(checkpoint_path)
            
            self.logger.success(f"📂 Checkpoint berhasil dimuat dari: {checkpoint_path}")
            
            # Tambahkan metadata berguna ke hasil
            checkpoint['loaded_from'] = checkpoint_path
            if 'epoch' not in checkpoint:
                checkpoint['epoch'] = 0
                
            return checkpoint
        
        except Exception as e:
            self.logger.error(f"❌ Gagal memuat checkpoint: {str(e)}")
            self.logger.error(f"📋 Detail stack trace: {traceback.format_exc()}")
            
            # Return default state yang tidak akan menyebabkan crash
            return {
                'epoch': 0,
                'model_state_dict': None,
                'optimizer_state_dict': None,
                'scheduler_state_dict': None,
                'metrics': {},
                'config': {},
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
    
    def _update_resume_history(self, checkpoint_path: str):
        """Update riwayat resume training"""
        try:
            with open(self.history_file, 'r') as f:
                history = yaml.safe_load(f) or {'total_runs': 0, 'runs': [], 'last_resume': None}
            
            history['last_resume'] = {
                'checkpoint': os.path.basename(checkpoint_path),
                'timestamp': datetime.now().isoformat()
            }
            
            with open(self.history_file, 'w') as f:
                yaml.safe_dump(history, f)
        except Exception as e:
            self.logger.warning(f"⚠️ Gagal memperbarui riwayat resume: {str(e)}")
    
    def find_best_checkpoint(self) -> Optional[str]:
        """
        Temukan checkpoint terbaik berdasarkan riwayat training
        
        Returns:
            Path ke checkpoint terbaik, atau None jika tidak ada
        """
        try:
            with open(self.history_file, 'r') as f:
                history = yaml.safe_load(f) or {'runs': []}
            
            # Urutkan berdasarkan metrik yang relevan
            best_runs = [
                run for run in history['runs'] 
                if run.get('is_best', False)
            ]
            
            if best_runs:
                # Ambil checkpoint terbaik terakhir
                best_run = max(best_runs, key=lambda x: x.get('timestamp', ''))
                best_checkpoint_name = best_run['checkpoint_name']
                best_checkpoint_path = self.output_dir / best_checkpoint_name
                
                # Pastikan file ada
                if best_checkpoint_path.exists():
                    return str(best_checkpoint_path)
                else:
                    self.logger.warning(f"⚠️ File checkpoint terbaik tidak ditemukan: {best_checkpoint_path}")
            
            # Jika tidak ada checkpoint terbaik valid, cari checkpoint lain
            return None
        
        except Exception as e:
            self.logger.error(f"❌ Gagal mencari checkpoint terbaik: {str(e)}")
            return None
    
    def find_latest_checkpoint(self) -> Optional[str]:
        """
        Temukan checkpoint terakhir berdasarkan waktu modifikasi
        
        Returns:
            Path ke checkpoint terakhir, atau None jika tidak ada
        """
        try:
            # Cari semua file checkpoint
            checkpoint_files = list(self.output_dir.glob('*.pth'))
            
            if not checkpoint_files:
                return None
            
            # Urutkan berdasarkan waktu modifikasi (terbaru dulu)
            latest_checkpoint = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
            
            return str(latest_checkpoint)
        
        except Exception as e:
            self.logger.error(f"❌ Gagal mencari checkpoint terakhir: {str(e)}")
            return None
    
    def list_checkpoints(self) -> Dict[str, List[Path]]:
        """
        Dapatkan daftar semua checkpoint yang tersedia
        
        Returns:
            Dict berisi list checkpoint berdasarkan tipe
        """
        try:
            checkpoints = {
                'best': list(self.output_dir.glob('*_best_*.pth')),
                'latest': list(self.output_dir.glob('*_latest_*.pth')),
                'epoch': list(self.output_dir.glob('*_epoch_*.pth')),
                'emergency': list(self.output_dir.glob('*emergency*.pth'))
            }
            
            # Sort berdasarkan waktu modifikasi (terbaru dulu)
            for category, files in checkpoints.items():
                checkpoints[category] = sorted(files, key=lambda x: x.stat().st_mtime, reverse=True)
            
            return checkpoints
        
        except Exception as e:
            self.logger.error(f"❌ Gagal mendapatkan daftar checkpoint: {str(e)}")
            return {'best': [], 'latest': [], 'epoch': [], 'emergency': []}
    
    def display_checkpoints(self):
        """
        Tampilkan daftar checkpoint yang tersedia dalam format yang mudah dibaca.
        """
        try:
            checkpoints = self.list_checkpoints()
            
            # Tampilkan hasil
            for category, files in checkpoints.items():
                if files:
                    print(f"📦 {category.capitalize()} Checkpoints:")
                    for f in files:
                        size_mb = f.stat().st_size / (1024 * 1024)
                        print(f"- {f.name} ({size_mb:.2f} MB, terakhir diubah: {datetime.fromtimestamp(f.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')})")
                    print()
            
            # Tampilkan riwayat resume terakhir
            try:
                with open(self.history_file, 'r') as f:
                    history = yaml.safe_load(f) or {}
                
                last_resume = history.get('last_resume')
                if last_resume:
                    print(f"🔄 Resume terakhir: {last_resume['checkpoint']} pada {last_resume['timestamp']}")
            except:
                pass
                
        except Exception as e:
            self.logger.error(f"❌ Gagal menampilkan checkpoint: {str(e)}")
    
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
            checkpoints = self.list_checkpoints()
            
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
                best_checkpoint = self.find_best_checkpoint()
                if best_checkpoint:
                    checkpoint_paths = [Path(best_checkpoint)]
                else:
                    self.logger.warning("⚠️ Tidak ditemukan checkpoint terbaik")
                    return []
            else:
                # Salin semua checkpoint best dan latest terbaru
                checkpoints = self.list_checkpoints()
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
                history_dest = os.path.join(drive_dir, self.history_file.name)
                shutil.copy2(self.history_file, history_dest)
                self.logger.info(f"📂 Riwayat training disalin ke {history_dest}")
            except Exception as e:
                self.logger.warning(f"⚠️ Gagal menyalin riwayat training: {str(e)}")
            
            return copied_paths
            
        except Exception as e:
            self.logger.error(f"❌ Gagal menyalin checkpoint ke Drive: {str(e)}")
            return []

    def get_checkpoint_path(self, checkpoint_path: Optional[str] = None) -> Optional[str]:
        """
        Dapatkan path checkpoint yang valid
        
        Args:
            checkpoint_path: Path ke checkpoint (opsional)
            
        Returns:
            Path checkpoint yang valid atau None jika tidak ditemukan
        """
        if checkpoint_path and os.path.exists(checkpoint_path):
            return checkpoint_path
            
        # Coba best checkpoint
        best_checkpoint = self.find_best_checkpoint()
        if best_checkpoint:
            return best_checkpoint
            
        # Fallback ke latest checkpoint
        latest_checkpoint = self.find_latest_checkpoint()
        return latest_checkpoint