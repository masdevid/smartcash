"""
File: smartcash/handlers/ui_handlers/training_execution_handlers.py
Author: Alfrida Sabar
Deskripsi: Handler untuk komponen UI eksekusi training model, menangani proses training,
          visualisasi progress, dan callbacks.
"""

import gc
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import threading
import os
import shutil
from IPython.display import display, clear_output, HTML
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable

# Import common utilities
from smartcash.handlers.ui_handlers.common_utils import memory_manager, is_colab, display_gpu_info

class TrainingMetricsTracker:
    """Kelas untuk mencatat metrics selama training."""
    
    def __init__(self):
        """Inisialisasi tracker metrics."""
        self.reset()
    
    def reset(self):
        """Reset semua metrics tracking."""
        self.training_metrics = {
            'train_loss': [],
            'val_loss': [],
            'lr': [],
            'epochs': [],
            'best_epoch': 0,
            'best_val_loss': float('inf'),
            'is_training': False,
            'start_time': None,
            'last_update': None,
            'gpu_memory_history': []
        }
    
    def set_training_status(self, is_training):
        """Set status training."""
        self.training_metrics['is_training'] = is_training
        if is_training:
            self.training_metrics['start_time'] = time.time()
        
    def record_metrics(self, epoch, metrics):
        """
        Catat metrics dari callbacks pipeline.
        
        Args:
            epoch: Epoch saat ini
            metrics: Dictionary metrics yang dicatat
        """
        # Cek apakah epoch sudah ada di training_metrics (untuk mencegah duplikasi)
        if epoch in self.training_metrics['epochs']:
            idx = self.training_metrics['epochs'].index(epoch)
            # Update nilai yang ada
            self.training_metrics['train_loss'][idx] = metrics.get('train_loss', self.training_metrics['train_loss'][idx])
            self.training_metrics['val_loss'][idx] = metrics.get('val_loss', self.training_metrics['val_loss'][idx])
            if 'lr' in metrics and self.training_metrics['lr'] and len(self.training_metrics['lr']) > idx:
                self.training_metrics['lr'][idx] = metrics.get('lr', self.training_metrics['lr'][idx])
        else:
            # Tambah epoch baru
            self.training_metrics['epochs'].append(epoch)
            self.training_metrics['train_loss'].append(metrics.get('train_loss', 0))
            self.training_metrics['val_loss'].append(metrics.get('val_loss', 0))
            self.training_metrics['lr'].append(metrics.get('lr', 0))
        
        # Update best metrics
        if metrics.get('val_loss', float('inf')) < self.training_metrics['best_val_loss']:
            self.training_metrics['best_val_loss'] = metrics.get('val_loss', float('inf'))
            self.training_metrics['best_epoch'] = epoch
        
        # Catat GPU memory jika tersedia
        if torch.cuda.is_available():
            self.training_metrics['gpu_memory_history'].append({
                'epoch': epoch,
                'allocated': torch.cuda.memory_allocated() / (1024**2),
                'reserved': torch.cuda.memory_reserved() / (1024**2)
            })
        
        self.training_metrics['last_update'] = time.time()
        
    def get_metrics(self):
        """Dapatkan semua metrics training."""
        return self.training_metrics

def metrics_callback(epoch, metrics, metrics_tracker, ui_components, update_functions):
    """
    Callback function untuk mencatat metrics dari pipeline training.
    
    Args:
        epoch: Current epoch
        metrics: Dictionary metrics yang dicatat
        metrics_tracker: TrainingMetricsTracker instance
        ui_components: Dictionary UI components
        update_functions: Dictionary dengan fungsi update UI
    """
    with memory_manager():
        # Catat metrics
        metrics_tracker.record_metrics(epoch, metrics)
        
        # Update UI
        if 'update_plot' in update_functions:
            update_functions['update_plot'](ui_components, metrics_tracker)
        
        if 'update_metrics_table' in update_functions:
            update_functions['update_metrics_table'](ui_components, metrics_tracker)
        
        if 'update_status' in update_functions:
            update_functions['update_status'](ui_components, metrics_tracker)
def update_plot(ui_components, metrics_tracker):
    """
    Update plot dengan data metrics terbaru.
    
    Args:
        ui_components: Dictionary UI components
        metrics_tracker: TrainingMetricsTracker instance
    """
    # Validate required UI components
    if 'live_plot_tab' not in ui_components:
        return
    
    metrics = metrics_tracker.get_metrics()
    
    with ui_components['live_plot_tab']:
        clear_output(wait=True)
        
        if not metrics['train_loss']:
            print("📊 Belum ada data training untuk divisualisasikan")
            return
        
        try:
            # Setup figure dengan dua subplot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Plot training & validation loss
            epochs = metrics['epochs']
            train_loss = metrics['train_loss']
            val_loss = metrics['val_loss']
            
            ax1.plot(epochs, train_loss, 'bo-', label='Training Loss')
            ax1.plot(epochs, val_loss, 'ro-', label='Validation Loss')
            
            # Highlight best epoch
            best_epoch = metrics['best_epoch']
            if best_epoch in epochs:
                idx = epochs.index(best_epoch)
                best_loss = val_loss[idx]
                
                ax1.scatter([best_epoch], [best_loss], c='gold', s=150, zorder=5, 
                          label=f'Best Model (Val Loss: {best_loss:.4f})')
            
            ax1.set_title('Training & Validation Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.grid(True, linestyle='--', alpha=0.7)
            ax1.legend()
            
            # Plot learning rate
            if metrics['lr']:
                ax2.plot(epochs, metrics['lr'], 'go-', label='Learning Rate')
                ax2.set_title('Learning Rate Schedule')
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('Learning Rate')
                ax2.grid(True, linestyle='--', alpha=0.7)
                ax2.set_yscale('log')
                ax2.legend()
                
            plt.tight_layout()
            plt.show()
            
            # Tampilkan tabel ringkasan
            if len(metrics['epochs']) > 0:
                # Ambil 5 epoch terakhir saja
                last_n = min(5, len(metrics['epochs']))
                
                metrics_data = {
                    'Epoch': metrics['epochs'][-last_n:],
                    'Train Loss': metrics['train_loss'][-last_n:],
                    'Val Loss': metrics['val_loss'][-last_n:]
                }
                
                if metrics['lr'] and len(metrics['lr']) >= last_n:
                    metrics_data['Learning Rate'] = metrics['lr'][-last_n:]
                    
                metrics_df = pd.DataFrame(metrics_data)
                
                # Define function to highlight the best epoch
                def highlight_best(row):
                    is_best = row['Epoch'] == metrics['best_epoch']
                    return ['background-color: #d4f7e7' if is_best else '' for _ in row]
                
                # Display styled dataframe
                display(metrics_df.style.apply(highlight_best, axis=1).format({
                    'Train Loss': '{:.4f}',
                    'Val Loss': '{:.4f}',
                    'Learning Rate': '{:.6f}'
                }))
                
        except Exception as e:
            print(f"❌ Error saat update plot: {str(e)}")
            # Minimal fallback
            try:
                plt.figure(figsize=(10, 6))
                plt.plot(metrics['epochs'], metrics['train_loss'], 'b-', label='Train Loss')
                plt.plot(metrics['epochs'], metrics['val_loss'], 'r-', label='Val Loss')
                plt.title('Training Progress')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()
                plt.grid(True)
                plt.show()
            except:
                print("❌ Tidak dapat membuat plot minimal")

def update_metrics_table(ui_components, metrics_tracker):
    """
    Update metrics table untuk menampilkan detail metrics training.
    
    Args:
        ui_components: Dictionary UI components
        metrics_tracker: TrainingMetricsTracker instance
    """
    # Validate required UI components
    if 'metrics_tab' not in ui_components:
        return
        
    metrics = metrics_tracker.get_metrics()
    
    with ui_components['metrics_tab']:
        clear_output(wait=True)
        
        if not metrics['epochs']:
            print("📊 Belum ada data training untuk ditampilkan")
            return
        
        try:
            # Prepare metrics data
            metrics_data = {
                'Epoch': metrics['epochs'],
                'Train Loss': metrics['train_loss'],
                'Val Loss': metrics['val_loss']
            }
            
            if metrics['lr']:
                metrics_data['Learning Rate'] = metrics['lr']
            
            # Create DataFrame
            metrics_df = pd.DataFrame(metrics_data)
            
            # Calculate improvement from previous epoch
            if len(metrics_df) > 1:
                metrics_df['Train Diff'] = metrics_df['Train Loss'].diff() * -1
                metrics_df['Val Diff'] = metrics_df['Val Loss'].diff() * -1
                
                # Mark best epoch
                metrics_df['Best'] = [epoch == metrics['best_epoch'] for epoch in metrics_df['Epoch']]
            
            # Display styled table
            styled_df = metrics_df.style.format({
                'Train Loss': '{:.4f}',
                'Val Loss': '{:.4f}',
                'Train Diff': '{:+.4f}',
                'Val Diff': '{:+.4f}',
                'Learning Rate': '{:.6f}'
            })
            
            # Add color highlighting
            if 'Train Diff' in metrics_df.columns:
                styled_df = styled_df.background_gradient(
                    subset=['Train Diff', 'Val Diff'], 
                    cmap='RdYlGn',
                    vmin=-0.05,
                    vmax=0.05
                )
            
            # Highlight best epoch
            def highlight_best(row):
                if 'Best' in row and row['Best']:
                    return ['background-color: #d4f7e7'] * len(row)
                return [''] * len(row)
            
            styled_df = styled_df.apply(highlight_best, axis=1)
            
            # Hide the Best column if it exists
            if 'Best' in metrics_df.columns:
                styled_df = styled_df.hide(columns=['Best'])
            
            display(styled_df)
            
            # Show summary statistics
            print("\n📊 Statistik Ringkasan:")
            summary = {
                'Minimum': [
                    min(metrics['train_loss']),
                    min(metrics['val_loss'])
                ],
                'Maximum': [
                    max(metrics['train_loss']),
                    max(metrics['val_loss'])
                ],
                'Mean': [
                    sum(metrics['train_loss']) / len(metrics['train_loss']),
                    sum(metrics['val_loss']) / len(metrics['val_loss'])
                ],
                'Last': [
                    metrics['train_loss'][-1],
                    metrics['val_loss'][-1]
                ],
                'Best': [
                    '-',
                    min(metrics['val_loss'])
                ]
            }
            
            summary_df = pd.DataFrame(summary, index=['Train Loss', 'Val Loss'])
            display(summary_df.style.format('{:.4f}'))
            
            # Tampilkan statistik GPU jika tersedia
            if metrics['gpu_memory_history'] and torch.cuda.is_available():
                print("\n🔥 GPU Memory Usage:")
                
                # Ambil 5 entri terakhir untuk ditampilkan
                recent_history = metrics['gpu_memory_history'][-5:]
                
                gpu_data = {
                    'Epoch': [item['epoch'] for item in recent_history],
                    'Allocated (MB)': [item['allocated'] for item in recent_history],
                    'Reserved (MB)': [item['reserved'] for item in recent_history]
                }
                
                gpu_df = pd.DataFrame(gpu_data)
                
                display(gpu_df.style.format({
                    'Allocated (MB)': '{:.1f}',
                    'Reserved (MB)': '{:.1f}'
                }))
                
                # Tambahkan info memory saat ini
                current_allocated = torch.cuda.memory_allocated() / (1024**2)
                current_reserved = torch.cuda.memory_reserved() / (1024**2)
                total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**2)
                
                print(f"\nMemory saat ini: {current_allocated:.1f}MB / {total_memory:.1f}MB ({(current_allocated/total_memory)*100:.1f}%)")
            
        except Exception as e:
            print(f"❌ Error saat update metrics table: {str(e)}")
            # Minimal fallback
            print("📋 Training metrics:")
            print(f"• Epochs: {len(metrics['epochs'])}")
            print(f"• Best epoch: {metrics['best_epoch']} (Val loss: {metrics['best_val_loss']:.4f})")
            
            if metrics['epochs']:
                print(f"• Latest epoch: {metrics['epochs'][-1]}")
                print(f"• Latest train loss: {metrics['train_loss'][-1]:.4f}")
                print(f"• Latest val loss: {metrics['val_loss'][-1]:.4f}")

def update_status(ui_components, metrics_tracker):
    """
    Update tampilan status training.
    
    Args:
        ui_components: Dictionary UI components
        metrics_tracker: TrainingMetricsTracker instance
    """
    # Validate required UI components
    required_components = ['status_tab', 'status_text', 'progress_bar']
    missing = [comp for comp in required_components if comp not in ui_components]
    if missing:
        return
    
    metrics = metrics_tracker.get_metrics()
    
    with ui_components['status_tab']:
        clear_output(wait=True)
        
        # Check if training is active
        if metrics['is_training']:
            # Get current stats
            current_epoch = metrics['epochs'][-1] if metrics['epochs'] else 0
            total_epochs = 30  # Default value, should be updated from config
            progress_percent = (current_epoch / total_epochs) * 100 if total_epochs > 0 else 0
            
            # Update progress bar
            ui_components['progress_bar'].value = int(progress_percent)
            
            # Calculate timing information
            elapsed = time.time() - metrics['start_time'] if metrics['start_time'] else 0
            epoch_time = elapsed / max(1, current_epoch)  # Avoid division by zero
            remaining_epochs = total_epochs - current_epoch
            eta = remaining_epochs * epoch_time if remaining_epochs > 0 else 0
            
            elapsed_str = str(timedelta(seconds=int(elapsed)))
            eta_str = str(timedelta(seconds=int(eta)))
            
            # Update status text
            ui_components['status_text'].value = (
                f"<p><b>Status:</b> <span style='color:green'>Training</span></p>"
                f"<p><b>Epoch:</b> {current_epoch}/{total_epochs} (<b>{progress_percent:.1f}%</b>)</p>"
                f"<p><b>Best Val Loss:</b> <span style='color:blue'>{metrics['best_val_loss']:.4f}</span> (Epoch {metrics['best_epoch']})</p>"
                f"<p><b>Waktu berjalan:</b> {elapsed_str}</p>"
                f"<p><b>Estimasi selesai:</b> {eta_str}</p>"
            )
            
            # GPU stats if available
            if torch.cuda.is_available():
                gpu_memory_allocated = torch.cuda.memory_allocated() / (1024**2)
                gpu_memory_reserved = torch.cuda.memory_reserved() / (1024**2)
                
                ui_components['status_text'].value += (
                    f"<p><b>GPU Memory:</b> {gpu_memory_allocated:.1f}MB (allocated) / "
                    f"{gpu_memory_reserved:.1f}MB (reserved)</p>"
                )
            
            # Display hardware info
            if torch.cuda.is_available():
                gpu_memory_allocated = torch.cuda.memory_allocated() / (1024**3)
                gpu_memory_reserved = torch.cuda.memory_reserved() / (1024**3)
                
                print(f"💻 Info Hardware:")
                print(f"  • GPU: {torch.cuda.get_device_name(0)}")
                print(f"  • VRAM: {gpu_memory_allocated:.2f}GB (terpakai) / {gpu_memory_reserved:.2f}GB (total)")
                print(f"  • CUDA Version: {torch.version.cuda}")
            else:
                print(f"💻 Info Hardware: CPU Only")
                
            # Display checkpoints information if available
            try:
                from pathlib import Path
                checkpoints_dir = Path('runs/train/weights')
                if checkpoints_dir.exists():
                    checkpoint_files = list(checkpoints_dir.glob('*.pt')) + list(checkpoints_dir.glob('*.pth'))
                    if checkpoint_files:
                        print("\n📦 Checkpoint Training:")
                        # Group by type
                        best_ckpts = [f for f in checkpoint_files if 'best' in f.name.lower()]
                        last_ckpts = [f for f in checkpoint_files if 'last' in f.name.lower()]
                        if best_ckpts:
                            best = sorted(best_ckpts, key=lambda x: x.stat().st_mtime, reverse=True)[0]
                            size_mb = best.stat().st_size / (1024*1024)
                            print(f"• Best: {best.name} ({size_mb:.1f} MB)")
                        if last_ckpts:
                            last = sorted(last_ckpts, key=lambda x: x.stat().st_mtime, reverse=True)[0]
                            size_mb = last.stat().st_size / (1024*1024)
                            print(f"• Latest: {last.name} ({size_mb:.1f} MB)")
            except Exception as e:
                print(f"⚠️ Tidak dapat menampilkan info checkpoint: {str(e)}")
                
        else:
            # Update status for non-training state
            ui_components['status_text'].value = (
                f"<p><b>Status:</b> <span style='color:gray'>Idle</span></p>"
            )
            
            if metrics['best_val_loss'] < float('inf'):
                ui_components['status_text'].value += (
                    f"<p><b>Best Val Loss:</b> <span style='color:blue'>{metrics['best_val_loss']:.4f}</span> (Epoch {metrics['best_epoch']})</p>"
                )
            
            # Display training tips
            print("\n💡 Tips Training:")
            print("• Gunakan batch size yang lebih kecil jika mengalami out-of-memory")
            print("• Opsi 'Resume dari checkpoint' untuk melanjutkan training yang terhenti")
            print("• Jalankan notebook di environment dengan GPU untuk performa lebih baik")
def run_training(ui_components, pipeline, dataloaders, resume_from_checkpoint, metrics_tracker, checkpoint_handler, config, logger):
    """
    Jalankan proses training model.
    
    Args:
        ui_components: Dictionary UI components
        pipeline: TrainingPipeline instance
        dataloaders: Dictionary dataloaders
        resume_from_checkpoint: Boolean yang menunjukkan apakah melanjutkan dari checkpoint
        metrics_tracker: TrainingMetricsTracker instance
        checkpoint_handler: CheckpointHandler instance
        config: Dictionary konfigurasi
        logger: Logger untuk mencatat aktivitas
        
    Returns:
        Boolean yang menunjukkan keberhasilan training
    """
    if not pipeline or not dataloaders:
        logger.error("❌ Pipeline training belum diinisialisasi")
        return False
    
    # Reset status training
    metrics_tracker.set_training_status(True)
    
    # Jika tidak melanjutkan training, reset metrics history
    if not resume_from_checkpoint:
        metrics_tracker.reset()
        metrics_tracker.set_training_status(True)
    
    try:
        # Dapatkan parameter training dari config
        epochs = config.get('training', {}).get('epochs', 30)
        batch_size = config.get('training', {}).get('batch_size', 16)
        save_every = config.get('training', {}).get('save_every', 5)
        
        checkpoint_path = None
        if resume_from_checkpoint:
            # Cari checkpoint terbaik
            if checkpoint_handler:
                checkpoint_path = checkpoint_handler.find_best_checkpoint()
                if not checkpoint_path:
                    logger.warning("⚠️ Tidak menemukan checkpoint terbaik, mencari yang terakhir...")
                    checkpoint_path = checkpoint_handler.find_latest_checkpoint()
                
                if checkpoint_path:
                    logger.info(f"📂 Melanjutkan training dari checkpoint: {checkpoint_path}")
                else:
                    logger.warning("⚠️ Tidak menemukan checkpoint, memulai training dari awal")
            else:
                logger.warning("⚠️ Checkpoint handler tidak tersedia")
        
        # Register callback untuk metrics jika pipeline mendukungnya
        if hasattr(pipeline, 'register_callback'):
            pipeline.register_callback('epoch_end', 
                                    lambda epoch, metrics: metrics_callback(
                                        epoch, metrics, metrics_tracker, ui_components, 
                                        {'update_plot': update_plot, 
                                            'update_metrics_table': update_metrics_table, 
                                            'update_status': update_status}
                                    ))
        
        # Jalankan training lewat pipeline
        results = pipeline.train(
            dataloaders=dataloaders,
            resume_from_checkpoint=checkpoint_path,
            save_every=save_every,
            epochs=epochs
        )
        
        # Reset status
        metrics_tracker.set_training_status(False)
        
        logger.success("✅ Training selesai!")
        return True
        
    except KeyboardInterrupt:
        logger.warning("⚠️ Training dihentikan oleh pengguna.")
        metrics_tracker.set_training_status(False)
        return False
        
    except Exception as e:
        logger.error(f"❌ Error saat training: {str(e)}")
        import traceback
        traceback.print_exc()
        
        metrics_tracker.set_training_status(False)
        return False

def backup_to_drive(checkpoint_handler, logger):
    """
    Backup checkpoint ke Google Drive.
    
    Args:
        checkpoint_handler: CheckpointHandler instance
        logger: Logger untuk mencatat aktivitas
        
    Returns:
        Boolean yang menunjukkan keberhasilan backup
    """
    try:
        from google.colab import drive
        if not os.path.exists('/content/drive'):
            drive.mount('/content/drive')
            
        # Buat direktori backup
        backup_dir = '/content/drive/MyDrive/SmartCash/checkpoints'
        os.makedirs(backup_dir, exist_ok=True)
        
        # Salin checkpoint terbaik
        if hasattr(checkpoint_handler, 'copy_to_drive'):
            # Gunakan metode built-in jika tersedia
            copied_paths = checkpoint_handler.copy_to_drive(backup_dir, best_only=True)
            logger.success(f"✅ Checkpoint disalin ke Google Drive: {backup_dir}")
            return True
        else:
            # Fallback manual backup
            import shutil
            checkpoints = checkpoint_handler.list_checkpoints()
            
            copied_count = 0
            # Salin best checkpoint jika ada
            if checkpoints.get('best'):
                best_checkpoint = checkpoints['best'][0]
                dest_path = os.path.join(backup_dir, best_checkpoint.name)
                shutil.copy2(best_checkpoint, dest_path)
                copied_count += 1
                logger.info(f"🔄 Checkpoint terbaik disalin ke Drive: {best_checkpoint.name}")
            
            # Salin latest checkpoint jika ada dan berbeda dari best
            if checkpoints.get('latest'):
                latest_checkpoint = checkpoints['latest'][0]
                # Cek apakah best dan latest berbeda
                if not checkpoints.get('best') or latest_checkpoint.name != checkpoints['best'][0].name:
                    dest_path = os.path.join(backup_dir, latest_checkpoint.name)
                    shutil.copy2(latest_checkpoint, dest_path)
                    copied_count += 1
                    logger.info(f"🔄 Checkpoint terakhir disalin ke Drive: {latest_checkpoint.name}")
            
            logger.success(f"✅ {copied_count} checkpoint berhasil disalin ke Google Drive")
            return copied_count > 0
            
    except Exception as e:
        logger.warning(f"⚠️ Error saat backup ke Google Drive: {str(e)}")
        try:
            # Coba otomatis mount drive jika belum di-mount
            if not os.path.exists('/content/drive'):
                from google.colab import drive
                drive.mount('/content/drive')
                logger.info("🔄 Google Drive berhasil di-mount, coba backup lagi")
                return backup_to_drive(checkpoint_handler, logger)  # Coba lagi setelah mount
        except:
            logger.error("❌ Gagal melakukan backup ke Google Drive")
        return False

def on_start_button_clicked(ui_components, pipeline, dataloaders, checkpoint_handler, drive_backup_checkbox, metrics_tracker, config, logger):
    """
    Handler untuk tombol start training.
    
    Args:
        ui_components: Dictionary UI components
        pipeline: TrainingPipeline instance
        dataloaders: Dictionary dataloaders
        checkpoint_handler: CheckpointHandler instance
        drive_backup_checkbox: Checkbox widget untuk backup ke Drive
        metrics_tracker: TrainingMetricsTracker instance
        config: Dictionary konfigurasi
        logger: Logger untuk mencatat aktivitas
    """
    # Validate required UI components
    required_components = ['start_button', 'stop_button', 'resume_checkbox',
                          'batch_size_dropdown', 'output']
                          
    for component in required_components:
        if component not in ui_components:
            logger.error(f"❌ Required UI component '{component}' not found")
            return
    
    with memory_manager():
        # Update UI controls
        ui_components['start_button'].disabled = True
        ui_components['stop_button'].disabled = False
        ui_components['resume_checkbox'].disabled = True
        ui_components['batch_size_dropdown'].disabled = True
        
        with ui_components['output']:
            clear_output(wait=True)
            
            # Override batch size jika diperlukan
            if ui_components['batch_size_dropdown'].value != config.get('training', {}).get('batch_size', 16):
                config['training']['batch_size'] = ui_components['batch_size_dropdown'].value
                logger.info(f"🔄 Menggunakan batch size: {ui_components['batch_size_dropdown'].value}")
                
                # Perbarui dataloader jika perlu
                if hasattr(pipeline, 'update_dataloaders'):
                    dataloaders = pipeline.update_dataloaders(batch_size=ui_components['batch_size_dropdown'].value)
            
            # Jalankan training di thread terpisah
            training_thread = threading.Thread(
                target=run_training_thread,
                args=(ui_components, pipeline, dataloaders, ui_components['resume_checkbox'].value, 
                    metrics_tracker, checkpoint_handler, 
                    drive_backup_checkbox.value if hasattr(drive_backup_checkbox, 'value') else False, 
                    config, logger)
            )
            training_thread.daemon = True
            training_thread.start()

def run_training_thread(ui_components, pipeline, dataloaders, resume_from_checkpoint, 
                        metrics_tracker, checkpoint_handler, do_drive_backup, config, logger):
    """
    Thread function untuk menjalankan training dan menangani cleanup setelah selesai.
    
    Args:
        ui_components: Dictionary UI components
        pipeline: TrainingPipeline instance
        dataloaders: Dictionary dataloaders
        resume_from_checkpoint: Boolean yang menunjukkan apakah melanjutkan dari checkpoint
        metrics_tracker: TrainingMetricsTracker instance
        checkpoint_handler: CheckpointHandler instance
        do_drive_backup: Boolean yang menunjukkan apakah melakukan backup ke Drive
        config: Dictionary konfigurasi
        logger: Logger untuk mencatat aktivitas
    """
    try:
        # Jalankan training
        success = run_training(ui_components, pipeline, dataloaders, resume_from_checkpoint, 
                            metrics_tracker, checkpoint_handler, config, logger)
        
        # Backup checkpoint ke Drive jika diminta dan training berhasil
        if success and do_drive_backup:
            backup_to_drive(checkpoint_handler, logger)
            
    except Exception as e:
        logger.error(f"❌ Error pada training thread: {str(e)}")
    finally:
        # Update UI controls
        ui_components['start_button'].disabled = False
        ui_components['stop_button'].disabled = True
        ui_components['resume_checkbox'].disabled = False
        ui_components['batch_size_dropdown'].disabled = False
        
        # Update visualisasi terakhir
        update_plot(ui_components, metrics_tracker)
        update_metrics_table(ui_components, metrics_tracker)
        update_status(ui_components, metrics_tracker)

def on_stop_button_clicked(ui_components, pipeline, metrics_tracker, logger):
    """
    Handler untuk tombol stop training.
    
    Args:
        ui_components: Dictionary UI components
        pipeline: TrainingPipeline instance
        metrics_tracker: TrainingMetricsTracker instance
        logger: Logger untuk mencatat aktivitas
    """
    # Validate required UI components
    if 'output' not in ui_components or 'stop_button' not in ui_components:
        logger.error("❌ Required UI components 'output' or 'stop_button' not found")
        return
    
    with ui_components['output']:
        if metrics_tracker.get_metrics()['is_training']:
            logger.warning("⚠️ Menghentikan training...")
            # Set flag untuk menghentikan training
            metrics_tracker.set_training_status(False)
            
            # Disable tombol stop
            ui_components['stop_button'].disabled = True
            
            # Callback untuk pipeline stop jika ada
            if hasattr(pipeline, 'stop_training'):
                pipeline.stop_training()

def setup_training_handlers(ui_components, pipeline, dataloaders, checkpoint_handler, config, logger):
    """
    Setup semua event handlers untuk UI training.
    
    Args:
        ui_components: Dictionary UI components
        pipeline: TrainingPipeline instance
        dataloaders: Dictionary dataloaders
        checkpoint_handler: CheckpointHandler instance
        config: Dictionary konfigurasi
        logger: Logger untuk mencatat aktivitas
        
    Returns:
        Tuple (metrics_tracker, ui_components) dengan metrics tracker yang diinisialisasi dan 
        UI components yang telah di-setup handler-nya
    """
    # Validate required UI components
    required_components = ['start_button', 'stop_button', 'resume_checkbox', 
                         'batch_size_dropdown', 'drive_backup_checkbox', 'output',
                         'live_plot_tab', 'metrics_tab', 'status_tab',
                         'status_text', 'progress_bar']
    
    missing_components = [comp for comp in required_components if comp not in ui_components]
    if missing_components:
        logger.error(f"❌ Missing UI components: {', '.join(missing_components)}")
        if len(missing_components) > len(required_components) / 2:
            logger.error("❌ Too many missing components, aborting setup")
            return None, ui_components
    
    # Inisialisasi metrics tracker
    metrics_tracker = TrainingMetricsTracker()
    
    # Update default batch size from config
    if 'training' in config and 'batch_size' in config['training']:
        ui_components['batch_size_dropdown'].value = config['training']['batch_size']
    
    # Setup handler untuk tombol start training
    if 'start_button' in ui_components:
        ui_components['start_button'].on_click(
            lambda b: on_start_button_clicked(
                ui_components, pipeline, dataloaders, checkpoint_handler,
                ui_components.get('drive_backup_checkbox'), metrics_tracker, config, logger
            )
        )
    
    # Setup handler untuk tombol stop training
    if 'stop_button' in ui_components:
        ui_components['stop_button'].on_click(
            lambda b: on_stop_button_clicked(
                ui_components, pipeline, metrics_tracker, logger
            )
        )
    
    # Initialize visualizations
    if 'live_plot_tab' in ui_components:
        update_plot(ui_components, metrics_tracker)
    if 'metrics_tab' in ui_components:
        update_metrics_table(ui_components, metrics_tracker)
    if 'status_tab' in ui_components:
        update_status(ui_components, metrics_tracker)
    
    return metrics_tracker, ui_components