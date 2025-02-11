# File: src/train/callbacks.py
# Author: Alfrida Sabar
# Deskripsi: Implementasi callback untuk monitoring dan visualisasi training

from pathlib import Path
import matplotlib.pyplot as plt
from utils.logging import ColoredLogger

class ProgressCallback:
    def __init__(self):
        self.logger = ColoredLogger('Progress')
        self.best_map = 0
        
    def on_epoch_end(self, trainer, epoch, logs):
        mAP = logs['mAP']
        if mAP > self.best_map:
            self.best_map = mAP
            self.logger.info(f'🎯 Best mAP: {mAP:.4f}')
            
class LRSchedulerCallback:
    def __init__(self, scheduler_fn, **kwargs):
        self.scheduler = None
        self.scheduler_fn = scheduler_fn
        self.kwargs = kwargs
        
    def on_train_begin(self, trainer):
        self.scheduler = self.scheduler_fn(
            trainer.optimizer, 
            **self.kwargs
        )
        
    def on_epoch_end(self, trainer, epoch, logs):
        self.scheduler.step()
        
class VisualizationCallback:
    def __init__(self, log_dir='runs'):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.logger = ColoredLogger('Visualization')
        self.train_losses = []
        self.val_losses = []
        self.maps = []
        
    def on_epoch_end(self, trainer, epoch, logs):
        self.train_losses.append(logs['loss'])
        if 'val_loss' in logs:
            self.val_losses.append(logs['val_loss'])
        self.maps.append(logs['mAP'])
        
        if (epoch + 1) % 5 == 0:
            self._plot_metrics(epoch)
            
    def _plot_metrics(self, epoch):
        plt.figure(figsize=(12, 4))
        
        # Plot losses
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        if self.val_losses:
            plt.plot(self.val_losses, label='Val Loss')
        plt.title('Training Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot mAP
        plt.subplot(1, 2, 2)
        plt.plot(self.maps, label='mAP', color='g')
        plt.title('Mean Average Precision')
        plt.xlabel('Epoch')
        plt.ylabel('mAP')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.log_dir / f'metrics_epoch_{epoch}.png')
        plt.close()
        
        self.logger.info(f'📊 Plot metrik tersimpan: metrics_epoch_{epoch}.png')