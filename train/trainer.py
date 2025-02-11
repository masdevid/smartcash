# File: src/train/trainer.py
# Author: Alfrida Sabar
# Deskripsi: Implementasi pipeline training dengan callbacks dan checkpointing

import torch
from pathlib import Path
from tqdm import tqdm

from smartcash.src.models.loss import YOLOLoss
from smartcash.src.utils.metrics import MeanAveragePrecision
from utils.logging import ColoredLogger

class TrainingCallback:
    def on_train_begin(self, trainer): pass
    def on_train_end(self, trainer): pass
    def on_epoch_begin(self, trainer, epoch): pass
    def on_epoch_end(self, trainer, epoch, logs): pass
    def on_batch_begin(self, trainer, batch): pass
    def on_batch_end(self, trainer, batch, logs): pass

class EarlyStopping(TrainingCallback):
    def __init__(self, patience=5, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.stop_training = False
        
    def on_epoch_end(self, trainer, epoch, logs):
        val_loss = logs['val_loss']
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop_training = True
        else:
            self.best_loss = val_loss
            self.counter = 0

class ModelCheckpoint(TrainingCallback):
    def __init__(self, filepath, monitor='val_loss', save_best_only=True):
        self.filepath = Path(filepath)
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.best_value = float('inf')
        
    def on_epoch_end(self, trainer, epoch, logs):
        current = logs[self.monitor]
        if self.save_best_only:
            if current < self.best_value:
                self.best_value = current
                self._save_model(trainer.model, epoch, logs)
        else:
            self._save_model(trainer.model, epoch, logs)
            
    def _save_model(self, model, epoch, logs):
        save_path = self.filepath / f'model_epoch_{epoch}.pt'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'logs': logs
        }, save_path)

class MetricsLogger(TrainingCallback):
    def __init__(self):
        self.logger = ColoredLogger('MetricsLogger')
        self.history = {'train': [], 'val': []}
        
    def on_epoch_end(self, trainer, epoch, logs):
        metrics = {
            'loss': f"{logs['loss']:.4f}",
            'val_loss': f"{logs['val_loss']:.4f}",
            'mAP': f"{logs['mAP']:.4f}"
        }
        self.logger.metric(f'Epoch {epoch}', metrics=metrics)
        
        for k, v in logs.items():
            if k.startswith('val_'):
                phase = 'val'
                metric = k[4:]
            else:
                phase = 'train'
                metric = k
            
            if len(self.history[phase]) <= epoch:
                self.history[phase].append({})
            self.history[phase][epoch][metric] = v

class Trainer:
    def __init__(self, model, train_loader, device, 
                 num_classes=7, callbacks=None, save_dir='weights'):
        self.logger = ColoredLogger('Trainer')
        self.model = model
        self.train_loader = train_loader
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Loss and metrics
        self.criterion = YOLOLoss(num_classes=num_classes)
        self.map_metric = MeanAveragePrecision(num_classes=num_classes)
        
        # Default callbacks
        self.callbacks = callbacks or [
            EarlyStopping(patience=5),
            ModelCheckpoint(self.save_dir),
            MetricsLogger()
        ]
        
        # Initialize optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=0.001,
            weight_decay=0.0001
        )
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=0.001,
            epochs=100,
            steps_per_epoch=len(train_loader)
        )
        
    def _run_epoch(self, epoch, validation=False):
        loader = self.val_loader if validation else self.train_loader
        phase = 'val' if validation else 'train'
        
        self.model.train(not validation)
        epoch_loss = 0
        
        with torch.set_grad_enabled(not validation):
            with tqdm(loader, desc=f'{phase} epoch {epoch}') as pbar:
                for batch_idx, (images, targets) in enumerate(pbar):
                    if not validation:
                        for cb in self.callbacks:
                            cb.on_batch_begin(self, batch_idx)
                    
                    images = images.to(self.device)
                    targets = targets.to(self.device)
                    
                    self.optimizer.zero_grad()
                    outputs = self.model(images)
                    loss = self.compute_loss(outputs, targets)
                    
                    if not validation:
                        loss.backward()
                        self.optimizer.step()
                        self.scheduler.step()
                    
                    epoch_loss += loss.item()
                    
                    if not validation:
                        logs = {'batch_loss': loss.item()}
                        for cb in self.callbacks:
                            cb.on_batch_end(self, batch_idx, logs)
                    
                    pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                    
        return epoch_loss / len(loader)

    def train(self, epochs, val_loader=None):
        self.val_loader = val_loader
        self.logger.info('🚀 Memulai training...')
        
        for cb in self.callbacks:
            cb.on_train_begin(self)
            
        for epoch in range(epochs):
            for cb in self.callbacks:
                cb.on_epoch_begin(self, epoch)
            
            train_loss = self._run_epoch(epoch)
            
            logs = {'loss': train_loss}
            if val_loader:
                val_loss = self._run_epoch(epoch, validation=True)
                logs['val_loss'] = val_loss
            
            # Calculate mAP
            mAP = self.calculate_mAP()
            logs['mAP'] = mAP
            
            for cb in self.callbacks:
                cb.on_epoch_end(self, epoch, logs)
                
            if any(cb.stop_training for cb in self.callbacks 
                  if isinstance(cb, EarlyStopping)):
                break
                
        for cb in self.callbacks:
            cb.on_train_end(self)
            
        self.logger.info('✨ Training selesai!')

    def compute_loss(self, outputs, targets):
        loss, loss_dict = self.criterion(outputs, targets)
        return loss
        
    def calculate_mAP(self):
        self.model.eval()
        self.map_metric.reset()
        
        with torch.no_grad():
            for images, targets in self.val_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(images)
                self.map_metric.update(outputs, targets)
                
        mAP = self.map_metric.compute()
        self.model.train()
        return mAP