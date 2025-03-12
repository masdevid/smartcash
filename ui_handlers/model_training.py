"""
File: smartcash/ui_handlers/model_training.py
Author: Alfrida Sabar (refactored)
Deskripsi: Handler untuk komponen UI eksekusi training model SmartCash dengan perbaikan import.
"""

import threading
import time
import os
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output, HTML
from pathlib import Path

from smartcash.utils.ui_utils import create_status_indicator, create_metric_display

def setup_model_training_handlers(ui_components, config=None):
    """Setup handlers untuk komponen UI eksekusi training model."""
    # Inisialisasi dependencies
    logger = None
    model_manager = None
    dataset_manager = None
    observer_manager = None
    checkpoint_manager = None
    
    try:
        from smartcash.utils.logger import get_logger
        from smartcash.handlers.model import ModelManager
        from smartcash.handlers.dataset import DatasetManager
        from smartcash.utils.observer.observer_manager import ObserverManager
        from smartcash.handlers.checkpoint import CheckpointManager
        from smartcash.utils.observer import EventTopics, EventDispatcher
        
        # Perbaikan import training callback
        # Coba beberapa kemungkinan path untuk training_callbacks
        try:
            # Opsi 1: Path asli
            from smartcash.utils.training.training_callbacks import TrainingCallbacks
        except ImportError:
            try:
                # Opsi 2: Path yang mungkin dari struktur file
                from smartcash.utils.training import training_callbacks
            except ImportError:
                try:
                    # Opsi 3: Path alternatif
                    from smartcash.handlers.model.core import training_callbacks
                except ImportError:
                    # Fallback: Tidak perlu import jika tidak digunakan langsung
                    pass
        
        logger = get_logger("model_training")
        observer_manager = ObserverManager(auto_register=True)
        observer_group = "model_training_observers"
        
        # Clean up any existing observers from previous runs
        observer_manager.unregister_group(observer_group)
        
        # Initialize managers if config available
        if config:
            model_manager = ModelManager(config, logger=logger)
            dataset_manager = DatasetManager(config, logger=logger)
            checkpoint_manager = CheckpointManager(output_dir=config.get('model', {}).get('output_dir', 'runs/train/weights'), logger=logger)
        
    except ImportError as e:
        if ui_components and 'training_status' in ui_components:
            with ui_components['training_status']:
                display(HTML(f"<p style='color:orange'>⚠️ Beberapa modul tidak tersedia: {str(e)}</p>"))
    
    # Training state tracking
    training_state = {
        'running': False,
        'paused': False,
        'epoch': 0,
        'total_epochs': config.get('training', {}).get('epochs', 30) if config else 30,
        'batch': 0,
        'total_batches': 0,
        'metrics': {
            'train_loss': [],
            'val_loss': [],
            'precision': [],
            'recall': [],
            'mAP': [],
            'lr': []
        },
        'best_metrics': {
            'mAP': 0,
            'epoch': 0
        },
        'checkpoints': []
    }
    
    # Thread for training
    training_thread = None
    
    # Setup observer callbacks
    if observer_manager:
        # Epoch progress observer
        observer_manager.create_simple_observer(
            event_type=EventTopics.EPOCH_END,
            callback=lambda event_type, sender, epoch=0, epochs=0, **kwargs: update_epoch_progress(epoch, epochs),
            name="EpochProgressObserver",
            group=observer_group
        )
        
        # Batch progress observer
        observer_manager.create_simple_observer(
            event_type=EventTopics.BATCH_END,
            callback=lambda event_type, sender, batch=0, batches=0, **kwargs: update_batch_progress(batch, batches),
            name="BatchProgressObserver",
            group=observer_group
        )
        
        # Metrics observer
        observer_manager.create_simple_observer(
            event_type=EventTopics.EPOCH_END,
            callback=lambda event_type, sender, metrics=None, **kwargs: update_metrics(metrics) if metrics else None,
            name="MetricsObserver",
            group=observer_group
        )
        
        # Checkpoint observer
        observer_manager.create_simple_observer(
            event_type=EventTopics.CHECKPOINT_SAVE,
            callback=lambda event_type, sender, path=None, **kwargs: update_checkpoints(path) if path else None,
            name="CheckpointObserver",
            group=observer_group
        )
        
        # Training status observer
        observer_manager.create_logging_observer(
            event_types=[
                EventTopics.TRAINING_START,
                EventTopics.TRAINING_END,
                EventTopics.EPOCH_START,
                EventTopics.EPOCH_END,
                EventTopics.TRAINING_ERROR
            ],
            log_level="info",
            name="TrainingStatusObserver",
            group=observer_group
        )
    
    # Helper functions for UI updates
    def update_epoch_progress(epoch, epochs):
        """Update epoch progress bar"""
        if ui_components and 'epoch_progress' in ui_components:
            bar = ui_components['epoch_progress']
            bar.max = epochs
            bar.value = epoch
            bar.description = f"{epoch}/{epochs} Epochs:"
    
    def update_batch_progress(batch, batches):
        """Update batch progress bar"""
        if ui_components and 'batch_progress' in ui_components:
            bar = ui_components['batch_progress']
            bar.max = batches
            bar.value = batch
            bar.description = f"{batch}/{batches} Batches:"
    
    def update_metrics(metrics):
        """Update metrics display and charts"""
        if not metrics or not isinstance(metrics, dict):
            return
            
        # Update metrics state
        for key in training_state['metrics']:
            if key in metrics:
                training_state['metrics'][key].append(metrics[key])
        
        # Check for best model
        if 'mAP' in metrics and metrics['mAP'] > training_state['best_metrics']['mAP']:
            training_state['best_metrics']['mAP'] = metrics['mAP']
            training_state['best_metrics']['epoch'] = training_state['epoch']
        
        # Update metrics display
        if ui_components and 'metrics_display' in ui_components:
            with ui_components['metrics_display']:
                clear_output(wait=True)
                
                metrics_row = widgets.HBox([
                    create_metric_display("Epoch", training_state['epoch'] + 1),
                    create_metric_display("Train Loss", f"{metrics.get('train_loss', 0):.4f}"),
                    create_metric_display("Val Loss", f"{metrics.get('val_loss', 0):.4f}"),
                    create_metric_display("Precision", f"{metrics.get('precision', 0):.4f}"),
                    create_metric_display("Recall", f"{metrics.get('recall', 0):.4f}"),
                    create_metric_display("mAP", f"{metrics.get('mAP', 0):.4f}", 
                                         is_good=metrics.get('mAP', 0) > 0.7)
                ])
                display(metrics_row)
                
                # Show best metrics
                if training_state['best_metrics']['mAP'] > 0:
                    best_info = f"""
                    <div style="margin-top: 10px; padding: 5px 10px; background-color: #d4edda; 
                               border-left: 4px solid #28a745; border-radius: 3px;">
                        <span style="font-weight: bold;">🏆 Best mAP:</span> 
                        {training_state['best_metrics']['mAP']:.4f} (epoch {training_state['best_metrics']['epoch'] + 1})
                    </div>
                    """
                    display(HTML(best_info))
        
        # Update charts
        update_charts()
    
    def update_charts():
        """Update all metric charts"""
        if ui_components and 'metrics_tabs' in ui_components:
            charts = ui_components['metrics_tabs']
            
            # Loss chart
            with charts.children[0]:
                clear_output(wait=True)
                if len(training_state['metrics']['train_loss']) > 0:
                    plt.figure(figsize=(9, 5))
                    epochs = range(1, len(training_state['metrics']['train_loss']) + 1)
                    plt.plot(epochs, training_state['metrics']['train_loss'], 'b-', label='Training Loss')
                    plt.plot(epochs, training_state['metrics']['val_loss'], 'r-', label='Validation Loss')
                    plt.title('Loss Curves')
                    plt.xlabel('Epoch')
                    plt.ylabel('Loss')
                    plt.legend()
                    plt.grid(linestyle='--', alpha=0.6)
                    plt.tight_layout()
                    display(plt.gcf())
                    plt.close()
            
            # Precision/Recall chart
            with charts.children[1]:
                clear_output(wait=True)
                if len(training_state['metrics']['precision']) > 0:
                    plt.figure(figsize=(9, 5))
                    epochs = range(1, len(training_state['metrics']['precision']) + 1)
                    plt.plot(epochs, training_state['metrics']['precision'], 'g-', label='Precision')
                    plt.plot(epochs, training_state['metrics']['recall'], 'b-', label='Recall')
                    plt.plot(epochs, training_state['metrics']['mAP'], 'r-', label='mAP')
                    plt.title('Metrics History')
                    plt.xlabel('Epoch')
                    plt.ylabel('Score')
                    plt.legend()
                    plt.grid(linestyle='--', alpha=0.6)
                    plt.tight_layout()
                    display(plt.gcf())
                    plt.close()
            
            # Learning rate chart
            with charts.children[2]:
                clear_output(wait=True)
                if len(training_state['metrics']['lr']) > 0:
                    plt.figure(figsize=(9, 5))
                    epochs = range(1, len(training_state['metrics']['lr']) + 1)
                    plt.plot(epochs, training_state['metrics']['lr'], 'b-')
                    plt.title('Learning Rate Schedule')
                    plt.xlabel('Epoch')
                    plt.ylabel('Learning Rate')
                    plt.yscale('log')
                    plt.grid(linestyle='--', alpha=0.6)
                    plt.tight_layout()
                    display(plt.gcf())
                    plt.close()
    
    def update_checkpoints(path):
        """Update checkpoint list when a new checkpoint is saved"""
        if path and isinstance(path, (str, Path)):
            path_str = str(path)
            if path_str not in training_state['checkpoints']:
                training_state['checkpoints'].append(path_str)
            
            # Update checkpoint selector
            if ui_components and 'checkpoint_selector' in ui_components:
                selector = ui_components['checkpoint_selector']
                selector.options = training_state['checkpoints']
                selector.disabled = len(training_state['checkpoints']) == 0
            
            # Update checkpoint info
            if ui_components and 'checkpoint_info' in ui_components:
                with ui_components['checkpoint_info']:
                    clear_output(wait=True)
                    
                    # Check if best or last checkpoint
                    is_best = "best" in path_str.lower()
                    is_last = training_state['epoch'] == training_state['total_epochs'] - 1
                    
                    info_html = f"""
                    <div style="padding: 10px; background-color: {('#d4edda' if is_best else '#f8f9fa')}; 
                              border-left: 4px solid {('#28a745' if is_best else '#6c757d')}; 
                              border-radius: 4px; margin-bottom: 10px;">
                        <h4 style="margin-top: 0;">{'🏆 Best Model Saved' if is_best else '💾 Checkpoint Saved'}</h4>
                        <p><b>Path:</b> {path_str}</p>
                        <p><b>Epoch:</b> {training_state['epoch'] + 1}/{training_state['total_epochs']}</p>
                        <p><b>Metrics:</b> mAP {training_state['best_metrics']['mAP']:.4f}</p>
                    </div>
                    """
                    display(HTML(info_html))
                    
                    # Display all checkpoints
                    if training_state['checkpoints']:
                        checkpoint_list = "<h4>📋 Available Checkpoints</h4><ul>"
                        for cp in training_state['checkpoints']:
                            checkpoint_list += f"<li>{os.path.basename(cp)}</li>"
                        checkpoint_list += "</ul>"
                        display(HTML(checkpoint_list))
    
    # Function to prepare and load data
    def prepare_dataloaders():
        """Prepare training and validation dataloaders"""
        if not dataset_manager:
            return None, None
            
        try:
            with ui_components['training_status']:
                display(create_status_indicator("info", "🔄 Preparing dataloaders..."))
                
            # Load dataset for training and validation
            train_dataloader = dataset_manager.get_dataloader(
                split='train',
                batch_size=config.get('training', {}).get('batch_size', 16),
                shuffle=True
            )
            
            val_dataloader = dataset_manager.get_dataloader(
                split='valid',
                batch_size=config.get('training', {}).get('batch_size', 16),
                shuffle=False
            )
            
            with ui_components['training_status']:
                display(create_status_indicator(
                    "success", 
                    f"✅ Dataloaders prepared: {len(train_dataloader)} train batches, {len(val_dataloader)} validation batches"
                ))
                
            return train_dataloader, val_dataloader
            
        except Exception as e:
            with ui_components['training_status']:
                display(create_status_indicator("error", f"❌ Error preparing dataloaders: {str(e)}"))
            return None, None
    
    # Function to train the model
    def train_model():
        """Start model training"""
        nonlocal training_state
        
        # Set running flags
        training_state['running'] = True
        training_state['paused'] = False
        
        try:
            # Update UI state
            update_ui_state(True)
            
            # Prepare dataloaders
            train_loader, val_loader = prepare_dataloaders()
            if not train_loader or not val_loader:
                raise ValueError("Failed to prepare dataloaders")
            
            # Get total batches for progress
            training_state['total_batches'] = len(train_loader)
            
            # Get training mode
            train_mode = ui_components['training_options'].children[0].value
            use_gpu = ui_components['training_options'].children[1].value
            enable_tensorboard = ui_components['training_options'].children[2].value
            enable_checkpointing = ui_components['training_options'].children[3].value
            enable_early_stopping = ui_components['training_options'].children[4].value
            
            # Get additional settings from config
            epochs = config.get('training', {}).get('epochs', 30)
            training_state['total_epochs'] = epochs
            
            # Load checkpoint if continuing training
            model = None
            start_epoch = 0
            
            if train_mode == 'Continue From Last Checkpoint':
                if checkpoint_manager:
                    checkpoint_path = checkpoint_manager.find_latest_checkpoint()
                    if checkpoint_path:
                        with ui_components['training_status']:
                            display(create_status_indicator("info", f"🔄 Loading checkpoint: {checkpoint_path}"))
                        
                        model, checkpoint_data = checkpoint_manager.load_checkpoint(checkpoint_path)
                        start_epoch = checkpoint_data.get('epoch', 0) + 1
                        
                        with ui_components['training_status']:
                            display(create_status_indicator(
                                "success", 
                                f"✅ Checkpoint loaded (epoch {start_epoch})"
                            ))
            elif train_mode == 'Transfer Learning':
                # Create model with pretrained weights
                if model_manager:
                    model = model_manager.create_model(pretrained=True)
                    with ui_components['training_status']:
                        display(create_status_indicator("info", "🧠 Using transfer learning with pretrained weights"))
            
            # Create new model if needed
            if model is None and model_manager:
                model = model_manager.create_model()
            
            # Start training
            with ui_components['training_status']:
                display(create_status_indicator(
                    "info", 
                    f"🚀 Starting training from epoch {start_epoch + 1}/{epochs}"
                ))
            
            # Notify training start
            if observer_manager:
                EventDispatcher.notify(
                    event_type=EventTopics.TRAINING_START,
                    sender="training_handler",
                    message=f"Starting training for {epochs} epochs",
                    start_epoch=start_epoch,
                    total_epochs=epochs
                )
            
            # Train the model using model_manager
            if model_manager:
                # Set up observers
                observers = []
                if observer_manager:
                    # Add ColabObserver if in Colab
                    try:
                        from smartcash.handlers.model.observers import ColabObserver
                        observers.append(ColabObserver(logger))
                    except ImportError:
                        pass
                
                # Train model
                training_state['epoch'] = start_epoch
                
                # Create training kwargs
                train_kwargs = {
                    'epochs': epochs,
                    'start_epoch': start_epoch,
                    'enable_tensorboard': enable_tensorboard,
                    'save_checkpoints': enable_checkpointing,
                    'early_stopping': enable_early_stopping,
                    'use_gpu': use_gpu,
                    'observers': observers
                }
                
                # Execute training
                result = model_manager.train(
                    train_loader=train_loader,
                    val_loader=val_loader,
                    model=model,
                    **train_kwargs
                )
                
                # Process training result
                if isinstance(result, dict):
                    with ui_components['training_status']:
                        display(create_status_indicator(
                            "success", 
                            f"✅ Training completed in {result.get('total_time', 0):.2f} seconds"
                        ))
                    
                    # Save best metrics
                    training_state['best_metrics'] = result.get('best_metrics', training_state['best_metrics'])
                    
                    # Update checkpoint list
                    if 'checkpoints' in result:
                        for checkpoint in result['checkpoints']:
                            update_checkpoints(checkpoint)
                
                # Notify training completion
                if observer_manager:
                    EventDispatcher.notify(
                        event_type=EventTopics.TRAINING_END,
                        sender="training_handler",
                        message="Training completed successfully",
                        result=result
                    )
                
        except Exception as e:
            with ui_components['training_status']:
                display(create_status_indicator("error", f"❌ Error during training: {str(e)}"))
            
            # Notify training error
            if observer_manager:
                EventDispatcher.notify(
                    event_type=EventTopics.TRAINING_ERROR,
                    sender="training_handler",
                    message=f"Training error: {str(e)}",
                    error=str(e)
                )
        
        finally:
            # Reset UI state
            training_state['running'] = False
            training_state['paused'] = False
            update_ui_state(False)
    
    # Helper to update UI based on training state
    def update_ui_state(is_training):
        """Update UI buttons and state based on training state"""
        if ui_components:
            ui_components['start_button'].disabled = is_training
            ui_components['pause_button'].disabled = not is_training or training_state['paused']
            ui_components['stop_button'].disabled = not is_training
            
            # Update pause button text
            if is_training and training_state['paused']:
                ui_components['pause_button'].description = "Resume Training"
                ui_components['pause_button'].icon = "play"
                ui_components['pause_button'].disabled = False
            elif is_training:
                ui_components['pause_button'].description = "Pause Training"
                ui_components['pause_button'].icon = "pause"
                ui_components['pause_button'].disabled = False
    
    # Handler for start button
    def on_start_training(b):
        """Start training process"""
        nonlocal training_thread
        
        if training_state['running']:
            return
        
        with ui_components['training_status']:
            clear_output()
            display(create_status_indicator("info", "🚀 Initializing training..."))
        
        # Reset metrics if starting from scratch
        train_mode = ui_components['training_options'].children[0].value
        if train_mode == 'From Scratch':
            training_state['metrics'] = {
                'train_loss': [],
                'val_loss': [],
                'precision': [],
                'recall': [],
                'mAP': [],
                'lr': []
            }
            training_state['best_metrics'] = {
                'mAP': 0,
                'epoch': 0
            }
        
        # Start training in a thread
        training_thread = threading.Thread(target=train_model)
        training_thread.daemon = True
        training_thread.start()
    
    # Handler for pause button
    def on_pause_training(b):
        """Pause/resume training"""
        training_state['paused'] = not training_state['paused']
        
        # Update UI
        update_ui_state(training_state['running'])
        
        with ui_components['training_status']:
            if training_state['paused']:
                display(create_status_indicator("warning", "⏸️ Training paused"))
            else:
                display(create_status_indicator("info", "▶️ Training resumed"))
    
    # Handler for stop button
    def on_stop_training(b):
        """Stop training process"""
        training_state['running'] = False
        
        with ui_components['training_status']:
            display(create_status_indicator("warning", "🛑 Training stopped"))
        
        # Update UI
        update_ui_state(False)
        
        # Notify training end due to stop
        if observer_manager:
            EventDispatcher.notify(
                event_type=EventTopics.TRAINING_END,
                sender="training_handler",
                message="Training stopped by user",
                stopped_early=True,
                epoch=training_state['epoch']
            )
    
    # Handler for load model button
    def on_load_model(b):
        """Load selected checkpoint"""
        selected = ui_components['checkpoint_selector'].value
        
        if not selected:
            with ui_components['training_status']:
                display(create_status_indicator("warning", "⚠️ No checkpoint selected"))
            return
        
        with ui_components['training_status']:
            display(create_status_indicator("info", f"🔄 Loading checkpoint: {selected}"))
        
        try:
            if checkpoint_manager and model_manager:
                # Load checkpoint
                model, checkpoint_data = checkpoint_manager.load_checkpoint(selected)
                
                # Display model info
                with ui_components['training_status']:
                    display(create_status_indicator(
                        "success", 
                        f"✅ Model loaded from checkpoint (epoch {checkpoint_data.get('epoch', 0) + 1})"
                    ))
                    
                    # Show metrics if available
                    if 'metrics' in checkpoint_data:
                        metrics_table = f"""
                        <div style="margin-top: 10px; padding: 10px; background-color: #f8f9fa; border-radius: 4px;">
                            <h4 style="margin-top: 0;">📊 Checkpoint Metrics</h4>
                            <table style="width: 100%; border-collapse: collapse;">
                                <tr style="background-color: #f0f0f0;">
                                    <th style="padding: 8px; text-align: left; border: 1px solid #ddd;">Metric</th>
                                    <th style="padding: 8px; text-align: left; border: 1px solid #ddd;">Value</th>
                                </tr>
                        """
                        
                        metrics = checkpoint_data['metrics']
                        for key, value in metrics.items():
                            if isinstance(value, (int, float)):
                                metrics_table += f"""
                                <tr>
                                    <td style="padding: 8px; text-align: left; border: 1px solid #ddd;">{key}</td>
                                    <td style="padding: 8px; text-align: left; border: 1px solid #ddd;">{value:.4f if isinstance(value, float) else value}</td>
                                </tr>
                                """
                        
                        metrics_table += """
                            </table>
                        </div>
                        """
                        display(HTML(metrics_table))
            else:
                with ui_components['training_status']:
                    display(create_status_indicator(
                        "error", 
                        "❌ Checkpoint manager or model manager not available"
                    ))
        except Exception as e:
            with ui_components['training_status']:
                display(create_status_indicator("error", f"❌ Error loading checkpoint: {str(e)}"))
    
    # Handler for export model button
    def on_export_model(b):
        """Export model to selected format"""
        export_format = ui_components['export_format'].value
        export_path = ui_components['export_path'].value
        
        with ui_components['training_status']:
            display(create_status_indicator("info", f"🔄 Exporting model as {export_format}..."))
        
        try:
            if model_manager:
                # Check if we have loaded checkpoints
                if not training_state['checkpoints']:
                    # Try to find best checkpoint
                    if checkpoint_manager:
                        best_checkpoint = checkpoint_manager.find_best_checkpoint()
                        if best_checkpoint:
                            training_state['checkpoints'].append(best_checkpoint)
                            ui_components['checkpoint_selector'].options = training_state['checkpoints']
                            ui_components['checkpoint_selector'].value = best_checkpoint
                        else:
                            raise ValueError("No checkpoint available for export")
                    else:
                        raise ValueError("No checkpoint available for export")
                
                # Get selected checkpoint
                checkpoint_path = ui_components['checkpoint_selector'].value
                
                # Export model
                exported_path = model_manager.export_model(
                    checkpoint_path=checkpoint_path,
                    format=export_format.lower(),
                    output_path=export_path
                )
                
                if exported_path:
                    with ui_components['training_status']:
                        display(create_status_indicator(
                            "success", 
                            f"✅ Model exported to {exported_path}"
                        ))
                else:
                    with ui_components['training_status']:
                        display(create_status_indicator("error", "❌ Failed to export model"))
            else:
                with ui_components['training_status']:
                    display(create_status_indicator("error", "❌ Model manager not available"))
        except Exception as e:
            with ui_components['training_status']:
                display(create_status_indicator("error", f"❌ Error exporting model: {str(e)}"))
    
    # Initialize checkpoint list if available
    def initialize_checkpoint_list():
        """Initialize checkpoint list from existing checkpoints"""
        if checkpoint_manager:
            # Find available checkpoints
            checkpoints = checkpoint_manager.list_checkpoints()
            if checkpoints:
                training_state['checkpoints'] = checkpoints
                ui_components['checkpoint_selector'].options = checkpoints
                ui_components['checkpoint_selector'].disabled = False
                
                # Update checkpoint info
                with ui_components['checkpoint_info']:
                    clear_output()
                    display(HTML("<h4>📋 Available Checkpoints</h4>"))
                    
                    checkpoint_list = "<ul>"
                    for cp in checkpoints:
                        checkpoint_list += f"<li>{os.path.basename(cp)}</li>"
                    checkpoint_list += "</ul>"
                    display(HTML(checkpoint_list))
    
    # Ensure widgets is imported for metrics display
    try:
        import ipywidgets as widgets
    except ImportError:
        pass
    
    # Register handlers
    ui_components['start_button'].on_click(on_start_training)
    ui_components['pause_button'].on_click(on_pause_training)
    ui_components['stop_button'].on_click(on_stop_training)
    ui_components['load_model_button'].on_click(on_load_model)
    ui_components['export_model_button'].on_click(on_export_model)
    
    # Initialize UI
    initialize_checkpoint_list()
    
    # Cleanup function
    def cleanup():
        """Clean up resources when cell is rerun or notebook is closed"""
        if observer_manager:
            observer_manager.unregister_group(observer_group)
            if logger:
                logger.info("✅ Training execution observers cleaned up")
    
    # Add cleanup to UI components
    ui_components['cleanup'] = cleanup
    
    return ui_components