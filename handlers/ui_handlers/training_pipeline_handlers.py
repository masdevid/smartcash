"""
File: smartcash/handlers/ui_handlers/training_pipeline_handlers.py
Author: Alfrida Sabar
Deskripsi: Handler untuk komponen UI pipeline training, menangani inisialisasi pipeline
          dan manajemen status training.
"""

import os
import pickle
import torch
import yaml
import gc
from IPython.display import display, clear_output, HTML
from pathlib import Path

# Import common utilities
from smartcash.handlers.ui_handlers.common_utils import memory_manager, display_gpu_info, load_config

def on_check_status_button_clicked(ui_components, components_dict):
    """
    Handler untuk tombol refresh status training.
    
    Args:
        ui_components: Dictionary berisi komponen UI dari create_training_pipeline_ui()
        components_dict: Dictionary berisi komponen lain yang diperlukan (pipeline, logger, dll)
    """
    with ui_components['status_output']:
        clear_output()
        
        try:
            # Memuat komponen
            pipeline = components_dict.get('pipeline')
            logger = components_dict.get('logger', print)
            
            if pipeline:
                status = pipeline.get_training_status()
                
                if status['status'] == 'training':
                    # Update progress bar
                    ui_components['progress_bar'].value = status.get('progress', 0)
                    
                    # Update info text dengan styling
                    ui_components['info_text'].value = f"""
                    <div style="padding: 10px; border-radius: 5px; background-color: #f0f7ff; margin-bottom: 10px">
                        <p><b>Status:</b> <span style="color: #3498db">Training</span></p>
                        <p><b>Epoch:</b> {status.get('current_epoch', 0)+1}/{status.get('total_epochs', '?')}</p>
                        <p><b>Best Val Loss:</b> <span style="color: #2ecc71; font-weight: bold">{status.get('best_val_loss', '-'):.4f}</span></p>
                        <p><b>Estimasi Waktu:</b> {status.get('estimated_time_remaining', 'Menghitung...')}</p>
                        <p><b>Device:</b> {("GPU" + (" - " + str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else "")) if torch.cuda.is_available() else "CPU"}</p>
                    </div>
                    """
                else:
                    # Reset progress bar
                    ui_components['progress_bar'].value = 0
                    
                    # Update info text
                    ui_components['info_text'].value = f"""
                    <div style="padding: 10px; border-radius: 5px; background-color: #f5f5f5; margin-bottom: 10px">
                        <p><b>Status:</b> <span style="color: gray">Idle</span></p>
                        <p><b>Pesan:</b> {status.get('message', 'Tidak ada informasi')}</p>
                        <p><b>Device:</b> {("GPU" + (" - " + str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else "")) if torch.cuda.is_available() else "CPU"}</p>
                    </div>
                    """
            else:
                logger("⚠️ Pipeline belum diinisialisasi")
                ui_components['info_text'].value = """
                <div style="padding: 10px; border-radius: 5px; background-color: #fff4e5; margin-bottom: 10px">
                    <p><b>Status:</b> <span style="color: #e67e22">Tidak Tersedia</span></p>
                    <p><b>Pesan:</b> Pipeline belum diinisialisasi</p>
                </div>
                """
                
            # Tampilkan hardware info jika dalam training
            if torch.cuda.is_available():
                gpu_info = display_gpu_info()
                
                print(f"💻 Info Hardware:")
                print(f"  • GPU: {gpu_info['name']}")
                print(f"  • VRAM: {gpu_info['memory_allocated']:.2f}GB / {gpu_info['total_memory']:.2f}GB")
                print(f"  • CUDA Version: {torch.version.cuda}")
            else:
                print(f"💻 Info Hardware: CPU Only")
            
            # Tampilkan apakah ada checkpoint yang tersedia
            try:
                checkpoint_handler = components_dict.get('checkpoint_handler')
                if checkpoint_handler:
                    checkpoints = checkpoint_handler.list_checkpoints()
                    if any(checkpoints.values()):
                        print("\n📦 Checkpoint tersedia:")
                        for category, ckpts in checkpoints.items():
                            if ckpts:
                                print(f"  • {category.capitalize()}: {len(ckpts)} file")
            except Exception as e:
                print(f"⚠️ Tidak dapat mengakses checkpoint: {str(e)}")
        except Exception as e:
            logger(f"❌ Error saat update status: {str(e)}")
            ui_components['info_text'].value = f"""
            <div style="padding: 10px; border-radius: 5px; background-color: #ffe5e5; margin-bottom: 10px">
                <p><b>Status:</b> <span style="color: #e74c3c">Error</span></p>
                <p><b>Pesan:</b> {str(e)}</p>
            </div>
            """
            print(f"Error saat update status: {str(e)}")

def init_components(config, logger):
    """
    Inisialisasi komponen training dengan status informatif.
    
    Args:
        config: Dictionary konfigurasi
        logger: Logger untuk mencatat aktivitas
        
    Returns:
        Dictionary berisi komponen yang terinisialisasi
    """
    with memory_manager():
        components_dict = {}
        try:
            # 1. Inisialisasi Data Manager
            logger.start("🔄 Memuat dataset dan dataloader...")
            try:
                from smartcash.handlers.data_manager import DataManager
                data_manager = DataManager(
                    config_path='configs/base_config.yaml',
                    data_dir=config.get('data_dir', 'data'),
                    logger=logger
                )
                
                # Tambahkan ke components dictionary
                components_dict['data_manager'] = data_manager
            except ImportError:
                logger.warning("⚠️ DataManager tidak dapat diimpor, mungkin file tidak ditemukan atau path salah")
                return components_dict
            
            # Dapatkan informasi dataset untuk validasi
            dataset_stats = data_manager.get_dataset_stats('train')
            if dataset_stats['image_count'] == 0:
                logger.warning("⚠️ Dataset kosong! Pastikan data telah dipreparasi")
            else:
                logger.info(f"📊 Dataset: {dataset_stats['image_count']} gambar, {dataset_stats['label_count']} label")
                
                # Tampilkan statistik layer
                for layer, count in dataset_stats.get('layer_stats', {}).items():
                    if count > 0:
                        logger.info(f"📊 Layer '{layer}': {count} anotasi")
            
            # 2. Siapkan DataLoader
            dataloaders = {
                'train': data_manager.get_train_loader(
                    batch_size=config.get('training', {}).get('batch_size', 16),
                    num_workers=config.get('model', {}).get('workers', 4)
                ),
                'val': data_manager.get_val_loader(
                    batch_size=config.get('training', {}).get('batch_size', 16),
                    num_workers=config.get('model', {}).get('workers', 4)
                ),
                'test': data_manager.get_test_loader(
                    batch_size=config.get('training', {}).get('batch_size', 16),
                    num_workers=config.get('model', {}).get('workers', 4)
                )
            }
            
            components_dict['dataloaders'] = dataloaders
            logger.success(f"✅ Dataloader siap: {len(dataloaders['train'])} batch training")
            
            # 3. Inisialisasi Model Handler
            logger.start("🤖 Mempersiapkan model...")
            try:
                from smartcash.handlers.model_handler import ModelHandler
                model_handler = ModelHandler(
                    config=config,
                    config_path='configs/base_config.yaml',
                    num_classes=config.get('model', {}).get('num_classes', 17),
                    logger=logger
                )
                components_dict['model_handler'] = model_handler
            except ImportError:
                logger.warning("⚠️ ModelHandler tidak dapat diimpor, mungkin file tidak ditemukan atau path salah")
                return components_dict
            
            # 4. Inisialisasi Checkpoint Handler
            logger.start("💾 Mempersiapkan checkpoint handler...")
            try:
                from smartcash.handlers.checkpoint_handler import CheckpointHandler
                checkpoint_handler = CheckpointHandler(
                    output_dir=config.get('output_dir', 'runs/train') + '/weights',
                    logger=logger
                )
                components_dict['checkpoint_handler'] = checkpoint_handler
            
                # Cek checkpoint yang tersedia
                checkpoints = checkpoint_handler.list_checkpoints()
                if any(checkpoints.values()):
                    for category, ckpts in checkpoints.items():
                        if ckpts:
                            logger.info(f"📦 {category.capitalize()} checkpoint tersedia: {len(ckpts)}")
                else:
                    logger.info("ℹ️ Belum ada checkpoint tersedia")
            except ImportError:
                logger.warning("⚠️ CheckpointHandler tidak dapat diimpor")
                components_dict['checkpoint_handler'] = None
                
            # 5. Inisialisasi Experiment Tracker
            from datetime import datetime
            experiment_name = f"{config['model']['backbone']}_{datetime.now().strftime('%Y%m%d_%H%M')}"
            try:
                from smartcash.utils.experiment_tracker import ExperimentTracker
                tracker = ExperimentTracker(
                    experiment_name=experiment_name,
                    output_dir=config.get('output_dir', 'runs/train') + '/experiments',
                    logger=logger
                )
                components_dict['experiment_tracker'] = tracker
            except ImportError:
                logger.warning("⚠️ ExperimentTracker tidak dapat diimpor")
                components_dict['experiment_tracker'] = None
            
            # 6. Inisialisasi Training Pipeline
            logger.start("🚀 Mempersiapkan pipeline training...")
            try:
                from smartcash.utils.training_pipeline import TrainingPipeline
                pipeline = TrainingPipeline(
                    config=config,
                    model_handler=model_handler,
                    data_manager=data_manager,
                    logger=logger
                )
                components_dict['pipeline'] = pipeline
            except ImportError:
                logger.warning("⚠️ TrainingPipeline tidak dapat diimpor")
                components_dict['pipeline'] = None
                return components_dict
            
            # Simpan komponen untuk penggunaan pada module selanjutnya
            with open('training_components.pkl', 'wb') as f:
                pickle.dump(components_dict, f)
            
            # Simpan config yang digunakan
            with open('config.pkl', 'wb') as f:
                pickle.dump(config, f)
                
            logger.success("✨ Pipeline training berhasil diinisialisasi!")
            
            # Always save logger to components_dict
            components_dict['logger'] = logger
            
            return components_dict
            
        except Exception as e:
            logger.error(f"❌ Gagal menginisialisasi pipeline: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Ensure we always return the components_dict, even in error cases
            components_dict['logger'] = logger
            return components_dict

def setup_training_pipeline_handlers(ui_components, config, logger):
    """
    Setup event handlers untuk komponen UI training pipeline.
    
    Args:
        ui_components: Dictionary berisi komponen UI dari create_training_pipeline_ui()
        config: Dictionary konfigurasi
        logger: Logger untuk mencatat aktivitas
        
    Returns:
        Dictionary berisi komponen yang terinisialisasi
    """
    # 1. Inisialisasi komponen
    components_dict = init_components(config, logger)
    
    # 2. Bind event handlers
    ui_components['check_status_button'].on_click(
        lambda button: on_check_status_button_clicked(ui_components, components_dict)
    )
    
    # 3. Update status awal
    # Panggil handler check status untuk inisialisasi tampilan
    on_check_status_button_clicked(ui_components, components_dict)
    
    # Kembalikan components untuk digunakan oleh module lain
    return components_dict