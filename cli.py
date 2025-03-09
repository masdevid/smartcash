#!/usr/bin/env python
"""
File: smartcash/cli.py
Author: Alfrida Sabar
Deskripsi: Command Line Interface untuk SmartCash dengan fitur kritis 
yang dioptimalkan dengan struktur proyek terbaru
"""

import os
import sys
import argparse
import logging
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Import handlers sesuai struktur proyek terbaru
from smartcash.handlers.model import ModelManager
from smartcash.handlers.dataset import DatasetManager
from smartcash.handlers.evaluation import EvaluationManager
from smartcash.handlers.detection import DetectionManager
from smartcash.handlers.preprocessing import PreprocessingManager
from smartcash.config import get_config_manager
from smartcash.exceptions import SmartCashError, ErrorHandler

# Setup logging dasar
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("smartcash.cli")

def setup_dirs():
    """Buat direktori yang diperlukan jika belum ada."""
    dirs = [
        "logs", "configs", "runs", "pretrained",
        "data/train/images", "data/train/labels",
        "data/valid/images", "data/valid/labels",
        "data/test/images", "data/test/labels"
    ]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    # Buat file gitkeep agar folder kosong tetap terlacak di git
    for dir_path in dirs:
        gitkeep_path = os.path.join(dir_path, ".gitkeep")
        if not os.path.exists(gitkeep_path):
            with open(gitkeep_path, "w") as f:
                f.write("")

def get_log_file() -> str:
    """Dapatkan path file log yang sesuai."""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    return os.path.join(log_dir, f"cli_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

def setup_logger(debug_mode: bool) -> None:
    """Setup logger untuk CLI."""
    # Tambahkan file handler
    file_handler = logging.FileHandler(get_log_file())
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    logging.getLogger().addHandler(file_handler)
    
    # Setup level logging berdasarkan mode debug
    if debug_mode:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("🔍 Mode debug diaktifkan")
    else:
        logging.getLogger().setLevel(logging.INFO)

def train(args):
    """Menjalankan pelatihan model."""
    try:
        logger.info(f"🏋️ Memulai pelatihan dengan konfigurasi: {args.config}")
        
        # Gunakan ConfigManager untuk mengelola konfigurasi
        config_manager = get_config_manager(args.config)
        config = config_manager.get_config()
        
        # Override config dari argumen CLI
        if args.epochs:
            config_manager.set("training.epochs", args.epochs)
        if args.batch_size:
            config_manager.set("training.batch_size", args.batch_size)
        if args.backbone:
            config_manager.set("model.backbone", args.backbone)
            
        # Inisialisasi ModelManager
        model_manager = ModelManager(config=config)
        
        # Setup DatasetManager untuk memuat data
        dataset_manager = DatasetManager(config=config)
        train_loader = dataset_manager.get_train_loader()
        val_loader = dataset_manager.get_val_loader()
        
        # Training model
        if args.resume:
            logger.info(f"🔄 Melanjutkan pelatihan dari checkpoint: {args.resume}")
            results = model_manager.train_model(
                train_loader=train_loader,
                val_loader=val_loader,
                resume_from=args.resume
            )
        else:
            results = model_manager.train_model(
                train_loader=train_loader,
                val_loader=val_loader
            )
        
        logger.info(f"✅ Pelatihan selesai. Hasil: {results}")
        
    except Exception as e:
        error_handler = ErrorHandler()
        error_handler.handle(e, exit_on_error=False)
        logger.error(f"❌ Pelatihan gagal: {str(e)}")
        return 1
    
    return 0

def evaluate(args):
    """Menjalankan evaluasi model."""
    try:
        logger.info(f"📊 Memulai evaluasi model: {args.model}")
        
        # Gunakan ConfigManager dengan default
        config_manager = get_config_manager()
        config = config_manager.get_config()
        
        # Inisialisasi EvaluationManager
        eval_manager = EvaluationManager(config=config)
        
        # Evaluasi model
        results = eval_manager.evaluate_model(
            model_path=args.model,
            dataset_split=args.split,
            conf_threshold=args.conf
        )
        
        # Simpan hasil jika diperlukan
        if args.output:
            logger.info(f"💾 Menyimpan hasil evaluasi ke: {args.output}")
            eval_manager.save_results(results, args.output)
        
        # Tampilkan hasil
        logger.info("📑 Hasil Evaluasi:")
        for metric, value in results.items():
            if isinstance(value, float):
                logger.info(f"  • {metric}: {value:.4f}")
            else:
                logger.info(f"  • {metric}: {value}")
                
    except Exception as e:
        error_handler = ErrorHandler()
        error_handler.handle(e, exit_on_error=False)
        logger.error(f"❌ Evaluasi gagal: {str(e)}")
        return 1
    
    return 0

def detect(args):
    """Menjalankan deteksi pada gambar."""
    try:
        logger.info(f"🔍 Mendeteksi mata uang pada: {args.source}")
        
        # Gunakan ConfigManager dengan default
        config_manager = get_config_manager()
        config = config_manager.get_config()
        
        # Inisialisasi DetectionManager
        detection_manager = DetectionManager(config=config)
        
        # Jalankan deteksi
        results = detection_manager.detect(
            source=args.source,
            model_path=args.model,
            conf_threshold=args.conf,
            visualize=args.visualize,
            output_dir=args.output
        )
        
        # Tampilkan hasil jika tidak banyak
        if isinstance(results, dict) and 'detections' in results:
            num_detections = len(results['detections'])
            logger.info(f"✅ Deteksi berhasil: {num_detections} objek terdeteksi")
            
            if num_detections <= 10:  # Batasi output untuk keterbacaan
                for i, det in enumerate(results['detections']):
                    logger.info(f"  • Deteksi #{i+1}: {det['class_name']} ({det['confidence']:.2f})")
            
            if args.output:
                logger.info(f"💾 Hasil deteksi disimpan di: {args.output}")
                
    except Exception as e:
        error_handler = ErrorHandler()
        error_handler.handle(e, exit_on_error=False)
        logger.error(f"❌ Deteksi gagal: {str(e)}")
        return 1
    
    return 0

def augment(args):
    """Menjalankan augmentasi dataset."""
    try:
        logger.info(f"🔄 Augmentasi dataset: split={args.split}")
        
        # Gunakan ConfigManager dengan default
        config_manager = get_config_manager()
        config = config_manager.get_config()
        
        # Inisialisasi PreprocessingManager
        preprocessing_manager = PreprocessingManager(config=config)
        
        # Parse daftar teknik augmentasi
        augmentation_types = args.techniques.split(',') if args.techniques else ["combined"]
        
        # Jalankan augmentasi
        results = preprocessing_manager.augment_dataset(
            split=args.split,
            augmentation_types=augmentation_types,
            num_variations=args.factor,
            validate_results=args.validate
        )
        
        if results and 'num_augmented' in results:
            logger.info(f"✅ Augmentasi selesai. {results['num_augmented']} gambar diaugmentasi")
            
    except Exception as e:
        error_handler = ErrorHandler()
        error_handler.handle(e, exit_on_error=False)
        logger.error(f"❌ Augmentasi gagal: {str(e)}")
        return 1
    
    return 0

def download(args):
    """Download dataset dari Roboflow."""
    try:
        logger.info(f"⬇️ Mendownload dataset dari Roboflow: {args.workspace}/{args.project}")
        
        # Gunakan ConfigManager dengan default
        config_manager = get_config_manager()
        config = config_manager.get_config()
        
        # Override konfigurasi Roboflow
        config['data']['roboflow'] = {
            'api_key': args.api_key,
            'workspace': args.workspace,
            'project': args.project,
            'version': args.version
        }
        
        # Inisialisasi DatasetManager
        dataset_manager = DatasetManager(config=config)
        
        # Download dataset
        results = dataset_manager.download_dataset(
            output_format=args.format
        )
        
        logger.info(f"✅ Dataset berhasil didownload!")
            
    except Exception as e:
        error_handler = ErrorHandler()
        error_handler.handle(e, exit_on_error=False)
        logger.error(f"❌ Download dataset gagal: {str(e)}")
        return 1
    
    return 0

def export_model(args):
    """Mengekspor model ke format lain."""
    try:
        logger.info(f"📦 Mengekspor model {args.model} ke format {args.format}")
        
        # Gunakan ConfigManager dengan default
        config_manager = get_config_manager()
        config = config_manager.get_config()
        
        # Inisialisasi ModelManager
        model_manager = ModelManager(config=config)
        
        # Parse input shape jika disediakan
        if args.input_shape:
            input_shape = tuple(map(int, args.input_shape.split(',')))
        else:
            input_shape = (640, 640)
        
        # Export model
        output_path = model_manager.export_model(
            model_path=args.model,
            format=args.format,
            output_path=args.output,
            input_shape=input_shape,
            optimize=args.optimize
        )
        
        logger.info(f"✅ Model berhasil diekspor ke: {output_path}")
            
    except Exception as e:
        error_handler = ErrorHandler()
        error_handler.handle(e, exit_on_error=False)
        logger.error(f"❌ Export model gagal: {str(e)}")
        return 1
    
    return 0

def main():
    """Entry point utama CLI."""
    parser = argparse.ArgumentParser(description="SmartCash CLI: Command Line Interface")
    parser.add_argument("--debug", action="store_true", help="Tampilkan pesan debug")
    
    # Setup subparsers untuk commands
    subparsers = parser.add_subparsers(dest="command", help="Perintah yang tersedia")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Melatih model")
    train_parser.add_argument("--config", default="configs/base_config.yaml", help="Path ke file konfigurasi")
    train_parser.add_argument("--resume", help="Resume pelatihan dari checkpoint")
    train_parser.add_argument("--epochs", type=int, help="Jumlah epoch")
    train_parser.add_argument("--batch-size", type=int, help="Ukuran batch")
    train_parser.add_argument("--backbone", help="Backbone model (efficientnet_b4, cspdarknet)")
    
    # Evaluate command
    eval_parser = subparsers.add_parser("eval", help="Evaluasi model")
    eval_parser.add_argument("--model", required=True, help="Path ke model checkpoint")
    eval_parser.add_argument("--split", default="test", choices=["train", "valid", "test"], help="Split dataset untuk evaluasi")
    eval_parser.add_argument("--conf", type=float, default=0.25, help="Threshold confidence")
    eval_parser.add_argument("--output", help="Path untuk menyimpan hasil evaluasi")
    
    # Detect command
    detect_parser = subparsers.add_parser("detect", help="Deteksi mata uang pada gambar")
    detect_parser.add_argument("--model", required=True, help="Path ke model checkpoint")
    detect_parser.add_argument("--source", required=True, help="Path ke gambar atau direktori gambar")
    detect_parser.add_argument("--conf", type=float, default=0.25, help="Threshold confidence")
    detect_parser.add_argument("--output", help="Direktori untuk menyimpan hasil deteksi")
    detect_parser.add_argument("--visualize", action="store_true", help="Visualisasikan hasil deteksi")
    
    # Augment command
    augment_parser = subparsers.add_parser("augment", help="Augmentasi dataset")
    augment_parser.add_argument("--split", default="train", choices=["train", "valid", "test"], help="Split dataset untuk augmentasi")
    augment_parser.add_argument("--factor", type=int, default=2, help="Jumlah variasi per gambar")
    augment_parser.add_argument("--techniques", help="Teknik augmentasi dipisahkan koma (combined,lighting,position)")
    augment_parser.add_argument("--validate", action="store_true", help="Validasi hasil augmentasi")
    
    # Download command
    download_parser = subparsers.add_parser("download", help="Download dataset dari Roboflow")
    download_parser.add_argument("--workspace", required=True, help="Workspace Roboflow")
    download_parser.add_argument("--project", required=True, help="Nama project di Roboflow")
    download_parser.add_argument("--version", required=True, help="Versi dataset")
    download_parser.add_argument("--api-key", required=True, help="Roboflow API key")
    download_parser.add_argument("--format", default="yolov5", help="Format dataset (default: yolov5)")
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Ekspor model ke format lain")
    export_parser.add_argument("--model", required=True, help="Path ke model checkpoint")
    export_parser.add_argument("--format", required=True, choices=["onnx", "torchscript"], help="Format ekspor")
    export_parser.add_argument("--output", help="Path output model")
    export_parser.add_argument("--input-shape", help="Bentuk input (contoh: 640,640)")
    export_parser.add_argument("--optimize", action="store_true", help="Optimasi model untuk inferensi")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Setup directories dan logging
    setup_dirs()
    setup_logger(args.debug)
    
    # Handler untuk setiap command
    cmd_handlers = {
        "train": train,
        "eval": evaluate,
        "detect": detect,
        "augment": augment,
        "download": download,
        "export": export_model
    }
    
    if args.command in cmd_handlers:
        exit_code = cmd_handlers[args.command](args)
        sys.exit(exit_code)
    else:
        parser.print_help()
        sys.exit(0)

if __name__ == "__main__":
    main()