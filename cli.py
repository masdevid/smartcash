#!/usr/bin/env python
"""
SmartCash CLI: Command Line Interface untuk SmartCash

Entry point untuk akses fitur-fitur SmartCash melalui command line.
"""

import os
import sys
import argparse
import logging
import yaml
from datetime import datetime

# Import handlers
from handlers.model_handler import ModelHandler
from handlers.data_manager import DataManager
from handlers.evaluation_handler import EvaluationHandler 
from handlers.detection_handler import DetectionHandler
from handlers.checkpoint_handler import CheckpointHandler
from handlers.multilayer_dataset_handler import MultilayerDatasetHandler
from handlers.roboflow_handler import RoboflowHandler

# Import models
from models import yolov5_model

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join("logs", f"cli_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"))
    ]
)
logger = logging.getLogger(__name__)

def setup_dirs():
    """Buat direktori yang diperlukan jika belum ada."""
    os.makedirs("logs", exist_ok=True)
    os.makedirs("configs", exist_ok=True)
    os.makedirs("runs", exist_ok=True)
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("pretrained", exist_ok=True)

def load_config(config_path):
    """Load konfigurasi dari file yaml."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"Error loading config file: {e}")
        sys.exit(1)

def train(args):
    """Menjalankan pelatihan model."""
    logger.info(f"Memulai pelatihan dengan konfigurasi: {args.config}")
    config = load_config(args.config)
    
    # Override config dengan args
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    
    # Initialize model handler
    model_handler = ModelHandler()
    
    # Setup model and training
    model_handler.setup_from_config(config)
    
    # Train model
    if args.resume:
        logger.info(f"Melanjutkan pelatihan dari checkpoint: {args.resume}")
        model_handler.train(resume=args.resume)
    else:
        model_handler.train()
    
    logger.info("Pelatihan selesai")

def evaluate(args):
    """Menjalankan evaluasi model."""
    logger.info(f"Memulai evaluasi model: {args.model}")
    
    # Initialize evaluation handler
    eval_handler = EvaluationHandler()
    
    # Load model
    eval_handler.load_model(args.model)
    
    # Run evaluation
    results = eval_handler.evaluate(args.data)
    
    # Save results if output path is provided
    if args.output:
        logger.info(f"Menyimpan hasil evaluasi ke: {args.output}")
        eval_handler.save_results(results, args.output)
    
    # Print results
    logger.info("Hasil Evaluasi:")
    for metric, value in results.items():
        logger.info(f"{metric}: {value}")

def detect(args):
    """Menjalankan deteksi pada gambar."""
    logger.info(f"Mendeteksi mata uang pada: {args.image}")
    
    # Initialize detection handler
    detect_handler = DetectionHandler()
    
    # Load model
    detect_handler.load_model(args.model)
    
    # Run detection
    results = detect_handler.detect(
        source=args.image,
        conf_thres=args.conf,
        save_results=True if args.output else False,
        output_path=args.output
    )
    
    # Print results
    for i, (img_path, detections) in enumerate(results):
        logger.info(f"Gambar {i+1}: {img_path}")
        for det in detections:
            class_id, conf, bbox = det
            logger.info(f"  - Kelas: {class_id}, Confidence: {conf:.2f}, BBox: {bbox}")

def augment(args):
    """Menjalankan augmentasi dataset."""
    logger.info(f"Augmentasi dataset: {args.data}")
    
    # Initialize data manager
    data_manager = DataManager()
    
    # Run augmentation
    data_manager.augment_dataset(
        dataset_path=args.data,
        output_path=args.output,
        augmentation_factor=args.factor,
        techniques=args.techniques.split(',') if args.techniques else None
    )
    
    logger.info(f"Augmentasi selesai. Hasil disimpan di: {args.output}")

def export(args):
    """Mengekspor model ke format lain."""
    logger.info(f"Mengekspor model {args.model} ke format {args.format}")
    
    # Initialize model handler
    model_handler = ModelHandler()
    
    # Load model
    model_handler.load_model(args.model)
    
    # Export model
    model_handler.export_model(
        format=args.format,
        output_path=args.output,
        input_shape=args.input_shape if args.input_shape else None
    )
    
    logger.info(f"Model berhasil diekspor ke: {args.output}")

def download_dataset(args):
    """Download dataset dari Roboflow."""
    logger.info(f"Mendownload dataset dari Roboflow: {args.dataset}")
    
    # Initialize Roboflow handler
    roboflow_handler = RoboflowHandler()
    
    # Load API key from .env or args
    api_key = args.api_key or os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        logger.error("API key tidak ditemukan. Gunakan --api-key atau set ROBOFLOW_API_KEY di .env")
        sys.exit(1)
    
    # Download dataset
    roboflow_handler.download_dataset(
        dataset_name=args.dataset,
        version=args.version,
        api_key=api_key,
        output_format=args.format,
        output_path=args.output
    )
    
    logger.info(f"Dataset berhasil didownload ke: {args.output}")

def main():
    """Entry point utama CLI."""
    parser = argparse.ArgumentParser(description="SmartCash CLI: Command Line Interface")
    subparsers = parser.add_subparsers(dest="command", help="Perintah yang tersedia")
    
    # Setup debug mode
    parser.add_argument("--debug", action="store_true", help="Tampilkan pesan debug")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Melatih model")
    train_parser.add_argument("--config", required=True, help="Path ke file konfigurasi")
    train_parser.add_argument("--resume", help="Resume pelatihan dari checkpoint")
    train_parser.add_argument("--epochs", type=int, help="Jumlah epoch")
    train_parser.add_argument("--batch-size", type=int, help="Ukuran batch")
    train_parser.set_defaults(func=train)
    
    # Evaluate command
    eval_parser = subparsers.add_parser("eval", help="Evaluasi model")
    eval_parser.add_argument("--model", required=True, help="Path ke model checkpoint")
    eval_parser.add_argument("--data", required=True, help="Path ke dataset test")
    eval_parser.add_argument("--output", help="Path untuk menyimpan hasil evaluasi")
    eval_parser.set_defaults(func=evaluate)
    
    # Detect command
    detect_parser = subparsers.add_parser("detect", help="Deteksi mata uang pada gambar")
    detect_parser.add_argument("--model", required=True, help="Path ke model checkpoint")
    detect_parser.add_argument("--image", required=True, help="Path ke gambar atau direktori gambar")
    detect_parser.add_argument("--output", help="Path untuk menyimpan hasil deteksi")
    detect_parser.add_argument("--conf", type=float, default=0.25, help="Threshold confidence")
    detect_parser.set_defaults(func=detect)
    
    # Augment command
    augment_parser = subparsers.add_parser("augment", help="Augmentasi dataset")
    augment_parser.add_argument("--data", required=True, help="Path ke dataset")
    augment_parser.add_argument("--output", required=True, help="Path output hasil augmentasi")
    augment_parser.add_argument("--factor", type=int, default=2, help="Faktor augmentasi")
    augment_parser.add_argument("--techniques", help="Teknik augmentasi yang dipisahkan koma (rotate,flip,blur,etc)")
    augment_parser.set_defaults(func=augment)
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Ekspor model ke format lain")
    export_parser.add_argument("--model", required=True, help="Path ke model checkpoint")
    export_parser.add_argument("--format", required=True, choices=["onnx", "torchscript", "coreml", "tflite"], help="Format ekspor")
    export_parser.add_argument("--output", required=True, help="Path output model")
    export_parser.add_argument("--input-shape", help="Bentuk input (contoh: 640,640,3)")
    export_parser.set_defaults(func=export)
    
    # Download command
    download_parser = subparsers.add_parser("download", help="Download dataset dari Roboflow")
    download_parser.add_argument("--dataset", required=True, help="Nama dataset di Roboflow")
    download_parser.add_argument("--version", required=True, help="Versi dataset")
    download_parser.add_argument("--api-key", help="Roboflow API key (jika tidak ada di .env)")
    download_parser.add_argument("--format", default="yolov5", help="Format dataset (default: yolov5)")
    download_parser.add_argument("--output", default="data/raw", help="Path output dataset")
    download_parser.set_defaults(func=download_dataset)
    
    args = parser.parse_args()
    setup_dirs()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Jalankan fungsi sesuai command
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()