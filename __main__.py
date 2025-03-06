# File: smartcash/__main__.py
# Author: Alfrida Sabar
# Deskripsi: Entry point dengan perbaikan handling error

import os
import sys
import argparse
from pathlib import Path
from typing import Optional, Dict, Any

from smartcash.utils.logger import SmartCashLogger
from smartcash.cli.configuration_manager import ConfigurationManager
from smartcash.exceptions.handler import ErrorHandler
from smartcash.exceptions.base import ConfigError, ValidationError
from smartcash.utils.debug_helper import DebugHelper

# Import handlers
from smartcash.handlers.training_pipeline import TrainingPipeline
from smartcash.handlers.detection_handler import DetectionHandler
from smartcash.handlers.evaluation_handler import EvaluationHandler
from smartcash.handlers.data_manager import DataManager
from smartcash.handlers.model_handler import ModelHandler
from smartcash.utils.visualization import ResultVisualizer

# Setup logger
logger = SmartCashLogger("smartcash")
error_handler = ErrorHandler("smartcash.cli")

def parse_args() -> argparse.Namespace:
    """Parse command line arguments with improved help messages."""
    parser = argparse.ArgumentParser(
        description="SmartCash - Sistem Deteksi Nilai Mata Uang Rupiah",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Contoh penggunaan:\n"
            "  python -m smartcash --train --backbone efficientnet --data-source local\n"
            "  python -m smartcash --eval --weights models/best.pt --scenario research\n"
            "  python -m smartcash --detect --source images/test.jpg --conf-thres 0.4\n"
        )
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='configs/base_config.yaml',
        help='Path ke file konfigurasi (default: configs/base_config.yaml)'
    )
    
    # Mode arguments
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        '--train', '-t',
        action='store_true',
        help='Jalankan mode training via CLI'
    )
    mode_group.add_argument(
        '--eval', '-e',
        action='store_true',
        help='Jalankan mode evaluasi via CLI'
    )
    mode_group.add_argument(
        '--detect', '-d',
        action='store_true',
        help='Jalankan mode deteksi via CLI'
    )
    
    # Training arguments
    train_group = parser.add_argument_group('Training arguments')
    train_group.add_argument(
        '--data-source',
        choices=['local', 'roboflow'],
        help='Sumber dataset untuk training (local atau roboflow)'
    )
    train_group.add_argument(
        '--backbone',
        choices=['cspdarknet', 'efficientnet'],
        help='Backbone architecture (cspdarknet atau efficientnet)'
    )
    train_group.add_argument(
        '--detection-mode',
        choices=['single', 'multi'],
        help='Mode deteksi (single/multi layer)'
    )
    train_group.add_argument(
        '--batch-size',
        type=int,
        help='Ukuran batch untuk training'
    )
    train_group.add_argument(
        '--epochs',
        type=int,
        help='Jumlah epoch training'
    )
    train_group.add_argument(
        '--learning-rate', '--lr',
        type=float,
        help='Learning rate untuk training'
    )
    train_group.add_argument(
        '--seed',
        type=int,
        help='Random seed untuk reproducibility'
    )
    train_group.add_argument(
        '--resume',
        type=str,
        help='Path ke checkpoint untuk melanjutkan training'
    )
    train_group.add_argument(
        '--pretrained',
        action='store_true',
        help='Gunakan pre-trained weights untuk backbone'
    )
    train_group.add_argument(
        '--augment',
        choices=['none', 'basic', 'advanced'],
        default='basic',
        help='Level augmentasi data (default: basic)'
    )
    train_group.add_argument(
        '--early-stopping',
        type=int,
        metavar='PATIENCE',
        help='Aktifkan early stopping dengan patience tertentu'
    )
    train_group.add_argument(
        '--save-every',
        type=int,
        default=1,
        help='Interval epoch untuk menyimpan checkpoint (default: 1)'
    )
    
    # Evaluation arguments
    eval_group = parser.add_argument_group('Evaluation arguments')
    eval_group.add_argument(
        '--weights', '-w',
        type=str,
        help='Path ke model weights'
    )
    eval_group.add_argument(
        '--test-data',
        type=str,
        help='Path ke data testing'
    )
    eval_group.add_argument(
        '--scenario',
        choices=['regular', 'research', 'all'],
        default='regular',
        help='Skenario evaluasi (regular, research, atau all)'
    )
    eval_group.add_argument(
        '--batch-eval',
        action='store_true',
        help='Evaluasi dalam batch untuk dataset besar'
    )
    eval_group.add_argument(
        '--save-predictions',
        action='store_true',
        help='Simpan hasil prediksi untuk analisis'
    )
    eval_group.add_argument(
        '--metrics',
        nargs='+',
        choices=['accuracy', 'precision', 'recall', 'f1', 'mAP', 'confusion'],
        default=['all'],
        help='Metrik evaluasi yang akan dihitung'
    )
    
    # Detection arguments
    detect_group = parser.add_argument_group('Detection arguments')
    detect_group.add_argument(
        '--source', '-s',
        type=str,
        help='Path ke gambar/video atau 0 untuk webcam'
    )
    detect_group.add_argument(
        '--conf-thres',
        type=float,
        default=0.25,
        help='Confidence threshold (default: 0.25)'
    )
    detect_group.add_argument(
        '--output-dir', '-o',
        type=str,
        default='output',
        help='Direktori output untuk hasil deteksi (default: output)'
    )
    detect_group.add_argument(
        '--save-txt',
        action='store_true',
        help='Simpan hasil deteksi dalam format teks'
    )
    detect_group.add_argument(
        '--save-conf',
        action='store_true',
        help='Simpan nilai confidence bersama hasil deteksi'
    )
    detect_group.add_argument(
        '--hide-labels',
        action='store_true',
        help='Sembunyikan label pada visualisasi'
    )
    detect_group.add_argument(
        '--hide-conf',
        action='store_true',
        help='Sembunyikan confidence pada visualisasi'
    )
    detect_group.add_argument(
        '--line-thickness',
        type=int,
        default=3,
        help='Ketebalan garis bounding box (default: 3)'
    )
    
    # General arguments
    general_group = parser.add_argument_group('General arguments')
    general_group.add_argument(
        '--device',
        choices=['cpu', 'cuda', '0', '1', '2', '3'],
        help='Device untuk komputasi (cpu, cuda, atau nomor GPU)'
    )
    general_group.add_argument(
        '--verbose', '-v',
        action='count',
        default=0,
        help='Tingkat verbosity (bisa diulang untuk meningkatkan level)'
    )
    general_group.add_argument(
        '--version',
        action='store_true',
        help='Tampilkan versi SmartCash dan keluar'
    )
    general_group.add_argument(
        '--debug',
        action='store_true',
        help='Aktifkan mode debug'
    )
    general_group.add_argument(
        '--profile',
        action='store_true',
        help='Aktifkan profiling untuk analisis performa'
    )
    general_group.add_argument(
        '--no-color',
        action='store_true',
        help='Nonaktifkan output berwarna'
    )
    
    return parser.parse_args()

def validate_config_path(config_path: str) -> Path:
    """
    Validate configuration file path dengan pesan error yang lebih jelas.
    
    Args:
        config_path: Path ke file konfigurasi
        
    Returns:
        Path: Path yang sudah divalidasi
        
    Raises:
        ConfigError: Jika file tidak ditemukan atau tidak valid
    """
    path = Path(config_path)
    if not path.exists():
        raise ConfigError(f"File konfigurasi tidak ditemukan: {config_path}")
    
    if not path.is_file() or path.suffix not in ['.yaml', '.yml']:
        raise ConfigError(f"File konfigurasi harus berformat YAML: {config_path}")
    
    return path

def validate_config(config: Dict) -> None:
    """
    Validasi konfigurasi dengan pengecekan yang lebih ketat.
    
    Args:
        config: Dictionary konfigurasi
        
    Raises:
        ValidationError: Jika konfigurasi tidak valid
    """
    # Validasi backbone
    backbone = config.get('backbone')
    if backbone not in ['cspdarknet', 'efficientnet']:
        raise ValidationError(f"Backbone tidak valid: {backbone}")
        
    # Validasi detection mode
    detection_mode = config.get('detection_mode')
    if detection_mode not in ['single', 'multi']:
        raise ValidationError(f"Mode deteksi tidak valid: {detection_mode}")
        
    # Validasi data source
    data_source = config.get('data_source')
    if data_source not in ['local', 'roboflow']:
        raise ValidationError(f"Sumber data tidak valid: {data_source}")
        
    # Validasi training params
    training = config.get('training', {})
    if not isinstance(training.get('batch_size', 16), int):
        raise ValidationError("Batch size harus berupa integer")
    if not isinstance(training.get('epochs', 30), int):
        raise ValidationError("Epochs harus berupa integer")
    if not isinstance(training.get('learning_rate', 0.001), float):
        raise ValidationError("Learning rate harus berupa float")
        
    # Validasi model params
    model = config.get('model', {})
    if not isinstance(model.get('img_size', [640, 640]), list):
        raise ValidationError("Image size harus berupa list [width, height]")
        
    # Validasi output paths
    output_dir = config.get('output_dir', 'results')
    if not isinstance(output_dir, str):
        raise ValidationError("Output directory harus berupa string")
        
    # Validasi device
    device = config.get('device', 'cuda')
    if device not in ['cpu', 'cuda'] and not device.startswith('cuda:'):
        raise ValidationError(f"Device tidak valid: {device}")

def check_dependencies() -> None:
    """Verifikasi dependencies yang diperlukan tersedia."""
    try:
        import torch
        logger.info(f"🔍 PyTorch terdeteksi: {torch.__version__}")
        
        if torch.cuda.is_available():
            devices = [f"cuda:{i} ({torch.cuda.get_device_name(i)})" 
                      for i in range(torch.cuda.device_count())]
            logger.info(f"🚀 GPU tersedia: {', '.join(devices)}")
        else:
            logger.warning("⚠️ GPU tidak tersedia, menggunakan CPU")
            
    except ImportError:
        logger.warning("⚠️ PyTorch tidak terinstal, performa akan terbatas")
    
    try:
        import timm
        logger.info(f"🔍 Timm terdeteksi: {timm.__version__}")
    except ImportError:
        logger.warning("⚠️ Timm tidak terinstal, backbone EfficientNet tidak dapat digunakan")
        
    try:
        import cv2
        logger.info(f"🔍 OpenCV terdeteksi: {cv2.__version__}")
    except ImportError:
        logger.warning("⚠️ OpenCV tidak terinstal, visualisasi akan terbatas")

def show_version() -> None:
    """Tampilkan informasi versi SmartCash."""
    version_info = {
        "SmartCash": "1.0.0",
        "Author": "Alfrida Sabar",
        "Build Date": "2024-03-04",
        "Repository": "https://github.com/masdevid/smartcash"
    }
    
    logger.info("\n📊 Informasi SmartCash")
    for key, value in version_info.items():
        logger.info(f"  {key}: {value}")

def setup_environment(args: argparse.Namespace) -> None:
    """
    Setup environment variables dan konfigurasi runtime.
    
    Args:
        args: Command line arguments
    """
    # Set environment variables berdasarkan argumen
    if args.device:
        if args.device in ['0', '1', '2', '3']:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.device
            logger.info(f"🖥️ Menggunakan GPU: {args.device}")
        elif args.device == 'cpu':
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            logger.info("🖥️ Menggunakan CPU secara paksa")
            
    # Set CUDA flags untuk optimasi
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        logger.info("🚀 CUDA optimizations diaktifkan")
    
    # Set logging level berdasarkan verbosity
    if args.verbose >= 2:
        logger.setLevel("DEBUG")
        logger.info("🔊 Level log: DEBUG")
    elif args.verbose == 1:
        logger.setLevel("INFO")
        logger.info("🔊 Level log: INFO")
        
    # Set random seed untuk reproducibility
    if 'seed' in args:
        seed = args.seed
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        logger.info(f"🎲 Random seed: {seed}")

def run_cli_mode(args: argparse.Namespace, config_manager: ConfigurationManager) -> None:
    """
    Run SmartCash in CLI mode with improved error handling.
    
    Args:
        args: Command line arguments
        config_manager: ConfigurationManager instance
    """
    try:
        if args.train:
            # Update config from CLI args
            if args.data_source:
                config_manager.update('data_source', args.data_source)
            if args.backbone:
                config_manager.update('backbone', args.backbone)
            if args.detection_mode:
                config_manager.update('detection_mode', args.detection_mode)
            if args.batch_size:
                config_manager.update('training.batch_size', args.batch_size)
            if args.epochs:
                config_manager.update('training.epochs', args.epochs)
            if args.learning_rate:
                config_manager.update('training.learning_rate', args.learning_rate)
                
            # Save updated config
            config_manager.save()
            
            # Initialize handlers
            data_manager = DataManager(config=config_manager.current_config)
            model_handler = ModelHandler(config=config_manager.current_config)
            
            # Start training
            logger.info("🚀 Memulai training...")
            
            pipeline = TrainingPipeline(
                config=config_manager.current_config,
                model_handler=model_handler,
                data_manager=data_manager,
                logger=logger
            )
            
            results = pipeline.train()
            
            logger.success(
                f"✨ Training selesai!\n"
                f"📁 Model tersimpan di: {results['train_dir']}\n"
                f"📊 Metrik terbaik:\n"
                f"   • Akurasi: {results['best_metrics']['accuracy']:.4f}\n"
                f"   • F1-Score: {results['best_metrics']['f1']:.4f}\n"
                f"   • mAP: {results['best_metrics']['mAP']:.4f}"
            )
            
        elif args.eval:
            if not args.weights and not args.scenario == 'research':
                logger.error("❌ --weights diperlukan untuk mode evaluasi reguler")
                sys.exit(1)
                
            # Initialize evaluator
            evaluator = EvaluationHandler(config=config_manager.current_config)
            
            logger.info(f"📊 Memulai evaluasi (skenario: {args.scenario})...")
            
            if args.scenario == 'research':
                results = evaluator.evaluate(eval_type='research')
                logger.success(
                    f"✨ Evaluasi skenario penelitian selesai!\n"
                    f"📁 Hasil tersimpan di: {config_manager.current_config.get('output_dir', 'results')}"
                )
            elif args.scenario == 'all':
                # Evaluasi reguler dan penelitian
                reg_results = evaluator.evaluate(eval_type='regular', weights_path=args.weights)
                research_results = evaluator.evaluate(eval_type='research')
                logger.success("✨ Evaluasi semua skenario selesai!")
            else:  # regular
                results = evaluator.evaluate(
                    eval_type='regular',
                    weights_path=args.weights,
                    test_data_path=args.test_data
                )
                
                logger.success(
                    f"✨ Evaluasi selesai!\n"
                    f"📊 Hasil:\n"
                    f"   • Akurasi: {results['accuracy']:.4f}\n"
                    f"   • Precision: {results['precision']:.4f}\n"
                    f"   • Recall: {results['recall']:.4f}\n"
                    f"   • F1-Score: {results['f1']:.4f}\n"
                    f"   • mAP: {results['mAP']:.4f}\n"
                    f"   • Waktu Inferensi: {results['inference_time']*1000:.1f}ms"
                )
                
        elif args.detect:
            if not args.source:
                logger.error("❌ --source diperlukan untuk mode deteksi")
                sys.exit(1)
                
            # Initialize detector
            detector = DetectionHandler(
                config=config_manager.current_config,
                weights_path=args.weights
            )
            
            logger.info("🔍 Memulai deteksi...")
            
            results = detector.detect(
                source=args.source,
                conf_thres=args.conf_thres,
                output_dir=args.output_dir
            )
            
            logger.success(
                f"✨ Deteksi selesai!\n"
                f"📁 Hasil tersimpan di: {args.output_dir}\n"
                f"🎯 Objek terdeteksi: {len(results['detections'])}"
            )
            
        elif args.version:
            show_version()
            
    except Exception as e:
        error_handler.handle(e)
        sys.exit(1)

def main() -> None:
    """Main entry point dengan penanganan error yang lebih baik."""
    try:
        # Parse arguments
        args = parse_args()
        
        # Immediately show version and exit if requested
        if args.version:
            show_version()
            return
        
        # Setup environment
        setup_environment(args)
        
        # Verify dependencies
        check_dependencies()
        
        # Validate config path
        try:
            config_path = validate_config_path(args.config)
        except ConfigError as e:
            logger.error(f"❌ {str(e)}")
            logger.info("💡 Gunakan --config untuk menentukan path konfigurasi yang valid")
            sys.exit(1)
        
        # Initialize ConfigurationManager
        try:
            config_manager = ConfigurationManager(str(config_path))
        except Exception as e:
            logger.error(f"❌ Gagal memuat konfigurasi: {str(e)}")
            sys.exit(1)
        
        # Run CLI mode
        run_cli_mode(args, config_manager)
            
    except KeyboardInterrupt:
        logger.info("\n👋 Program dihentikan oleh pengguna")
        sys.exit(0)
    except Exception as e:
        error_handler.handle(e)
        sys.exit(1)

if __name__ == '__main__':
    main()