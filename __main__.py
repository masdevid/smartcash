# File: smartcash/__main__.py
# Author: Alfrida Sabar
# Deskripsi: Entry point dengan perbaikan handling error import curses

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

# Cek ketersediaan curses sebelum import
try:
    import curses
    CURSES_AVAILABLE = True
except ImportError:
    CURSES_AVAILABLE = False

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
            "  python -m smartcash --eval --weights models/best.pt\n"
            "  python -m smartcash --detect --source images/test.jpg\n"
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
    mode_group.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Jalankan mode interaktif dengan antarmuka TUI'
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
        
    # Cek curses
    if not CURSES_AVAILABLE:
        logger.warning("⚠️ Package 'curses' tidak tersedia, mode TUI tidak akan digunakan")
        if sys.platform == 'win32':
            logger.info("💡 Tip: Pada Windows, install dengan 'pip install windows-curses'")

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
    
    # Set logging level berdasarkan verbosity
    if args.verbose >= 2:
        logger.setLevel("DEBUG")
        logger.info("🔊 Level log: DEBUG")
    elif args.verbose == 1:
        logger.setLevel("INFO")
        logger.info("🔊 Level log: INFO")

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
            
            # Start training
            from smartcash.handlers.training_pipeline import TrainingPipeline
            logger.info("🚀 Memulai training...")
            
            pipeline = TrainingPipeline(config=config_manager.current_config)
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
                
            # Start evaluation
            from smartcash.handlers.evaluation_handler import EvaluationHandler
            logger.info(f"📊 Memulai evaluasi (skenario: {args.scenario})...")
            
            evaluator = EvaluationHandler(config=config_manager.current_config)
            
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
                
            # Start detection
            from smartcash.handlers.detection_handler import DetectionHandler
            logger.info("🔍 Memulai deteksi...")
            
            detector = DetectionHandler(config=config_manager.current_config)
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
            
        elif args.interactive:
            run_tui_mode(config_manager.base_config_path)
            
    except Exception as e:
        error_handler.handle(e)
        sys.exit(1)

def run_tui_mode(config_path: Path) -> None:
    """
    Run SmartCash in TUI mode.
    
    Args:
        config_path: Path ke file konfigurasi
    """
    try:
        if not CURSES_AVAILABLE:
            logger.error("❌ Package 'curses' tidak tersedia, mode TUI tidak dapat dijalankan")
            logger.info("💡 Gunakan mode CLI sebagai alternatif. Contoh: python -m smartcash --train")
            sys.exit(1)
            
        # Import hanya jika curses tersedia
        from smartcash.interface.app import SmartCashApp
        
        app = SmartCashApp(config_path)
        curses.wrapper(app.run)
        
    except KeyboardInterrupt:
        print("\nKeluar dari SmartCash...")
        sys.exit(0)
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
        
        # Determine run mode
        if any([args.train, args.eval, args.detect, args.version]):
            run_cli_mode(args, config_manager)
        else:
            run_tui_mode(config_path)
            
    except KeyboardInterrupt:
        logger.info("\n👋 Program dihentikan oleh pengguna")
        sys.exit(0)
    except Exception as e:
        error_handler.handle(e)
        sys.exit(1)

if __name__ == '__main__':
    main()