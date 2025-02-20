#!/usr/bin/env python3

# File: smartcash/__main__.py
# Author: Alfrida Sabar
# Deskripsi: Entry point untuk aplikasi SmartCash dengan dukungan CLI dan TUI

import os
import sys
import curses
import argparse
from pathlib import Path
from typing import Optional, Dict, Any

from smartcash.interface.app import SmartCashApp
from smartcash.utils.logger import SmartCashLogger
from smartcash.cli.configuration_manager import ConfigurationManager

# Setup logger
logger = SmartCashLogger("smartcash")

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="SmartCash - Sistem Deteksi Nilai Mata Uang Rupiah",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='configs/base_config.yaml',
        help='Path ke file konfigurasi'
    )
    
    # Mode arguments
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        '--train',
        action='store_true',
        help='Jalankan mode training via CLI'
    )
    mode_group.add_argument(
        '--eval',
        action='store_true',
        help='Jalankan mode evaluasi via CLI'
    )
    mode_group.add_argument(
        '--detect',
        action='store_true',
        help='Jalankan mode deteksi via CLI'
    )
    
    # Training arguments
    train_group = parser.add_argument_group('Training arguments')
    train_group.add_argument(
        '--data-source',
        choices=['local', 'roboflow'],
        help='Sumber dataset untuk training'
    )
    train_group.add_argument(
        '--backbone',
        choices=['cspdarknet', 'efficientnet'],
        help='Backbone architecture'
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
    
    # Evaluation arguments
    eval_group = parser.add_argument_group('Evaluation arguments')
    eval_group.add_argument(
        '--weights',
        type=str,
        help='Path ke model weights'
    )
    eval_group.add_argument(
        '--test-data',
        type=str,
        help='Path ke data testing'
    )
    
    # Detection arguments
    detect_group = parser.add_argument_group('Detection arguments')
    detect_group.add_argument(
        '--source',
        type=str,
        help='Path ke gambar/video atau 0 untuk webcam'
    )
    detect_group.add_argument(
        '--conf-thres',
        type=float,
        default=0.25,
        help='Confidence threshold'
    )
    
    return parser.parse_args()

def validate_config_path(config_path: str) -> Path:
    """
    Validate configuration file path.
    
    Args:
        config_path: Path ke file konfigurasi
        
    Returns:
        Path: Path yang sudah divalidasi
        
    Raises:
        FileNotFoundError: Jika file tidak ditemukan
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"File konfigurasi tidak ditemukan: {config_path}")
    return path

def run_cli_mode(args: argparse.Namespace, config_manager: ConfigurationManager) -> None:
    """
    Run SmartCash in CLI mode.
    
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
            if not args.weights:
                logger.error("❌ --weights diperlukan untuk mode evaluasi")
                sys.exit(1)
                
            # Start evaluation
            from smartcash.handlers.evaluation_handler import EvaluationHandler
            logger.info("📊 Memulai evaluasi...")
            
            evaluator = EvaluationHandler(config=config_manager.current_config)
            results = evaluator.evaluate(
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
            detector.detect(
                source=args.source,
                conf_thres=args.conf_thres
            )
            
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        sys.exit(1)

def run_tui_mode(config_path: Path) -> None:
    """
    Run SmartCash in TUI mode.
    
    Args:
        config_path: Path ke file konfigurasi
    """
    try:
        app = SmartCashApp(config_path)
        curses.wrapper(app.run)
    except KeyboardInterrupt:
        print("\nKeluar dari SmartCash...")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {str(e)}")
        sys.exit(1)

def main() -> None:
    """Main entry point."""
    try:
        # Parse arguments
        args = parse_args()
        
        # Validate config path
        config_path = validate_config_path(args.config)
        
        # Initialize ConfigurationManager
        config_manager = ConfigurationManager(str(config_path))
        
        # Determine run mode
        if any([args.train, args.eval, args.detect]):
            run_cli_mode(args, config_manager)
        else:
            run_tui_mode(config_path)
            
    except FileNotFoundError as e:
        print(f"\nError: {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"\nError tidak terduga: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()