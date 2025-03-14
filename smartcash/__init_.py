"""
File: smartcash/__init_.py
Deskripsi: Komponen utama SmartCash
"""

# Definisi versi
__version__ = "0.1.0"

# Export main modules
from smartcash.common.config import get_config_manager
from smartcash.common.logger import get_logger

# Konfigurasi dasar sebelum import modul lain
config_manager = get_config_manager()
logger = get_logger("smartcash")

# Export modules utama
try:
    from smartcash.dataset.manager import DatasetManager
    from smartcash.detection.detector import Detector
except ImportError:
    logger.warning("⚠️ Import komponen gagal. Pastikan semua dependencies sudah terinstall.")
    

# File: smartcash/main.py
"""
File: smartcash/main.py
Deskripsi: Entry point utama untuk aplikasi SmartCash
"""

import sys
import argparse
from pathlib import Path

from smartcash.common.config import get_config_manager
from smartcash.common.logger import get_logger
from smartcash.cli import handle_cli_command

def main():
    """
    Fungsi main untuk aplikasi SmartCash.
    Proses argumen command line dan jalankan perintah yang sesuai.
    """
    # Setup logger
    logger = get_logger("smartcash_main")
    
    # Parse argumen CLI
    parser = argparse.ArgumentParser(description="SmartCash: Sistem Deteksi Uang Kertas Rupiah")
    
    # Argument umum
    parser.add_argument('--config', type=str, help="Path file konfigurasi")
    parser.add_argument('--verbose', action='store_true', help="Tampilkan log verbose")
    
    # Subparsers untuk perintah
    subparsers = parser.add_subparsers(dest='command', help='Perintah yang tersedia')
    
    # Perintah 'dataset'
    dataset_parser = subparsers.add_parser('dataset', help='Operasi dataset')
    dataset_parser.add_argument('action', choices=['download', 'validate', 'augment', 'balance', 'split'])
    dataset_parser.add_argument('--data-dir', type=str, help="Direktori dataset")
    
    # Perintah 'train'
    train_parser = subparsers.add_parser('train', help='Training model')
    train_parser.add_argument('--data-dir', type=str, help="Direktori dataset")
    train_parser.add_argument('--weights', type=str, help="Path file weights pre-trained")
    train_parser.add_argument('--epochs', type=int, default=100, help="Jumlah epochs")
    train_parser.add_argument('--batch-size', type=int, default=16, help="Batch size")
    
    # Perintah 'detect'
    detect_parser = subparsers.add_parser('detect', help='Deteksi pada gambar')
    detect_parser.add_argument('source', type=str, help="Path gambar, video, atau direktori")
    detect_parser.add_argument('--model', type=str, required=True, help="Path file model")
    detect_parser.add_argument('--conf', type=float, default=0.25, help="Confidence threshold")
    detect_parser.add_argument('--save', action='store_true', help="Simpan hasil deteksi")
    detect_parser.add_argument('--output', type=str, help="Path output untuk hasil")
    
    # Perintah 'export'
    export_parser = subparsers.add_parser('export', help='Export model')
    export_parser.add_argument('--weights', type=str, required=True, help="Path file weights")
    export_parser.add_argument('--format', type=str, default='onnx', 
                              choices=['onnx', 'torchscript', 'tflite', 'tensorrt'],
                              help="Format model output")
    export_parser.add_argument('--output', type=str, help="Path output untuk model yang diekspor")
    
    # Parse argumen
    args = parser.parse_args()
    
    # Baca konfigurasi jika disediakan
    if args.config:
        try:
            config_manager = get_config_manager()
            config_manager.load_config(args.config)
            logger.info(f"✅ Konfigurasi dimuat dari {args.config}")
        except Exception as e:
            logger.error(f"❌ Gagal memuat konfigurasi: {str(e)}")
            return 1
    
    # Set level log berdasarkan --verbose
    if args.verbose:
        from smartcash.common.logger import LogLevel
        get_logger().level = LogLevel.DEBUG
        logger.debug("🐞 Mode verbose diaktifkan")
    
    # Handle perintah
    if not args.command:
        parser.print_help()
        return 0
        
    try:
        # Delegasi ke CLI handler
        return handle_cli_command(args)
    except KeyboardInterrupt:
        logger.info("🛑 Operasi dibatalkan oleh pengguna")
        return 130  # SIGINT exit code
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())


# File: smartcash/cli.py
"""
File: smartcash/cli.py
Deskripsi: Command-line interface untuk SmartCash
"""

import sys
import time
from pathlib import Path

from smartcash.common.logger import get_logger
from smartcash.common.utils import format_time, ensure_dir

logger = get_logger("smartcash_cli")

def handle_cli_command(args):
    """
    Handle perintah CLI.
    
    Args:
        args: Argumen yang diparsing
        
    Returns:
        Exit code (0 = sukses)
    """
    command = args.command
    
    # Perintah dataset
    if command == 'dataset':
        return handle_dataset_command(args)
        
    # Perintah train
    elif command == 'train':
        return handle_train_command(args)
        
    # Perintah detect
    elif command == 'detect':
        return handle_detect_command(args)
        
    # Perintah export
    elif command == 'export':
        return handle_export_command(args)
        
    else:
        logger.error(f"❌ Perintah tidak dikenal: {command}")
        return 1

def handle_dataset_command(args):
    """Handle perintah dataset."""
    from smartcash.dataset.manager import DatasetManager
    
    start_time = time.time()
    action = args.action
    
    # Inisialisasi DatasetManager
    try:
        data_dir = args.data_dir
        dataset_manager = DatasetManager(data_dir=data_dir if data_dir else None)
        
        # Eksekusi operasi yang sesuai
        if action == 'download':
            logger.info("🚀 Mendownload dataset...")
            dataset_manager.pull_dataset(show_progress=True)
            
        elif action == 'validate':
            logger.info("🔍 Memvalidasi dataset...")
            results = {}
            for split in ['train', 'valid', 'test']:
                logger.info(f"   • Validasi split '{split}'...")
                results[split] = dataset_manager.validate_dataset(split=split)
                
            # Print ringkasan
            logger.info("📊 Ringkasan validasi:")
            for split, result in results.items():
                if 'error' in result:
                    logger.warning(f"   • {split}: {result['error']}")
                else:
                    valid_pct = result.get('valid_percentage', 0)
                    logger.info(f"   • {split}: {valid_pct:.1f}% valid")
            
        elif action == 'augment':
            logger.info("🎨 Mengaugmentasi dataset...")
            result = dataset_manager.augment_dataset(
                split='train',
                augmentation_types=['position', 'lighting', 'combined'],
                num_variations=2
            )
            
            # Print ringkasan
            augmented = result.get('augmented', 0)
            logger.success(f"✅ Augmentasi selesai: {augmented} gambar dibuat")
            
        elif action == 'balance':
            logger.info("⚖️ Menyeimbangkan dataset...")
            result = dataset_manager.balance_by_undersampling(
                split='train',
                target_ratio=1.5
            )
            
            # Print ringkasan
            initial_ratio = result.get('initial_ratio', 0)
            final_ratio = result.get('final_ratio', 0)
            logger.success(
                f"✅ Balancing selesai:\n"
                f"   • Rasio ketidakseimbangan awal: {initial_ratio:.2f}x\n"
                f"   • Rasio ketidakseimbangan akhir: {final_ratio:.2f}x"
            )
            
        elif action == 'split':
            logger.info("✂️ Memecah dataset...")
            result = dataset_manager.split_dataset(
                train_ratio=0.7,
                val_ratio=0.2,
                test_ratio=0.1
            )
            
            # Print ringkasan
            logger.success(
                f"✅ Split dataset selesai:\n"
                f"   • Train: {result.get('train', 0)} gambar\n"
                f"   • Valid: {result.get('valid', 0)} gambar\n"
                f"   • Test: {result.get('test', 0)} gambar"
            )
        
        else:
            logger.error(f"❌ Aksi dataset tidak dikenal: {action}")
            return 1
            
        # Print elapsed time
        elapsed_time = time.time() - start_time
        logger.info(f"⏱️ Waktu eksekusi: {format_time(elapsed_time)}")
        
        return 0
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        return 1

def handle_train_command(args):
    """Handle perintah train."""
    from smartcash.detection.services.training import TrainingService
    
    start_time = time.time()
    
    # Inisialisasi TrainingService
    try:
        data_dir = args.data_dir
        weights = args.weights
        epochs = args.epochs
        batch_size = args.batch_size
        
        # Buat konfigurasi training
        config = {
            'data_dir': data_dir,
            'weights': weights,
            'epochs': epochs,
            'batch_size': batch_size
        }
        
        # Inisialisasi training service
        training_service = TrainingService(config)
        
        # Jalankan training
        logger.info("🚀 Memulai training...")
        results = training_service.train()
        
        # Print ringkasan
        best_epoch = results.get('best_epoch', 0)
        best_map = results.get('best_map', 0)
        logger.success(
            f"✅ Training selesai:\n"
            f"   • Epoch terbaik: {best_epoch}\n"
            f"   • mAP@0.5: {best_map:.4f}"
        )
        
        # Print elapsed time
        elapsed_time = time.time() - start_time
        logger.info(f"⏱️ Waktu eksekusi: {format_time(elapsed_time)}")
        
        return 0
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        return 1

def handle_detect_command(args):
    """Handle perintah detect."""
    import cv2
    from smartcash.detection.detector import Detector
    
    start_time = time.time()
    
    # Parse argumen
    source = args.source
    model_path = args.model
    conf_threshold = args.conf
    save_results = args.save
    output_path = args.output
    
    # Inisialisasi Detector
    try:
        # Buat konfigurasi detector
        config = {
            'confidence_threshold': conf_threshold
        }
        
        # Load model
        logger.info(f"📂 Loading model dari {model_path}...")
        detector = Detector(model_path, config=config)
        
        # Tentukan source type
        source_path = Path(source)
        is_image = source_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']
        is_video = source_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.webm']
        is_dir = source_path.is_dir()
        
        # Setup output path
        if save_results:
            output_path = output_path or 'output'
            ensure_dir(output_path)
            logger.info(f"💾 Hasil akan disimpan di {output_path}")
        
        # Proses berdasarkan tipe
        if is_image:
            # Proses satu gambar
            logger.info(f"🔍 Mendeteksi objek pada gambar {source}...")
            image = cv2.imread(source)
            if image is None:
                logger.error(f"❌ Gagal membaca gambar: {source}")
                return 1
                
            # Konversi BGR ke RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Deteksi
            results = detector.detect(image_rgb)
            
            # Proses hasil
            total_detections = sum(len(dets) for layer, dets in results.items())
            logger.info(f"📊 {total_detections} objek terdeteksi")
            
            # Tampilkan untuk setiap layer
            for layer, detections in results.items():
                if detections:
                    logger.info(f"   • Layer {layer}: {len(detections)} deteksi")
            
            # Simpan hasil jika diminta
            if save_results:
                # Visualisasi hasil
                from smartcash.detection.services.visualization import BBoxVisualizer
                visualizer = BBoxVisualizer(detector.get_class_names())
                vis_image = visualizer.draw_detections(image_rgb, results)
                
                # Konversi RGB ke BGR untuk disimpan
                vis_image_bgr = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
                
                # Tentukan output path
                result_path = Path(output_path) / f"{source_path.stem}_result{source_path.suffix}"
                cv2.imwrite(str(result_path), vis_image_bgr)
                logger.info(f"💾 Hasil disimpan ke {result_path}")
                
        elif is_video:
            logger.error("❌ Proses video belum diimplementasikan")
            return 1
            
        elif is_dir:
            # Proses semua gambar dalam direktori
            logger.info(f"🔍 Mendeteksi objek pada semua gambar di {source}...")
            
            # Cari semua file gambar
            image_files = []
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                image_files.extend(list(source_path.glob(f"*{ext}")))
                
            if not image_files:
                logger.error(f"❌ Tidak ada file gambar di {source}")
                return 1
                
            logger.info(f"🔍 Menemukan {len(image_files)} file gambar")
            
            # Inisialisasi visualizer jika perlu menyimpan hasil
            visualizer = None
            if save_results:
                from smartcash.detection.services.visualization import BBoxVisualizer
                visualizer = BBoxVisualizer(detector.get_class_names())
            
            # Proses setiap gambar
            total_detections = 0
            processed_count = 0
            
            for img_path in image_files:
                try:
                    # Baca gambar
                    image = cv2.imread(str(img_path))
                    if image is None:
                        logger.warning(f"⚠️ Gagal membaca gambar: {img_path}")
                        continue
                        
                    # Konversi BGR ke RGB
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # Deteksi
                    results = detector.detect(image_rgb)
                    
                    # Count detections
                    img_detections = sum(len(dets) for layer, dets in results.items())
                    total_detections += img_detections
                    
                    # Simpan hasil jika diminta
                    if save_results and visualizer:
                        # Visualisasi hasil
                        vis_image = visualizer.draw_detections(image_rgb, results)
                        
                        # Konversi RGB ke BGR untuk disimpan
                        vis_image_bgr = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
                        
                        # Tentukan output path
                        result_path = Path(output_path) / f"{img_path.stem}_result{img_path.suffix}"
                        cv2.imwrite(str(result_path), vis_image_bgr)
                    
                    processed_count += 1
                except Exception as e:
                    logger.warning(f"⚠️ Error memproses {img_path}: {str(e)}")
            
            # Print summary
            logger.success(
                f"✅ Deteksi selesai:\n"
                f"   • Gambar diproses: {processed_count}/{len(image_files)}\n"
                f"   • Total objek terdeteksi: {total_detections}\n"
                f"   • Rata-rata per gambar: {total_detections/max(1, processed_count):.1f}"
            )
        
        else:
            logger.error(f"❌ Source tidak valid: {source}")
            return 1
        
        # Print elapsed time
        elapsed_time = time.time() - start_time
        logger.info(f"⏱️ Waktu eksekusi: {format_time(elapsed_time)}")
        
        return 0
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        return 1

def handle_export_command(args):
    """Handle perintah export."""
    from smartcash.detection.utils.optimization import ModelExporter
    
    start_time = time.time()
    
    # Parse argumen
    weights_path = args.weights
    export_format = args.format
    output_path = args.output
    
    # Export model
    try:
        # Load model
        logger.info(f"📂 Loading weights dari {weights_path}...")
        
        # Inisialisasi exporter dan export model
        exporter = ModelExporter(weights_path)
        
        if export_format == 'onnx':
            # Export ke ONNX
            logger.info("🔄 Mengexport model ke format ONNX...")
            output_path = output_path or f"{Path(weights_path).stem}.onnx"
            exported_path = exporter.export_to_onnx(output_path)
            
        elif export_format == 'torchscript':
            # Export ke TorchScript
            logger.info("🔄 Mengexport model ke format TorchScript...")
            output_path = output_path or f"{Path(weights_path).stem}_torchscript.pt"
            exported_path = exporter.export_to_torchscript(output_path)
            
        elif export_format == 'tflite':
            # Export ke TFLite
            logger.info("🔄 Mengexport model ke format TFLite...")
            output_path = output_path or f"{Path(weights_path).stem}.tflite"
            exported_path = exporter.export_to_tflite(output_path)
            
        elif export_format == 'tensorrt':
            # Export ke TensorRT
            logger.info("🔄 Mengexport model ke format TensorRT...")
            output_path = output_path or f"{Path(weights_path).stem}.engine"
            exported_path = exporter.export_to_tensorrt(output_path)
            
        else:
            logger.error(f"❌ Format export tidak dikenal: {export_format}")
            return 1
        
        # Print summary
        logger.success(f"✅ Model berhasil diekspor ke {exported_path}")
        
        # Print elapsed time
        elapsed_time = time.time() - start_time
        logger.info(f"⏱️ Waktu eksekusi: {format_time(elapsed_time)}")
        
        return 0
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        return 1