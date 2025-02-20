#!/usr/bin/env python3

import curses
import sys
import os
from pathlib import Path
import yaml
import subprocess
import shutil
from typing import List, Callable, Dict, Any, Optional
from datetime import datetime
import psutil
from dotenv import load_dotenv

from smartcash.utils.logger import SmartCashLogger
from smartcash.handlers.dataset_cleanup import DatasetCleanupHandler

# Initialize logger
logger = SmartCashLogger("smartcash-menu")

# Global configuration
config: Dict = {
    'data_source': None,
    'detection_mode': None,
    'backbone': None,
    'training': {
        'batch_size': 32,
        'learning_rate': 0.001,
        'epochs': 100,
        'early_stopping_patience': 10
    }
}

class MenuItem:
    def __init__(self, title: str, action: Callable, description: str = "", category: str = ""):
        self.title = title
        self.action = action
        self.description = description
        self.category = category

class Menu:
    def __init__(self, title: str, items: List[MenuItem]):
        self.title = title
        self.items = items
        self.selected = 0
        self.categories = self._group_by_category()
        
    def _group_by_category(self):
        categories = {}
        for item in self.items:
            if item.category not in categories:
                categories[item.category] = []
            categories[item.category].append(item)
        return categories
        
    def draw(self, stdscr, start_y: int):
        height, width = stdscr.getmaxyx()
        # Draw title
        title_x = (width - len(self.title)) // 2
        stdscr.attron(curses.color_pair(2))
        stdscr.addstr(start_y, title_x, self.title)
        stdscr.attroff(curses.color_pair(2))
        
        current_y = start_y + 2
        item_index = 0
        
        # Draw categorized menu items
        for category, items in self.categories.items():
            if current_y >= height:
                break
                
            # Draw category header
            if category:
                stdscr.attron(curses.color_pair(4))
                stdscr.addstr(current_y, 2, f"=== {category} ===")
                stdscr.attroff(curses.color_pair(4))
                current_y += 1
            
            # Draw items in category
            for item in items:
                if current_y >= height:
                    break
                    
                if item_index == self.selected:
                    stdscr.attron(curses.color_pair(1))
                    stdscr.addstr(current_y, 2, f"> {item.title}")
                    stdscr.attroff(curses.color_pair(1))
                    # Show description
                    if item.description and current_y + 2 < height:
                        stdscr.addstr(current_y + 1, 4, item.description, curses.color_pair(3))
                        current_y += 2
                    else:
                        current_y += 1
                else:
                    stdscr.addstr(current_y, 2, f"  {item.title}")
                    current_y += 1
                    
                item_index += 1
            
            current_y += 1  # Add space between categories
                
    def handle_input(self, key) -> Optional[bool]:
        """Handle menu input.
        
        Returns:
            - True if action was executed successfully
            - False if we should return to previous menu
            - None if no action was taken
        """
        if key == curses.KEY_UP and self.selected > 0:
            self.selected -= 1
            return None
        elif key == curses.KEY_DOWN and self.selected < len(self.items) - 1:
            self.selected += 1
            return None
        elif key in [curses.KEY_ENTER, ord('\n'), ord('\r')]:
            result = self.items[self.selected].action()
            return result if result is not None else True
        return None

def check_system_requirements():
    """Memeriksa persyaratan sistem."""
    requirements = {
        "CPU Cores": psutil.cpu_count(),
        "RAM (GB)": round(psutil.virtual_memory().total / (1024**3), 2),
        "Disk Space (GB)": round(shutil.disk_usage("/").free / (1024**3), 2),
        "GPU": "Checking...",  # You might want to add proper GPU detection
    }
    
    curses.endwin()
    print("\n=== Status Sistem ===")
    for key, value in requirements.items():
        print(f"{key}: {value}")
    
    print("\nStatus Komponen:")
    # Check directories
    dirs_to_check = ['data', 'models', 'configs']
    for dir_name in dirs_to_check:
        status = "✓ Ada" if os.path.exists(dir_name) else "✗ Tidak ada"
        print(f"Folder {dir_name}: {status}")
    
    # Check configuration
    config_status = "✓ Ada" if os.path.exists("configs/base_config.yaml") else "✗ Tidak ada"
    print(f"File konfigurasi: {config_status}")
    
    input("\nTekan Enter untuk kembali ke menu...")
    return True

def show_help():
    """Menampilkan bantuan penggunaan."""
    curses.endwin()
    print("""
=== Panduan Penggunaan SmartCash ===

1. Persiapan Awal
   - Pastikan semua folder yang diperlukan sudah ada
   - Periksa status sistem di menu "Cek Sistem"
   - Siapkan dataset gambar uang rupiah

2. Langkah-Langkah Penggunaan
   a) Persiapan Dataset
      - Pilih menu "Persiapan Dataset"
      - Ikuti petunjuk untuk membersihkan dataset
   
   b) Pelatihan Model
      - Pilih menu "Latih Model"
      - Tunggu hingga proses selesai
   
   c) Pengujian
      - Pilih menu "Evaluasi Model" untuk mengukur akurasi
      - Gunakan menu "Deteksi" untuk mencoba model
   
3. Tips
   - Selalu cek status sistem sebelum memulai
   - Backup dataset sebelum membersihkan
   - Gunakan webcam untuk pengujian langsung

4. Penyelesaian Masalah
   - Jika terjadi error, cek log di folder 'logs'
   - Pastikan ruang disk mencukupi
   - Restart aplikasi jika terjadi masalah
""")
    input("\nTekan Enter untuk kembali ke menu...")
    return True

def show_about():
    """Menampilkan informasi tentang aplikasi."""
    curses.endwin()
    print("""
=== Tentang SmartCash ===

SmartCash adalah sistem deteksi nilai mata uang Rupiah menggunakan 
teknologi kecerdasan buatan (AI) dengan algoritma YOLOv5 dan 
arsitektur EfficientNet-B4.

Fitur Utama:
- Deteksi nilai mata uang Rupiah secara real-time
- Mendukung input dari kamera dan file gambar
- Akurasi tinggi dalam berbagai kondisi

Dikembangkan oleh:
[Alfrida Sabar]

Versi: 1.0.0
Lisensi: MIT

Kontak:
Email: [alfridasabar@gmail.com]
Website: [depodtech.com]
""")
    input("\nTekan Enter untuk kembali ke menu...")
    return True

def backup_data():
    """Membuat backup dataset."""
    curses.endwin()
    try:
        backup_dir = "backup_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        if not os.path.exists("data"):
            print("\nError: Folder 'data' tidak ditemukan!")
            input("\nTekan Enter untuk kembali ke menu...")
            return
            
        print("\nMembuat backup dataset...")
        shutil.copytree("data", backup_dir)
        print(f"\nBackup berhasil dibuat di folder: {backup_dir}")
    except Exception as e:
        print(f"\nError saat backup: {str(e)}")
    input("\nTekan Enter untuk kembali ke menu...")
    return True

def view_logs():
    """Menampilkan log aplikasi."""
    curses.endwin()
    log_file = "logs/smartcash.log"
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            print("\n=== Log Terakhir ===")
            # Show last 20 lines
            lines = f.readlines()[-20:]
            for line in lines:
                print(line.strip())
    else:
        print("\nFile log tidak ditemukan.")
    input("\nTekan Enter untuk kembali ke menu...")
    return True

def load_config() -> Dict[str, Any]:
    """Load configuration from latest file."""
    try:
        config_dir = Path('configs')
        if not config_dir.exists():
            return {}
            
        config_files = sorted(config_dir.glob('train_config_*.yaml'))
        if not config_files:
            return {}
            
        # Get latest config
        latest_config = config_files[-1]
        with open(latest_config, 'r') as f:
            config = yaml.safe_load(f)
            
        return config
        
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        return {}

def run_command(cmd: List[str]) -> None:
    """Run a command and wait for user input before continuing."""
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with exit code {e.returncode}")
    except Exception as e:
        logger.error(f"Error running command: {e}")
    input("\nPress Enter to continue...")

def cleanup_action():
    """Membersihkan dan menyiapkan dataset."""
    curses.endwin()
    try:
        # Check if data directory exists
        if not os.path.exists("data"):
            print("\nError: Folder 'data' tidak ditemukan!")
            print("Silakan buat folder 'data' dan masukkan dataset Anda.")
            input("\nTekan Enter untuk kembali ke menu...")
            return
            
        # Check if config exists
        config_path = "configs/base_config.yaml"
        if not os.path.exists(config_path):
            print(f"\nError: File konfigurasi '{config_path}' tidak ditemukan!")
            input("\nTekan Enter untuk kembali ke menu...")
            return True
            
        print("\nMemulai pembersihan dataset...")
        handler = DatasetCleanupHandler(
            config_path=config_path,
            data_dir="data",
            backup_dir="backup",
            logger=logger
        )
        
        stats = handler.cleanup(
            augmented_only=True,
            create_backup=True
        )
        
        print("\nHasil pembersihan dataset:")
        print(f"- Jumlah file sebelum: {stats['before']['images']} gambar, {stats['before']['labels']} label")
        print(f"- File yang dihapus: {stats['removed']['images']} gambar, {stats['removed']['labels']} label")
        print(f"- Jumlah file setelah: {stats['after']['images']} gambar, {stats['after']['labels']} label")
        
    except Exception as e:
        print(f"\nError saat membersihkan dataset: {str(e)}")
    
    input("\nTekan Enter untuk kembali ke menu...")
    return True

def train_action(stdscr):
    """Melatih model deteksi uang kertas."""
    config = show_training_menu(stdscr)
    if config:
        # Create model and start training
        model = YOLOv5Model(
            backbone_type=config['backbone'],
            num_classes=len(config['layers']),
            layers=config['layers']
        )
        
        # Initialize trainer
        trainer = ModelTrainer(
            model=model,
            config=config,
            logger=SmartCashLogger(__name__)
        )
        
        # Start training
        print("\n🚀 Starting training...")
        trainer.train()
    input("\nTekan Enter untuk kembali ke menu...")
    return True

def evaluate_action(stdscr):
    """Evaluate and compare model performance."""
    config = show_evaluation_menu(stdscr)
    if config:
        try:
            # Initialize evaluator
            evaluator = ModelEvaluator(config)
            
            # Start evaluation
            print("\n📊 Starting evaluation...")
            results = evaluator.evaluate()
            
            # Display results
            print("\nEvaluation Results:")
            for model_path in config['models']:
                model_name = os.path.basename(model_path)
                print(f"\nModel: {model_name}")
                
                model_results = results[model_path]
                for metric, value in model_results.items():
                    if config['metrics'][metric]:
                        print(f"{metric}: {value:.4f}")
            
        except Exception as e:
            print(f"❌ Evaluation error: {str(e)}")
            
    input("\nPress Enter to return to menu...")
    return True

def detect_action():
    # Get latest weights file
    weights_dir = Path("runs/train/exp/weights")
    if not weights_dir.exists() or not list(weights_dir.glob("*.pt")):
        logger.error("No weights found. Please train the model first.")
        input("\nPress Enter to continue...")
        return True
        
    latest_weights = str(max(weights_dir.glob("*.pt"), key=lambda x: x.stat().st_mtime))
    
    # Ask for input source
    curses.endwin()
    print("\nDetection Options:")
    print("1. Use webcam (enter '0')")
    print("2. Path to image/video file")
    source = input("\nEnter source: ").strip()
    
    cmd = ["python", "-m", "smartcash", "detect", 
           "--weights", latest_weights,
           "--source", source]
    run_command(cmd)
    return True

def quit_action():
    return False

def get_user_input(stdscr, prompt: str, timeout: int = 100) -> str:
    """Get user input with timeout to prevent freezing."""
    curses.echo()
    stdscr.timeout(timeout)  # Set timeout in milliseconds
    
    height, width = stdscr.getmaxyx()
    prompt_y = height - 3
    prompt_x = 2
    
    # Clear input area
    stdscr.addstr(prompt_y, 0, " " * (width - 1))
    stdscr.addstr(prompt_y, prompt_x, prompt)
    stdscr.refresh()
    
    # Initialize input buffer
    input_buffer = ""
    cursor_x = len(prompt) + prompt_x
    
    while True:
        try:
            ch = stdscr.getch()
            if ch == -1:  # No input within timeout
                continue
            elif ch in [ord('\n'), ord('\r')]:  # Enter key
                break
            elif ch == ord('\b') or ch == curses.KEY_BACKSPACE:  # Backspace
                if input_buffer:
                    input_buffer = input_buffer[:-1]
                    cursor_x -= 1
                    stdscr.addch(prompt_y, cursor_x, ' ')
                    stdscr.move(prompt_y, cursor_x)
            elif ch == 3:  # Ctrl+C
                raise KeyboardInterrupt
            elif 32 <= ch <= 126:  # Printable characters
                input_buffer += chr(ch)
                stdscr.addch(prompt_y, cursor_x, ch)
                cursor_x += 1
            
            stdscr.refresh()
            
        except curses.error:
            continue
    
    curses.noecho()
    stdscr.timeout(-1)  # Reset timeout
    return input_buffer.strip()

def show_config_status(stdscr, config: Dict[str, Any], start_y: int = 2, start_x: int = 50):
    """Show current configuration status."""
    # Show main configuration
    y = start_y
    stdscr.addstr(y, start_x, "Status Konfigurasi:")
    y += 2
    
    # Data source
    stdscr.addstr(y, start_x, "Sumber Data: ")
    if config.get('data_source'):
        stdscr.attron(curses.color_pair(2))  # Green
        stdscr.addstr(config['data_source'])
        stdscr.attroff(curses.color_pair(2))
    else:
        stdscr.attron(curses.color_pair(1))  # Red
        stdscr.addstr("Belum dipilih")
        stdscr.attroff(curses.color_pair(1))
    y += 1
    
    # Detection mode
    stdscr.addstr(y, start_x, "Mode Deteksi: ")
    if config.get('detection_mode'):
        stdscr.attron(curses.color_pair(2))
        stdscr.addstr(config['detection_mode'])
        stdscr.attroff(curses.color_pair(2))
    else:
        stdscr.attron(curses.color_pair(1))
        stdscr.addstr("Belum dipilih")
        stdscr.attroff(curses.color_pair(1))
    y += 1
    
    # Backbone
    stdscr.addstr(y, start_x, "Arsitektur: ")
    if config.get('backbone'):
        stdscr.attron(curses.color_pair(2))
        stdscr.addstr(config['backbone'])
        stdscr.attroff(curses.color_pair(2))
    else:
        stdscr.attron(curses.color_pair(1))
        stdscr.addstr("Belum dipilih")
        stdscr.attroff(curses.color_pair(1))
    y += 2
    
    # Training parameters
    stdscr.addstr(y, start_x, "Parameter Pelatihan:")
    y += 1
    
    training = config.get('training', {})
    params = [
        ('batch_size', 'Ukuran Batch', '32'),
        ('learning_rate', 'Learning Rate', '0.001'),
        ('epochs', 'Jumlah Epoch', '100'),
        ('early_stopping_patience', 'Early Stopping', '10')
    ]
    
    for param, label, default in params:
        stdscr.addstr(y, start_x + 2, f"{label}: ")
        value = training.get(param)
        if value is not None:
            stdscr.attron(curses.color_pair(2))
            stdscr.addstr(f"{value}")
            stdscr.attroff(curses.color_pair(2))
        else:
            stdscr.attron(curses.color_pair(1))
            stdscr.addstr(f"{default} (default)")
            stdscr.attroff(curses.color_pair(1))
        y += 1

def show_evaluation_menu(stdscr):
    """Show evaluation menu with model comparison."""
    menu_items = [
        MenuItem(
            title="Evaluasi Model Reguler",
            action=lambda: evaluate_regular(stdscr),
            description="Evaluasi model pada dataset testing standar",
            category="Evaluasi"
        ),
        MenuItem(
            title="Evaluasi Skenario Penelitian",
            action=lambda: evaluate_research(stdscr),
            description="Evaluasi model pada skenario penelitian",
            category="Evaluasi"
        ),
        MenuItem(
            title="Kembali",
            action=lambda: False,
            category="Navigasi"
        )
    ]
    
    menu = Menu("Menu Evaluasi", menu_items)
    return show_submenu(stdscr, menu)

def evaluate_regular(stdscr):
    """Evaluasi model pada dataset testing standar."""
    try:
        # Load configuration
        config = load_config()
        if not config:
            show_error(stdscr, "Konfigurasi tidak ditemukan. Silakan latih model terlebih dahulu.")
            return True
            
        # Initialize evaluator
        from smartcash.handlers.evaluation_handler import EvaluationHandler
        evaluator = EvaluationHandler(config=config)
        
        # Run evaluation
        stdscr.clear()
        show_header(stdscr, "Evaluasi Model Reguler")
        stdscr.addstr(5, 2, "🔄 Mengevaluasi model...")
        stdscr.refresh()
        
        results = evaluator.evaluate(eval_type='regular')
        
        # Display results
        stdscr.clear()
        show_header(stdscr, "Hasil Evaluasi")
        
        y = 5
        for model_name, metrics in results.items():
            stdscr.addstr(y, 2, f"📊 Model: {model_name}")
            y += 2
            for metric, value in metrics.items():
                if metric != 'confusion_matrix':
                    stdscr.addstr(y, 4, f"{metric}: {value:.4f}")
                    y += 1
            y += 1
        
        stdscr.addstr(y + 2, 2, "Tekan tombol apa saja untuk kembali...")
        stdscr.refresh()
        stdscr.getch()
        return True
        
    except Exception as e:
        show_error(stdscr, f"Gagal melakukan evaluasi: {str(e)}")
        return True

def evaluate_research(stdscr):
    """Evaluasi model pada skenario penelitian."""
    try:
        # Load configuration
        config = load_config()
        if not config:
            show_error(stdscr, "Konfigurasi tidak ditemukan. Silakan latih model terlebih dahulu.")
            return True
            
        # Initialize evaluator
        from smartcash.handlers.evaluation_handler import EvaluationHandler
        evaluator = EvaluationHandler(config=config)
        
        # Run evaluation
        stdscr.clear()
        show_header(stdscr, "Evaluasi Skenario Penelitian")
        stdscr.addstr(5, 2, "🔬 Menjalankan skenario penelitian...")
        stdscr.refresh()
        
        results = evaluator.evaluate(eval_type='research')
        
        # Display results
        stdscr.clear()
        show_header(stdscr, "Hasil Skenario Penelitian")
        
        y = 5
        df = results['research_results']
        for _, row in df.iterrows():
            stdscr.addstr(y, 2, f"📊 {row['Skenario']}")
            y += 1
            stdscr.addstr(y, 4, f"Akurasi: {row['Akurasi']:.4f}")
            y += 1
            stdscr.addstr(y, 4, f"Precision: {row['Precision']:.4f}")
            y += 1
            stdscr.addstr(y, 4, f"Recall: {row['Recall']:.4f}")
            y += 1
            stdscr.addstr(y, 4, f"F1-Score: {row['F1-Score']:.4f}")
            y += 1
            stdscr.addstr(y, 4, f"mAP: {row['mAP']:.4f}")
            y += 1
            stdscr.addstr(y, 4, f"Waktu Inferensi: {row['Waktu Inferensi']*1000:.1f}ms")
            y += 2
        
        output_dir = Path(config.get('output_dir', 'outputs'))
        stdscr.addstr(y, 2, f"💾 Hasil lengkap disimpan di: {output_dir}/research_results.csv")
        y += 2
        stdscr.addstr(y, 2, "Tekan tombol apa saja untuk kembali...")
        stdscr.refresh()
        stdscr.getch()
        return True
        
    except Exception as e:
        show_error(stdscr, f"Gagal menjalankan skenario penelitian: {str(e)}")
        return True

def show_training_menu(stdscr):
    """Show training configuration menu."""
    menu_items = [
        MenuItem(
            title="Pilih Sumber Data",
            action=lambda: show_data_source_menu(stdscr),
            description="Pilih sumber dataset untuk pelatihan",
            category="Konfigurasi"
        ),
        MenuItem(
            title="Pilih Mode Deteksi",
            action=lambda: show_detection_mode_menu(stdscr),
            description="Pilih mode deteksi lapis tunggal atau banyak",
            category="Konfigurasi"
        ),
        MenuItem(
            title="Pilih Arsitektur Model",
            action=lambda: show_backbone_menu(stdscr),
            description="Pilih arsitektur backbone model",
            category="Konfigurasi"
        ),
        MenuItem(
            title="Konfigurasi Parameter",
            action=lambda: show_training_params_menu(stdscr),
            description="Atur parameter pelatihan",
            category="Konfigurasi"
        ),
        MenuItem(
            title="Mulai Pelatihan",
            action=lambda: start_training(stdscr),
            description="Mulai proses pelatihan model",
            category="Aksi"
        ),
        MenuItem(
            title="Kembali",
            action=lambda: False,
            description="Kembali ke menu utama",
            category="Navigasi"
        )
    ]
    
    menu = Menu("Menu Pelatihan Model", menu_items)
    return show_submenu(stdscr, menu)

def show_submenu(stdscr: Optional[curses.window], menu: Menu) -> bool:
    """Show a submenu and handle input."""
    while True:
        stdscr.clear()
        menu.draw(stdscr, 2)
        show_config_status(stdscr, config)
        stdscr.refresh()
        
        key = stdscr.getch()
        if key == ord('q'):  # Allow 'q' to go back
            return True
            
        result = menu.handle_input(key)
        if result is False:  # Selected "Kembali" or action returned False
            return True
            
        if result is None:  # No action taken
            continue

def start_training(stdscr) -> bool:
    """Start model training."""
    if not all([
        config['data_source'],
        config['detection_mode'],
        config['backbone']
    ]):
        show_error(stdscr, "Silakan lengkapi semua konfigurasi terlebih dahulu")
        return True
        
    # Save configuration
    save_config()
    
    # Start training process
    logger.info("Memulai proses pelatihan...")
    logger.info(f"Konfigurasi: {config}")
    
    # TODO: Implement actual training
    return False

def save_config():
    """Save current configuration to file."""
    config_dir = Path('configs')
    config_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_file = config_dir / f"train_config_{timestamp}.yaml"
    
    with open(config_file, 'w') as f:
        yaml.dump(config, f)
    
    logger.info(f"Konfigurasi disimpan di: {config_file}")

def show_error(stdscr, message: str, timeout_ms: int = 2000):
    """Show error message with red color."""
    height, width = stdscr.getmaxyx()
    y = height - 2
    x = 2
    
    # Clear line
    stdscr.addstr(y, 0, " " * (width - 1))
    
    # Show error
    stdscr.attron(curses.color_pair(1))  # Red
    stdscr.addstr(y, x, f"❌ {message}")
    stdscr.attroff(curses.color_pair(1))
    stdscr.refresh()
    
    # Wait for timeout or key press
    stdscr.timeout(timeout_ms)
    key = stdscr.getch()
    stdscr.timeout(-1)
    
    # Clear message
    stdscr.addstr(y, 0, " " * (width - 1))
    stdscr.refresh()
    
    return key if key != -1 else None

def show_system_status(stdscr):
    """Show system status in header."""
    height, width = stdscr.getmaxyx()
    
    # Get system info
    import psutil
    import torch
    
    cpu_percent = psutil.cpu_percent()
    mem = psutil.virtual_memory()
    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name() if gpu_available else "Tidak tersedia"
    
    # Format status line
    status = [
        f"CPU: {cpu_percent}%",
        f"RAM: {mem.percent}%",
        f"GPU: {gpu_name}"
    ]
    
    # Draw status line
    y = 1
    x = 2
    for i, item in enumerate(status):
        if i > 0:
            # Draw separator
            stdscr.addstr(y, x - 1, "│")
            x += 1
            
        # Draw status item
        if i == 2:  # GPU status
            if gpu_available:
                stdscr.attron(curses.color_pair(2))  # Green
            else:
                stdscr.attron(curses.color_pair(1))  # Red
        else:
            if eval(item.split(":")[1].strip().rstrip("%")) > 80:
                stdscr.attron(curses.color_pair(1))  # Red
            else:
                stdscr.attron(curses.color_pair(2))  # Green
                
        stdscr.addstr(y, x, item)
        
        # Reset color
        for j in range(1, 7):
            stdscr.attroff(curses.color_pair(j))
            
        x += len(item) + 2
    
    # Draw help text
    help_text = "↑↓: Navigasi | Enter: Pilih | Ctrl+C: Keluar"
    help_x = width - len(help_text) - 2
    stdscr.attron(curses.color_pair(6))  # Blue
    stdscr.addstr(y, help_x, help_text)
    stdscr.attroff(curses.color_pair(6))

def set_config(key: str, value: str, **kwargs) -> bool:
    """Set configuration value and return to previous menu."""
    global config
    config[key] = value
    if kwargs:
        config.update(kwargs)
    return True

def show_data_source_menu(stdscr):
    """Show data source selection menu."""
    menu_items = [
        MenuItem(
            title="Dataset Lokal",
            action=lambda: set_config('data_source', 'local'),
            description="Gunakan dataset yang tersimpan lokal",
            category="Sumber Data"
        ),
        MenuItem(
            title="Dataset Roboflow",
            action=lambda: set_config('data_source', 'roboflow'),
            description="Unduh dan gunakan dataset dari Roboflow",
            category="Sumber Data"
        ),
        MenuItem(
            title="Kembali",
            action=lambda: False,
            category="Navigasi"
        )
    ]
    
    menu = Menu("Pilih Sumber Data", menu_items)
    return show_submenu(stdscr, menu)

def show_detection_mode_menu(stdscr):
    """Show detection mode selection menu."""
    menu_items = [
        MenuItem(
            title="Deteksi Lapis Tunggal",
            action=lambda: set_config('detection_mode', 'single', layers=['banknote']),
            description="Deteksi uang kertas saja",
            category="Mode Deteksi"
        ),
        MenuItem(
            title="Deteksi Lapis Banyak",
            action=lambda: set_config('detection_mode', 'multi', layers=['banknote', 'nominal', 'security']),
            description="Deteksi uang kertas, nominal, dan fitur keamanan",
            category="Mode Deteksi"
        ),
        MenuItem(
            title="Kembali",
            action=lambda: False,
            category="Navigasi"
        )
    ]
    
    menu = Menu("Pilih Mode Deteksi", menu_items)
    return show_submenu(stdscr, menu)

def show_backbone_menu(stdscr):
    """Show backbone selection menu."""
    menu_items = [
        MenuItem(
            title="CSPDarknet",
            action=lambda: set_config('backbone', 'cspdarknet'),
            description="Backbone standar YOLOv5",
            category="Arsitektur"
        ),
        MenuItem(
            title="EfficientNet-B4",
            action=lambda: set_config('backbone', 'efficientnet'),
            description="Backbone EfficientNet-B4",
            category="Arsitektur"
        ),
        MenuItem(
            title="Kembali",
            action=lambda: False,
            category="Navigasi"
        )
    ]
    
    menu = Menu("Pilih Arsitektur Model", menu_items)
    return show_submenu(stdscr, menu)

def show_training_params_menu(stdscr):
    """Show training parameters menu."""
    menu_items = [
        MenuItem(
            title="Ukuran Batch",
            action=lambda: set_training_param(stdscr, 'batch_size', 'Masukkan ukuran batch (8-128): ',
                                        lambda x: 8 <= int(x) <= 128),
            description="Jumlah sampel per batch (default: 32)",
            category="Parameter"
        ),
        MenuItem(
            title="Learning Rate",
            action=lambda: set_training_param(stdscr, 'learning_rate', 'Masukkan learning rate (0.0001-0.01): ',
                                        lambda x: 0.0001 <= float(x) <= 0.01),
            description="Tingkat pembelajaran (default: 0.001)",
            category="Parameter"
        ),
        MenuItem(
            title="Jumlah Epoch",
            action=lambda: set_training_param(stdscr, 'epochs', 'Masukkan jumlah epoch (10-1000): ',
                                        lambda x: 10 <= int(x) <= 1000),
            description="Jumlah iterasi pelatihan (default: 100)",
            category="Parameter"
        ),
        MenuItem(
            title="Early Stopping",
            action=lambda: set_training_param(stdscr, 'early_stopping_patience', 'Masukkan nilai patience (5-50): ',
                                        lambda x: 5 <= int(x) <= 50),
            description="Jumlah epoch sebelum berhenti jika tidak ada perbaikan (default: 10)",
            category="Parameter"
        ),
        MenuItem(
            title="Kembali",
            action=lambda: False,
            category="Navigasi"
        )
    ]
    
    menu = Menu("Konfigurasi Parameter Pelatihan", menu_items)
    return show_submenu(stdscr, menu)

def set_training_param(stdscr, param: str, prompt: str, validator) -> bool:
    """Set training parameter with validation."""
    while True:
        try:
            # Get user input
            value = get_user_input(stdscr, prompt)
            if not value:  # User cancelled
                return True
                
            # Validate input
            if validator(float(value) if 'rate' in param else int(value)):
                config['training'][param] = float(value) if 'rate' in param else int(value)
                return True
            else:
                raise ValueError("Nilai di luar rentang yang diizinkan")
                
        except ValueError as e:
            show_error(stdscr, f"Input tidak valid: {str(e)}")
            continue

def show_confirm_exit(stdscr) -> bool:
    """Show exit confirmation dialog."""
    height, width = stdscr.getmaxyx()
    msg = "Apakah Anda yakin ingin keluar? (y/N)"
    y = height // 2
    x = (width - len(msg)) // 2
    
    stdscr.clear()
    stdscr.attron(curses.color_pair(3))
    stdscr.addstr(y, x, msg)
    stdscr.attroff(curses.color_pair(3))
    stdscr.refresh()
    
    while True:
        key = stdscr.getch()
        if key in [ord('y'), ord('Y')]:
            return True
        elif key in [ord('n'), ord('N'), ord('\n'), ord('\r'), curses.KEY_ENTER]:
            return False

def main(stdscr):
    """Main function."""
    # Setup colors
    curses.start_color()
    curses.use_default_colors()
    
    # Initialize color pairs
    curses.init_pair(1, curses.COLOR_RED, -1)     # Error/Not Set
    curses.init_pair(2, curses.COLOR_GREEN, -1)   # Success/Set
    curses.init_pair(3, curses.COLOR_YELLOW, -1)  # Warning/Highlight
    curses.init_pair(4, curses.COLOR_CYAN, -1)    # Info/Title
    curses.init_pair(5, curses.COLOR_MAGENTA, -1) # Special
    curses.init_pair(6, curses.COLOR_BLUE, -1)    # Navigation
    
    # Hide cursor
    curses.curs_set(0)
    
    # Enable keypad
    stdscr.keypad(True)
    
    # Check system requirements
    if not check_system_requirements():
        return
    
    # Main menu items
    menu_items = [
        MenuItem(
            title="Pelatihan Model",
            action=lambda: show_training_menu(stdscr),
            description="Konfigurasi dan mulai pelatihan model",
            category="Model"
        ),
        MenuItem(
            title="Evaluasi Model",
            action=lambda: show_evaluation_menu(stdscr),
            description="Evaluasi performa model",
            category="Model"
        ),
        MenuItem(
            title="Keluar",
            action=lambda: False,
            description="Keluar dari aplikasi",
            category="Sistem"
        )
    ]
    
    menu = Menu("SmartCash - Sistem Deteksi Uang Kertas", menu_items)
    
    while True:
        try:
            stdscr.clear()
            menu.draw(stdscr, 2)
            show_config_status(stdscr, config)
            stdscr.refresh()
            
            key = stdscr.getch()
            if key == ord('q'):
                if show_confirm_exit(stdscr):
                    break
                continue
                
            result = menu.handle_input(key)
            if result is False:  # Exit selected from menu
                if show_confirm_exit(stdscr):
                    break
                continue
                
        except KeyboardInterrupt:
            if show_confirm_exit(stdscr):
                break
            continue
        except Exception as e:
            show_error(stdscr, f"Terjadi kesalahan: {str(e)}")
            stdscr.getch()

if __name__ == '__main__':
    try:
        curses.wrapper(main)
    except KeyboardInterrupt:
        print("\nKeluar dari SmartCash...")
        sys.exit(0)
