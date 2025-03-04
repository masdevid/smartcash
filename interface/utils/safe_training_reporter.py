# File: smartcash/interface/utils/safe_training_reporter.py
# Author: Alfrida Sabar
# Deskripsi: Reporter progres training dengan tampilan rapi dan log yang buat baris baru daripada menumpuk kekanan

import sys
import os
import time
from typing import Dict, Optional, List, Any, Union
import torch
import psutil
from datetime import datetime
from termcolor import colored
from tqdm import tqdm
import shutil  # Untuk mendapatkan ukuran terminal
from smartcash.utils.log_filter import PyTorchLogFilter
from smartcash.utils.logger import get_logger


class SafeTrainingReporter:
    """
    Reporter progress training dengan tampilan yang lebih bersih:
    - Log baris per baris (tidak menumpuk ke kanan)
    - Status sistem dan konfigurasi yang jelas
    - Progress bar yang informatif
    """
    
    # ANSI color codes untuk output terminal
    COLORS = {
        'header': '\033[95m',
        'blue': '\033[94m',
        'cyan': '\033[96m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'red': '\033[91m',
        'bold': '\033[1m',
        'underline': '\033[4m',
        'end': '\033[0m'
    }
    
    # Karakter untuk membuat panel
    BOX_CHARS = {
        'h_line': '─',
        'v_line': '│',
        'tl_corner': '┌',
        'tr_corner': '┐',
        'bl_corner': '└',
        'br_corner': '┘',
        'l_t': '├',
        'r_t': '┤',
        'b_t': '┬',
        't_b': '┴',
        'cross': '┼'
    }

    def __init__(
        self, 
        display_manager: Optional = None,
        show_memory: bool = True,
        show_gpu: bool = True
    ):
        """
        Inisialisasi reporter training.
        
        Args:
            display_manager: Display manager untuk TUI (opsional)
            show_memory: Tampilkan penggunaan memori
            show_gpu: Tampilkan penggunaan GPU jika tersedia
        """
        self.display_manager = display_manager
        self.show_memory = show_memory
        self.show_gpu = show_gpu and torch.cuda.is_available()
        self.log_buffer = []
        self.metrics_history = []
        self.is_tui_mode = display_manager is not None
        self.start_time = None
        self.has_gpu = torch.cuda.is_available()
        self.progress_bars = {}
        
        # Dapatkan ukuran terminal 
        self.terminal_width, self.terminal_height = self._get_terminal_size()
        
        # Status data
        self.status_data = {
            'sistem': {},
            'konfigurasi': {},
            'metrik': {}
        }
        
        # Catat waktu mulai
        self.epoch_times = {}

        # Inisialisasi logger yang hanya mencatat ke file
        self.logger = get_logger("training_reporter", log_to_console=False, log_to_file=True)
        PyTorchLogFilter.setup(log_file='logs/smartcash.log', filtered_modules=['torch.distributed','torch._dynamo'])

    def _get_terminal_size(self) -> tuple:
        """Dapatkan ukuran terminal saat ini."""
        try:
            columns, rows = shutil.get_terminal_size()
            # Set minimum width dan height
            columns = max(columns, 80)
            rows = max(rows, 24)
            return columns, rows
        except:
            # Nilai default jika gagal mendapatkan ukuran terminal
            return 100, 30
        
    def _supports_color(self) -> bool:
        """Cek apakah terminal mendukung warna."""
        plat = sys.platform
        supported_platform = plat != 'win32' or 'ANSICON' in os.environ
        is_a_tty = hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()
        return supported_platform and is_a_tty

    def _format_message(self, message: str, style: Optional[str] = None) -> str:
        """Format pesan dengan warna jika didukung."""
        if not self._supports_color():
            return message
            
        if style and style in self.COLORS:
            return f"{self.COLORS[style]}{message}{self.COLORS['end']}"
        return message
    
    def _clear_line(self):
        """Bersihkan baris saat ini."""
        print("\r\033[K", end="")
    
    def _draw_header(self):
        """Gambar header untuk sesi training."""
        # Bersihkan baris sebelumnya
        self._clear_line()
        
        # Buat header dengan box sederhana
        header_text = " SmartCash Training Progress "
        header_width = self.terminal_width - 2
        padding = (header_width - len(header_text)) // 2
        
        print(self._format_message("┌" + "─" * header_width + "┐", "bold"))
        print(self._format_message("│" + " " * padding + header_text + " " * (header_width - padding - len(header_text)) + "│", "bold"))
        print(self._format_message("└" + "─" * header_width + "┘", "bold"))
    
    def _draw_system_info(self):
        """Tampilkan informasi sistem."""
        # Bersihkan baris sebelumnya
        self._clear_line()
        
        # Dapatkan info sistem
        cpu_percent = psutil.cpu_percent()
        cpu_color = 'green' if cpu_percent < 70 else ('yellow' if cpu_percent < 90 else 'red')
        
        mem = psutil.virtual_memory()
        mem_percent = mem.percent
        mem_color = 'green' if mem_percent < 70 else ('yellow' if mem_percent < 90 else 'red')
        
        # Buat info sistem dalam satu baris
        system_info = f"CPU: {self._format_message(f'{cpu_percent}%', cpu_color)} | RAM: {self._format_message(f'{mem_percent}%', mem_color)}"
        
        # Tambahkan info GPU jika tersedia
        if self.show_gpu and self.has_gpu:
            try:
                # GPU utilization
                if torch.cuda.is_available():
                    gpu_util = torch.cuda.utilization(0)
                    gpu_mem_info = torch.cuda.mem_get_info(0)
                    gpu_mem_used = 1 - (gpu_mem_info[0] / gpu_mem_info[1])
                    gpu_mem_percent = round(gpu_mem_used * 100, 1)
                    
                    gpu_color = 'green' if gpu_util < 70 else ('yellow' if gpu_util < 90 else 'red')
                    gpu_mem_color = 'green' if gpu_mem_percent < 70 else ('yellow' if gpu_mem_percent < 90 else 'red')
                    
                    system_info += f" | GPU: {self._format_message(f'{gpu_util}%', gpu_color)} | VRAM: {self._format_message(f'{gpu_mem_percent}%', gpu_mem_color)}"
            except:
                system_info += f" | GPU: {self._format_message('Active', 'green')}"
        
        # Print sistem info dengan border
        print(self._format_message("┌" + "─" * (self.terminal_width - 2) + "┐", "bold"))
        print(self._format_message("│ " + system_info + " " * (self.terminal_width - len(system_info.replace('\033[92m', '').replace('\033[93m', '').replace('\033[91m', '').replace('\033[0m', '')) - 4) + " │", "bold"))
        print(self._format_message("└" + "─" * (self.terminal_width - 2) + "┘", "bold"))
    
    def _draw_configuration(self):
        """Tampilkan konfigurasi training."""
        if not self.status_data['konfigurasi']:
            return
            
        # Header konfigurasi
        print(self._format_message("┌" + "─" * (self.terminal_width - 2) + "┐", "bold"))
        header_text = " Konfigurasi Training "
        padding = (self.terminal_width - len(header_text) - 2) // 2
        print(self._format_message("│" + " " * padding + header_text + " " * (self.terminal_width - padding - len(header_text) - 2) + "│", "bold"))
        print(self._format_message("├" + "─" * (self.terminal_width - 2) + "┤", "bold"))
        
        # Bagi konfigurasi ke dalam dua kolom
        config_items = list(self.status_data['konfigurasi'].items())
        col_width = (self.terminal_width - 6) // 2
        
        # Print config dalam dua kolom
        for i in range(0, len(config_items), 2):
            left_item = config_items[i]
            left_text = f"{left_item[0]}: {left_item[1]}"
            left_text = left_text[:col_width]
            
            # Right column if available
            if i + 1 < len(config_items):
                right_item = config_items[i + 1]
                right_text = f"{right_item[0]}: {right_item[1]}"
                right_text = right_text[:col_width]
                print(self._format_message("│ " + left_text + " " * (col_width - len(left_text)) + " │ " + right_text + " " * (col_width - len(right_text)) + " │", "cyan"))
            else:
                # Last odd item
                print(self._format_message("│ " + left_text + " " * (col_width - len(left_text)) + " │" + " " * (col_width + 3) + "│", "cyan"))
        
        # Footer
        print(self._format_message("└" + "─" * (self.terminal_width - 2) + "┘", "bold"))
    
    def _draw_progress(self, key: str = "current"):
        """Tampilkan progress bar."""
        if key not in self.progress_bars or not self.progress_bars[key]:
            return
            
        prog = self.progress_bars[key]
        total = prog.get('total', 1)
        current = prog.get('current', 0)
        desc = prog.get('desc', 'Progress')
        
        # Hitung persentase progress
        progress_pct = min(1.0, current / total)
        
        # Buat progress bar
        bar_width = self.terminal_width - 20
        filled_width = int(bar_width * progress_pct)
        bar = f"[{'█' * filled_width}{' ' * (bar_width - filled_width)}]"
        
        # Bersihkan baris dan tampilkan progress
        self._clear_line()
        print(f"\r{desc}: {bar} {progress_pct * 100:.1f}% ({current}/{total})", end="", flush=True)
        print()  # Baris baru untuk mencegah tumpukan
    
    def _draw_metrics(self):
        """Tampilkan metrik training."""
        if not self.status_data['metrik']:
            return
            
        # Header metrik
        print(self._format_message("┌" + "─" * (self.terminal_width - 2) + "┐", "bold"))
        header_text = " Metrik Training "
        padding = (self.terminal_width - len(header_text) - 2) // 2
        print(self._format_message("│" + " " * padding + header_text + " " * (self.terminal_width - padding - len(header_text) - 2) + "│", "bold"))
        print(self._format_message("├" + "─" * (self.terminal_width - 2) + "┤", "bold"))
        
        # Metrik dalam satu baris
        metrics_text = " | ".join([f"{k}: {v}" for k, v in self.status_data['metrik'].items()])
        print(self._format_message("│ " + metrics_text + " " * (self.terminal_width - len(metrics_text) - 4) + " │", "green"))
        
        # Footer
        print(self._format_message("└" + "─" * (self.terminal_width - 2) + "┘", "bold"))
    
    def _draw_elapsed_time(self):
        """Tampilkan waktu yang telah berlalu."""
        if not self.start_time:
            return
            
        # Hitung waktu berlalu
        elapsed = time.time() - self.start_time
        hours, remainder = divmod(elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)
        elapsed_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
        
        # Tampilkan waktu dengan border
        print(self._format_message("┌" + "─" * (self.terminal_width - 2) + "┐", "bold"))
        elapsed_text = f"Waktu berlalu: {elapsed_str}"
        print(self._format_message("│ " + elapsed_text + " " * (self.terminal_width - len(elapsed_text) - 4) + " │", "yellow"))
        print(self._format_message("└" + "─" * (self.terminal_width - 2) + "┘", "bold"))
    
    def log(self, message: str, emoji: str = "ℹ️") -> None:
        """Log pesan dengan emoji."""
        formatted = f"{emoji} {message}"
        
        # Selalu log ke file logger
        self.logger.info(message)
        
        # Tambahkan ke buffer
        self.log_buffer.append(formatted)
        
        # Bersihkan baris dan print pesan
        self._clear_line()
        print(formatted)
        
    def info(self, message: str) -> None:
        """Log informasi."""
        self.log(message, "ℹ️")
        
    def success(self, message: str) -> None:
        """Log sukses."""
        self.log(message, "✅")
        
    def warning(self, message: str) -> None:
        """Log peringatan."""
        self.log(message, "⚠️")
        
    def error(self, message: str) -> None:
        """Log error."""
        self.log(message, "❌")
        
    def metric(self, message: str) -> None:
        """Log metrik."""
        self.log(message, "📊")
        
    def start_training(self, config: Dict[str, Any]) -> None:
        """
        Mulai sesi training.
        
        Args:
            config: Konfigurasi training
        """
        self.start_time = time.time()
        
        # Bersihkan layar untuk memulai sesi baru
        os.system('cls' if os.name == 'nt' else 'clear')
        
        # Tampilkan header
        self._draw_header()
        
        # Terapkan konfigurasi ke status panel
        self.status_data['konfigurasi'] = {
            'Backbone': config.get('backbone', 'default'),
            'Mode': config.get('detection_mode', 'single'),
            'Sumber': config.get('data_source', 'local'),
            'Batch Size': config.get('training', {}).get('batch_size', 32),
            'Learning Rate': config.get('training', {}).get('learning_rate', 0.001),
            'Epochs': config.get('training', {}).get('epochs', 100)
        }
        
        # Tampilkan informasi sistem
        self._draw_system_info()
        
        # Tampilkan konfigurasi
        self._draw_configuration()
        
        # Tampilkan device info
        device_info = f"Menggunakan {'GPU: ' + torch.cuda.get_device_name(0) if self.has_gpu else 'CPU'}"
        self.log(device_info, "🖥️")
        
    def create_progress_bar(
        self, 
        total: int, 
        desc: str,
        unit: str = "batch",
        key: str = "default"
    ) -> None:
        """
        Buat progress bar baru.
        
        Args:
            total: Total item
            desc: Deskripsi progress bar
            unit: Unit untuk progress bar
            key: Identifier unik untuk progress bar
        """
        if self.is_tui_mode:
            # Simpan info untuk progress bar di TUI mode
            self.progress_bars[key] = {
                'total': total,
                'current': 0,
                'desc': desc
            }
        else:
            # Gunakan internal tracking
            self.progress_bars['current'] = {
                'total': total,
                'current': 0,
                'desc': desc
            }
            
            # Tampilkan progress bar
            self._draw_progress()
            
    def update_progress(
        self, 
        n: int = 1, 
        key: str = "default",
        metrics: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Update progress bar.
        
        Args:
            n: Jumlah kemajuan
            key: Identifier progress bar
            metrics: Metrik untuk ditampilkan
        """
        if self.is_tui_mode:
            # Update tracking untuk progress bar TUI
            if key in self.progress_bars:
                progress = self.progress_bars[key]
                current = progress['current'] + n
                progress['current'] = min(current, progress['total'])
                
                # Tampilkan metrik jika ada dan ada display manager
                if metrics and self.display_manager:
                    metrics_str = ", ".join([
                        f"{k}: {v:.4f}" for k, v in metrics.items()
                    ])
                    self.display_manager.show_info(metrics_str)
        else:
            # Update progress bar internal
            if 'current' in self.progress_bars:
                prog = self.progress_bars['current']
                prog['current'] = min(prog['current'] + n, prog['total'])
                
                # Update metrik jika ada
                if metrics:
                    self.status_data['metrik'] = {
                        k: f"{v:.4f}" for k, v in metrics.items()
                    }
                
                # Tampilkan progress bar yang diupdate
                self._draw_progress()
    
    def close_progress_bar(self, key: str = "default") -> None:
        """
        Tutup progress bar dan menambahkan baris baru.
        
        Args:
            key: Identifier progress bar
        """
        if key in self.progress_bars:
            if self.is_tui_mode:
                del self.progress_bars[key]
            else:
                # Set progres ke 100% untuk menampilkan bar penuh
                if 'current' in self.progress_bars:
                    self.progress_bars['current']['current'] = self.progress_bars['current']['total']
                    self._draw_progress()
                    print()  # Tambahkan baris kosong setelah progress bar selesai
        
    def log_epoch_start(self, epoch: int, total_epochs: int, phase: str = "training") -> None:
        """
        Catat awal epoch.
        
        Args:
            epoch: Nomor epoch saat ini
            total_epochs: Total epoch
            phase: Phase training ('training' atau 'validation')
        """
        key = f"{phase}_epoch_{epoch}"
        self.epoch_times[key] = time.time()
        
        # Buat header epoch
        epoch_header = f"Epoch {epoch}/{total_epochs} - {phase.capitalize()}"
        print("\n" + "=" * len(epoch_header))  # Garis pemisah
        self.log(epoch_header, "🔄")
        
        # Update status panel dengan info epoch
        self.status_data['konfigurasi']['Epoch'] = f"{epoch}/{total_epochs}"
        self.status_data['konfigurasi']['Phase'] = phase.capitalize()
        
    def log_epoch_end(
        self, 
        epoch: int, 
        metrics: Dict[str, float], 
        phase: str = "training"
    ) -> None:
        """
        Catat akhir epoch dan tampilkan metrik.
        
        Args:
            epoch: Nomor epoch saat ini
            metrics: Metrik hasil epoch
            phase: Phase training ('training' atau 'validation')
        """
        key = f"{phase}_epoch_{epoch}"
        
        # Hitung durasi epoch
        if key in self.epoch_times:
            duration = time.time() - self.epoch_times[key]
            duration_str = f"{duration:.2f}s"
        else:
            duration_str = "N/A"
            
        # Format metrics
        metrics_str = " | ".join([
            f"{name}: {value:.4f}" 
            for name, value in metrics.items()
        ])
        
        # Log hasil epoch
        result = f"{phase.capitalize()} Epoch {epoch} selesai ({duration_str})"
        self.log(result, "⏱️")
        self.metric(metrics_str)  # Log metrik pada baris baru
        print("-" * self.terminal_width)  # Garis pemisah
            
        # Simpan metrik untuk tampilan
        self.status_data['metrik'] = {
            k: f"{v:.4f}" for k, v in metrics.items()
        }
        
        # Tampilkan metrik
        self._draw_metrics()
        
        # Simpan metrik untuk history
        self.metrics_history.append({
            'epoch': epoch,
            'phase': phase,
            **metrics
        })
        
    def log_best_model(self, metrics: Dict[str, float], model_path: str) -> None:
        """
        Log informasi model terbaik.
        
        Args:
            metrics: Metrik model terbaik
            model_path: Jalur penyimpanan model
        """
        # Tampilkan header untuk model terbaik
        print("\n" + "=" * 20 + " MODEL TERBAIK " + "=" * 20)
        
        # Log informasi model terbaik
        self.log(f"Model terbaik disimpan: {model_path}", "🏆")
        
        # Buat daftar metrik
        for name, value in metrics.items():
            self.metric(f"{name}: {value:.4f}")
        
        print("-" * self.terminal_width)  # Garis pemisah
        
        # Update metrik terbaik untuk tampilan
        self.status_data['metrik'] = {
            f"Best {k}": f"{v:.4f}" for k, v in metrics.items()
        }
        self._draw_metrics()
        
    def log_training_complete(self, total_time: float, best_metrics: Dict[str, float]) -> None:
        """
        Log penyelesaian training.
        
        Args:
            total_time: Total waktu training dalam detik
            best_metrics: Metrik terbaik dari training
        """
        # Format waktu
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        time_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
        
        # Tampilkan header kompletion yang jelas
        print("\n" + "=" * 20 + " TRAINING SELESAI " + "=" * 20)
        
        # Info penyelesaian
        self.log(f"Training Selesai - Total waktu: {time_str}", "✨")
        
        # Metrik terbaik pada baris terpisah
        self.log("Metrik terbaik:", "🏆")
        for name, value in best_metrics.items():
            self.metric(f"{name}: {value:.4f}")
        
        # Garis pemisah akhir
        print("=" * self.terminal_width)
        
        # Tampilkan ringkasan waktu
        self._draw_elapsed_time()
        
    def setup_interactive_mode(self) -> None:
        """
        Setup untuk mode interaktif TUI dengan display_manager.
        """
        if not self.is_tui_mode:
            return
            
        # Initialize internal progress tracking
        self.progress_bars = {}