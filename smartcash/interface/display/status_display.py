# File: smartcash/interface/display/status_display.py
# Author: Alfrida Sabar
# Deskripsi: Komponen untuk menampilkan status konfigurasi dan sistem dengan layout yang lebih baik

import curses
import torch
import psutil
from typing import Dict, Optional, Any
from pathlib import Path

from smartcash.interface.display.base_display import BaseDisplay

class StatusDisplay(BaseDisplay):
    """
    Komponen untuk menampilkan status konfigurasi dan sistem.
    Menempati 25% bagian kanan dari area utama.
    """
    
    def __init__(self, stdscr: curses.window, parent=None):
        """
        Inisialisasi status display.
        
        Args:
            stdscr: Curses window
            parent: Parent display (opsional)
        """
        super().__init__(stdscr, parent)
        
        # Pengaturan layout
        self.padding_x = 2  # Jarak dari tepi kiri area
        self.padding_y = 1  # Jarak antar item status
        
        # Status system terakhir
        self.last_system_status = {
            'cpu': 0,
            'ram': 0,
            'gpu_available': False,
            'gpu_name': 'Tidak tersedia'
        }
        
        # Interval refresh status sistem (dalam putaran refresh)
        self.system_refresh_interval = 10
        self.refresh_counter = 0
    
    def draw(self, config: Dict[str, Any]) -> None:
        """
        Gambar status konfigurasi dan sistem.
        
        Args:
            config: Konfigurasi aplikasi saat ini
        """
        # Bersihkan area tampilan
        self.clear_area()
        
        # Ukuran status aktif (tidak termasuk border)
        content_width = self.display_width - 2
        
        # Gambar border
        self.draw_border("Status")
        
        # Refresh status sistem hanya setiap interval tertentu
        self.refresh_counter += 1
        if self.refresh_counter >= self.system_refresh_interval:
            self._update_system_status()
            self.refresh_counter = 0
        
        # Gambar status sistem di bagian atas
        self._draw_system_status(self.y + 1, self.x + self.padding_x, content_width)
        
        # Gambar garis pembatas
        separator_y = self.y + 5
        self.safe_addstr(
            separator_y, 
            self.x + 1, 
            "─" * content_width
        )
        
        # Gambar status konfigurasi di bawah separator
        self._draw_configuration_status(config, separator_y + 1, self.x + self.padding_x, content_width)
    
    def _update_system_status(self) -> None:
        """Update informasi status sistem."""
        try:
            self.last_system_status = {
                'cpu': psutil.cpu_percent(),
                'ram': psutil.virtual_memory().percent,
                'gpu_available': torch.cuda.is_available(),
                'gpu_name': torch.cuda.get_device_name() if torch.cuda.is_available() else "Tidak tersedia",
                'gpu_memory': self._get_gpu_memory_info() if torch.cuda.is_available() else None
            }
        except Exception:
            # Fallback jika terjadi error
            pass
    
    def _get_gpu_memory_info(self) -> Optional[Dict[str, float]]:
        """
        Dapatkan informasi memori GPU.
        
        Returns:
            Dict berisi informasi memori atau None jika gagal
        """
        try:
            if torch.cuda.is_available():
                # Dapatkan memori total dan terpakai (dalam MB)
                total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
                reserved_mem = torch.cuda.memory_reserved(0) / (1024 * 1024)
                allocated_mem = torch.cuda.memory_allocated(0) / (1024 * 1024)
                free_mem = total_mem - reserved_mem
                
                return {
                    'total': total_mem,
                    'reserved': reserved_mem,
                    'allocated': allocated_mem,
                    'free': free_mem,
                    'percent': (reserved_mem / total_mem) * 100 if total_mem > 0 else 0
                }
            return None
        except Exception:
            return None
    
    def _draw_system_status(self, start_y: int, start_x: int, width: int) -> None:
        """
        Gambar status sistem.
        
        Args:
            start_y: Posisi y awal
            start_x: Posisi x awal
            width: Lebar area
        """
        # Judul status sistem
        self.safe_addstr(
            start_y, 
            start_x, 
            "🖥️ Status Sistem:", 
            self.COLOR_INFO
        )
        
        # CPU usage dengan indikator
        cpu_percent = self.last_system_status['cpu']
        cpu_color = self._get_resource_color(cpu_percent)
        cpu_bar = self._create_progress_bar(cpu_percent, 10)
        
        self.safe_addstr(
            start_y + 1, 
            start_x, 
            f"CPU: {cpu_percent}% {cpu_bar}", 
            cpu_color
        )
        
        # RAM usage dengan indikator
        ram_percent = self.last_system_status['ram']
        ram_color = self._get_resource_color(ram_percent)
        ram_bar = self._create_progress_bar(ram_percent, 10)
        
        self.safe_addstr(
            start_y + 2, 
            start_x, 
            f"RAM: {ram_percent}% {ram_bar}", 
            ram_color
        )
        
        # GPU status
        gpu_available = self.last_system_status['gpu_available']
        gpu_name = self.last_system_status['gpu_name']
        
        # Warna berdasarkan ketersediaan GPU
        gpu_color = self.COLOR_SUCCESS if gpu_available else self.COLOR_ERROR
        
        self.safe_addstr(
            start_y + 3, 
            start_x, 
            f"GPU: {gpu_name}", 
            gpu_color
        )
        
        # GPU memory jika tersedia
        if gpu_available and 'gpu_memory' in self.last_system_status and self.last_system_status['gpu_memory']:
            mem_info = self.last_system_status['gpu_memory']
            mem_percent = mem_info['percent']
            mem_color = self._get_resource_color(mem_percent)
            mem_bar = self._create_progress_bar(mem_percent, 10)
            
            self.safe_addstr(
                start_y + 4, 
                start_x, 
                f"VRAM: {mem_percent:.1f}% {mem_bar} ({mem_info['allocated']:.0f}/{mem_info['total']:.0f} MB)", 
                mem_color
            )
    
    def _draw_configuration_status(
        self, 
        config: Dict[str, Any], 
        start_y: int, 
        start_x: int,
        width: int
    ) -> None:
        """
        Gambar status konfigurasi.
        
        Args:
            config: Konfigurasi aplikasi saat ini
            start_y: Posisi y awal
            start_x: Posisi x awal
            width: Lebar area
        """
        # Judul status konfigurasi
        self.safe_addstr(
            start_y, 
            start_x, 
            "⚙️ Konfigurasi:", 
            self.COLOR_INFO
        )
        current_y = start_y + 1
        
        # Data source
        self._draw_config_item(
            "Sumber Data", 
            config.get('data_source'), 
            current_y, 
            start_x
        )
        current_y += 1
        
        # Detection mode
        self._draw_config_item(
            "Mode Deteksi", 
            config.get('detection_mode'), 
            current_y, 
            start_x
        )
        current_y += 1
        
        # Backbone
        self._draw_config_item(
            "Arsitektur", 
            config.get('backbone'), 
            current_y, 
            start_x
        )
        current_y += 2
        
        # Training parameters
        training_config = config.get('training', {})
        if training_config:
            self.safe_addstr(
                current_y, 
                start_x, 
                "📊 Parameter Training:", 
                self.COLOR_INFO
            )
            current_y += 1
            
            # Tampilkan parameter training
            parameters = [
                ('batch_size', "Batch Size", '32'),
                ('learning_rate', "Learning Rate", '0.001'),
                ('epochs', "Epochs", '100'),
                ('early_stopping_patience', "Early Stop", '10')
            ]
            
            for key, label, default in parameters:
                value = training_config.get(key, default)
                
                # Format khusus untuk beberapa parameter
                if key == 'learning_rate' and isinstance(value, float):
                    value = f"{value:.6f}"
                
                # Tampilkan parameter dengan indentasi
                self._draw_config_item(
                    label, 
                    value, 
                    current_y, 
                    start_x + 2,  # Indentasi
                    label_width=12  # Label yang lebih pendek
                )
                current_y += 1
    
    def _draw_config_item(
        self, 
        label: str, 
        value: Any, 
        y: int, 
        x: int,
        label_width: int = 15
    ) -> None:
        """
        Gambar item konfigurasi dengan label dan nilai.
        
        Args:
            label: Label item
            value: Nilai item
            y: Posisi y
            x: Posisi x
            label_width: Lebar label
        """
        # Tampilkan label
        self.safe_addstr(y, x, f"{label}:")
        
        # Konversi nilai ke string
        value_str = str(value) if value is not None else 'Belum dipilih'
        
        # Pilih warna berdasarkan keberadaan nilai
        if value is None or value_str == 'Belum dipilih':
            color = self.COLOR_ERROR
        else:
            color = self.COLOR_SUCCESS
            
        # Tampilkan nilai dengan warna yang sesuai
        value_x = x + label_width
        self.safe_addstr(y, value_x, value_str, color)
    
    def _get_resource_color(self, percent: float) -> int:
        """
        Dapatkan warna berdasarkan persentase penggunaan resource.
        
        Args:
            percent: Persentase penggunaan
            
        Returns:
            Kode warna
        """
        if percent >= 90:
            return self.COLOR_ERROR  # Merah untuk penggunaan tinggi
        elif percent >= 70:
            return self.COLOR_WARNING  # Kuning untuk penggunaan sedang
        else:
            return self.COLOR_SUCCESS  # Hijau untuk penggunaan rendah
    
    def _create_progress_bar(self, percent: float, length: int = 10) -> str:
        """
        Buat progress bar dengan karakter.
        
        Args:
            percent: Persentase (0-100)
            length: Panjang bar
            
        Returns:
            String progress bar
        """
        # Pastikan percent dalam range valid
        percent = max(0, min(100, percent))
        
        # Hitung jumlah karakter yang diisi
        filled = int(length * percent / 100)
        
        # Buat progress bar
        return '█' * filled + '░' * (length - filled)
    
    def handle_input(self, key: int) -> Optional[bool]:
        """
        Tangani input keyboard (tidak mendukung interaksi khusus).
        
        Args:
            key: Kode keyboard
            
        Returns:
            None karena tidak menangani input
        """
        # Status display biasanya tidak menangani input langsung
        return None