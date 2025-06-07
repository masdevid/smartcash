#!/usr/bin/env python
# -*- coding: utf-8 -*-
# /Users/masdevid/Projects/smartcash/smartcash/ui/dataset/downloader/handlers/base_handler.py
# Kelas dasar untuk handler dataset yang menyediakan fungsionalitas umum

import os
import time
from pathlib import Path
from typing import Dict, Any, Callable, Optional, List, Tuple, Union

from smartcash.common.logger import get_logger
from smartcash.ui.dataset.downloader.utils.progress_mapping import ProgressMapping

class BaseDatasetDownloadHandler:
    """
    Kelas dasar untuk semua handler dataset yang menyediakan fungsionalitas umum
    seperti progress tracking, error handling, dan interaksi dengan UI components
    """
    
    def __init__(self, 
                 ui_components: Dict[str, Any] = None,
                 logger=None):
        """
        Inisialisasi base handler
        
        Args:
            ui_components: Dictionary komponen UI yang digunakan handler
            logger: Logger untuk mencatat log
        """
        self.ui_components = ui_components or {}
        self.logger = logger or get_logger(self.__class__.__name__)
        self.progress_tracker = self.ui_components.get('progress_tracker')
        self._is_processing = False
        
    def _create_progress_callback(self) -> Callable[[str, int, int, str], None]:
        """
        Membuat callback progress dengan signature standar menggunakan ProgressMapping
        
        Returns:
            Fungsi callback dengan signature (step, current, total, message)
        """
        # Import ProgressMapping di sini untuk menghindari circular import
        from smartcash.ui.dataset.downloader.utils.progress_mapping import ProgressMapping
        
        def progress_callback(step: str, current: int, total: int, message: str):
            """Standardized progress callback: (step, current, total, message)"""
            try:
                if not self.progress_tracker:
                    return
                
                # Hitung persentase menggunakan ProgressMapping
                percentage = ProgressMapping.calculate_percentage(step, current, total)
                
                # Format pesan dengan emoji yang sesuai
                formatted_message = ProgressMapping.format_message(step, message)
                
                # Update progress tracker
                self.progress_tracker.update_overall(percentage, formatted_message)
                
                # Log pesan ke log accordion jika tersedia
                if 'log_accordion' in self.ui_components and hasattr(self.ui_components['log_accordion'], 'add_log'):
                    # Tentukan level log berdasarkan step
                    log_level = 'info'
                    
                    if ProgressMapping.is_error_step(step):
                        log_level = 'error'
                        # Expand log accordion untuk error
                        if hasattr(self.ui_components['log_accordion'], 'selected_index'):
                            self.ui_components['log_accordion'].selected_index = 0
                    elif ProgressMapping.is_warning_step(step):
                        log_level = 'warning'
                        # Expand log accordion untuk warning
                        if hasattr(self.ui_components['log_accordion'], 'selected_index'):
                            self.ui_components['log_accordion'].selected_index = 0
                    elif 'success' in step.lower() or 'complete' in step.lower() or 'done' in step.lower():
                        log_level = 'success'
                    
                    # Tambahkan log
                    self.ui_components['log_accordion'].add_log(formatted_message, log_level)
                
            except Exception as e:
                # Silent exception untuk mencegah kegagalan download
                if self.logger:
                    self.logger.warning(f"⚠️ Error dalam progress callback: {str(e)}")
        
        return progress_callback
    
    def _handle_error(self, error: Union[str, Exception], button_to_restore: Optional[Any] = None) -> None:
        """
        Menangani error dengan update UI, logging, dan restore button state
        
        Args:
            error: Error message atau exception
            button_to_restore: Button yang perlu di-restore state-nya
        """
        error_msg = str(error)
        self.logger.error(f"❌ {error_msg}")
        
        # Update progress tracker jika tersedia
        if self.progress_tracker:
            self.progress_tracker.update_overall(0, f"❌ Error: {error_msg}")
        
        # Tambahkan error ke log accordion jika tersedia
        if 'log_accordion' in self.ui_components and hasattr(self.ui_components['log_accordion'], 'add_log'):
            self.ui_components['log_accordion'].add_log(f"❌ Error: {error_msg}", 'error')
            
            # Expand log accordion untuk menampilkan error
            if hasattr(self.ui_components['log_accordion'], 'selected_index'):
                self.ui_components['log_accordion'].selected_index = 0
        
        # Restore button state jika tersedia
        if button_to_restore and hasattr(button_to_restore, 'disabled'):
            button_to_restore.disabled = False
            if hasattr(button_to_restore, 'description'):
                button_to_restore.description = getattr(button_to_restore, '_original_description', 'Submit')
        
        # Reset processing flag
        self._is_processing = False
    
    def _prepare_button_state(self, button: Any, processing_text: str = "Processing...") -> None:
        """
        Menyiapkan button state untuk processing
        
        Args:
            button: Button yang akan diupdate
            processing_text: Teks yang ditampilkan saat processing
        """
        if button and hasattr(button, 'disabled'):
            # Simpan deskripsi asli jika belum disimpan
            if hasattr(button, 'description') and not hasattr(button, '_original_description'):
                setattr(button, '_original_description', button.description)
            
            # Update button state
            button.disabled = True
            if hasattr(button, 'description'):
                button.description = processing_text
    
    def _restore_button_state(self, button: Any = None) -> None:
        """
        Mengembalikan button state ke normal
        
        Args:
            button: Button yang akan diupdate, jika None akan menggunakan button dari _prepare_button_state
        """
        if button and hasattr(button, 'disabled'):
            button.disabled = False
            if hasattr(button, 'description') and hasattr(button, '_original_description'):
                button.description = getattr(button, '_original_description')
    
    def _format_file_size(self, size_bytes: int) -> str:
        """
        Format ukuran file dari bytes ke format yang lebih mudah dibaca
        
        Args:
            size_bytes: Ukuran file dalam bytes
            
        Returns:
            String ukuran file yang diformat
        """
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.1f} MB"
        else:
            return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"
    
    def _check_directory_exists(self, directory: Union[str, Path]) -> bool:
        """
        Memeriksa apakah direktori ada
        
        Args:
            directory: Path direktori yang akan diperiksa
            
        Returns:
            True jika direktori ada, False jika tidak
        """
        try:
            path = Path(directory)
            return path.exists() and path.is_dir()
        except Exception as e:
            self.logger.warning(f"⚠️ Error saat memeriksa direktori {directory}: {str(e)}")
            return False
    
    def _count_files_in_directory(self, directory: Union[str, Path], pattern: str = "*") -> Tuple[int, int]:
        """
        Menghitung jumlah file dan ukuran total dalam direktori
        
        Args:
            directory: Path direktori yang akan dihitung
            pattern: Pattern untuk filter file
            
        Returns:
            Tuple (jumlah_file, ukuran_total_bytes)
        """
        try:
            path = Path(directory)
            if not path.exists() or not path.is_dir():
                return 0, 0
            
            files = list(path.glob(pattern))
            file_count = len(files)
            total_size = sum(f.stat().st_size for f in files if f.is_file())
            
            return file_count, total_size
        except Exception as e:
            self.logger.warning(f"⚠️ Error saat menghitung file di {directory}: {str(e)}")
            return 0, 0
    
    def _get_dataset_path(self) -> Path:
        """
        Mendapatkan path dataset dari environment manager
        
        Returns:
            Path direktori dataset
        """
        try:
            from smartcash.common.environment import get_environment_manager
            env_manager = get_environment_manager()
            return Path(env_manager.get_dataset_path())
        except Exception as e:
            self.logger.warning(f"⚠️ Error saat mendapatkan dataset path: {str(e)}")
            # Fallback ke default path
            return Path(os.path.expanduser("~/datasets"))
    
    def _log_with_emoji(self, message: str, level: str = 'info', expand_accordion: bool = False) -> None:
        """
        Log pesan dengan emoji dan tambahkan ke log accordion jika tersedia
        
        Args:
            message: Pesan yang akan di-log
            level: Level log (info, warning, error, success)
            expand_accordion: Apakah perlu expand log accordion
        """
        # Tentukan emoji berdasarkan level
        emoji = "🔍"  # Default info
        if level == 'warning':
            emoji = "⚠️"
            self.logger.warning(f"{emoji} {message}")
        elif level == 'error':
            emoji = "❌"
            self.logger.error(f"{emoji} {message}")
        elif level == 'success':
            emoji = "✅"
            self.logger.info(f"{emoji} {message}")
        else:
            self.logger.info(f"{emoji} {message}")
        
        # Tambahkan ke log accordion jika tersedia
        if 'log_accordion' in self.ui_components and hasattr(self.ui_components['log_accordion'], 'add_log'):
            # Format pesan dengan emoji jika belum ada
            if not any(message.startswith(e) for e in ["🔍", "⚠️", "❌", "✅"]):
                formatted_message = f"{emoji} {message}"
            else:
                formatted_message = message
                
            self.ui_components['log_accordion'].add_log(formatted_message, level)
            
            # Expand log accordion jika diminta
            if expand_accordion and hasattr(self.ui_components['log_accordion'], 'selected_index'):
                self.ui_components['log_accordion'].selected_index = 0
