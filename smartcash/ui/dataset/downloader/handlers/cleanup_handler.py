#!/usr/bin/env python
# -*- coding: utf-8 -*-
# /Users/masdevid/Projects/smartcash/smartcash/ui/dataset/downloader/handlers/cleanup_handler.py
# Handler untuk pembersihan dataset yang mewarisi dari BaseDatasetDownloadHandler

import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Union

from smartcash.ui.dataset.downloader.handlers.base_handler import BaseDatasetDownloadHandler
from smartcash.ui.components.dialogs import confirm

class DatasetCleanupHandler(BaseDatasetDownloadHandler):
    """
    Handler untuk pembersihan dataset yang mewarisi dari BaseDatasetDownloadHandler
    Menyediakan fungsionalitas untuk membersihkan dataset yang sudah ada
    """
    
    def __init__(self, ui_components: Dict[str, Any], config: Dict[str, Any], logger=None):
        # Panggil konstruktor parent class
        super().__init__(ui_components, logger)
        
        # Inisialisasi properti spesifik DatasetCleanupHandler
        self.config = config
    
    async def cleanup_dataset(self, button=None) -> Dict[str, Any]:
        """
        Membersihkan dataset dengan konfirmasi, progress tracking, dan error handling
        
        Args:
            button: Button yang memicu pembersihan, untuk state management
            
        Returns:
            Dictionary hasil pembersihan
        """
        # Persiapkan UI untuk proses pembersihan
        if button:
            self._prepare_button_state(button, "Mempersiapkan...")
        
        if self.progress_tracker:
            self.progress_tracker.reset()
            self.progress_tracker.update_overall(0, "🔍 Memeriksa dataset untuk dibersihkan...")
        
        # Reset log accordion jika tersedia
        if 'log_accordion' in self.ui_components and hasattr(self.ui_components['log_accordion'], 'clear_logs'):
            self.ui_components['log_accordion'].clear_logs()
        
        try:
            # Dapatkan path dataset
            dataset_path = self._get_dataset_path()
            self._log_with_emoji(f"Path dataset: {dataset_path}", "info")
            
            # Periksa apakah dataset ada
            if not dataset_path.exists():
                self._log_with_emoji(f"Direktori dataset tidak ditemukan: {dataset_path}", "warning", True)
                if button:
                    self._restore_button_state(button)
                return {'status': 'warning', 'message': f"Direktori dataset tidak ditemukan: {dataset_path}"}
            
            # Periksa direktori yang akan dibersihkan
            dirs_to_check = {
                'downloads': dataset_path / "downloads",
                'splits': dataset_path / "splits"
            }
            
            # Hitung ukuran dan jumlah file
            cleanup_stats = {}
            total_files = 0
            total_size = 0
            
            for dir_name, dir_path in dirs_to_check.items():
                if dir_path.exists():
                    file_count, size = self._count_files_in_directory(dir_path, "**/*")
                    cleanup_stats[dir_name] = {
                        'path': str(dir_path),
                        'file_count': file_count,
                        'size': size,
                        'formatted_size': self._format_file_size(size)
                    }
                    total_files += file_count
                    total_size += size
            
            # Jika tidak ada file untuk dibersihkan
            if total_files == 0:
                self._log_with_emoji("Tidak ada file untuk dibersihkan", "info")
                if self.progress_tracker:
                    self.progress_tracker.update_overall(100, "✅ Tidak ada file untuk dibersihkan")
                if button:
                    self._restore_button_state(button)
                return {'status': 'success', 'message': "Tidak ada file untuk dibersihkan"}
            
            # Tampilkan ringkasan file yang akan dibersihkan
            summary = [
                "🗑️ File yang akan dibersihkan:",
                f"📊 Total: {total_files:,} file ({self._format_file_size(total_size)})"
            ]
            
            for dir_name, stats in cleanup_stats.items():
                if stats['file_count'] > 0:
                    summary.append(f"📂 {dir_name}: {stats['file_count']:,} file ({stats['formatted_size']})")
            
            summary_message = "\n".join(summary)
            self._log_with_emoji(summary_message, "info")
            
            # Update progress
            if self.progress_tracker:
                self.progress_tracker.update_overall(20, "🔍 Menunggu konfirmasi untuk pembersihan...")
            
            # Konfirmasi pembersihan
            confirm_message = f"Anda akan menghapus {total_files:,} file ({self._format_file_size(total_size)}) dari dataset. Lanjutkan?"
            confirmed = await confirm(confirm_message, "Konfirmasi Pembersihan")
            
            if not confirmed:
                self._log_with_emoji("Pembersihan dibatalkan oleh pengguna", "info")
                if self.progress_tracker:
                    self.progress_tracker.update_overall(0, "❌ Pembersihan dibatalkan")
                if button:
                    self._restore_button_state(button)
                return {'status': 'cancelled', 'message': "Pembersihan dibatalkan oleh pengguna"}
            
            # Lakukan pembersihan
            if button:
                self._prepare_button_state(button, "Membersihkan...")
            
            if self.progress_tracker:
                self.progress_tracker.update_overall(30, "🗑️ Mulai membersihkan dataset...")
            
            # Hapus direktori satu per satu
            result = await self._perform_cleanup(dirs_to_check)
            
            # Update progress
            if self.progress_tracker:
                if result.get('status') == 'success':
                    self.progress_tracker.update_overall(100, "✅ Pembersihan dataset selesai")
                else:
                    self.progress_tracker.update_overall(0, f"❌ Pembersihan gagal: {result.get('message', 'Unknown error')}")
            
            # Restore button state
            if button:
                self._restore_button_state(button)
                
            return result
            
        except Exception as e:
            # Handle error
            self._handle_error(f"Error saat membersihkan dataset: {str(e)}", button)
            return {'status': 'error', 'message': str(e)}
    
    async def _perform_cleanup(self, dirs_to_clean: Dict[str, Path]) -> Dict[str, Any]:
        """
        Melakukan pembersihan direktori
        
        Args:
            dirs_to_clean: Dictionary direktori yang akan dibersihkan
            
        Returns:
            Dictionary hasil pembersihan
        """
        try:
            cleaned_dirs = []
            errors = []
            
            # Hitung total direktori untuk progress tracking
            total_dirs = len(dirs_to_clean)
            current_dir = 0
            
            for dir_name, dir_path in dirs_to_clean.items():
                current_dir += 1
                progress = 30 + int((current_dir / total_dirs) * 60)
                
                if self.progress_tracker:
                    self.progress_tracker.update_overall(progress, f"🗑️ Membersihkan {dir_name}...")
                
                try:
                    if dir_path.exists():
                        self._log_with_emoji(f"Membersihkan direktori {dir_name}: {dir_path}", "info")
                        
                        # Hapus direktori
                        shutil.rmtree(dir_path)
                        
                        # Buat kembali direktori kosong
                        dir_path.mkdir(parents=True, exist_ok=True)
                        
                        cleaned_dirs.append(dir_name)
                        self._log_with_emoji(f"Direktori {dir_name} berhasil dibersihkan", "success")
                    else:
                        self._log_with_emoji(f"Direktori {dir_name} tidak ditemukan, membuat direktori baru", "info")
                        dir_path.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    error_msg = f"Error saat membersihkan {dir_name}: {str(e)}"
                    self._log_with_emoji(error_msg, "error")
                    errors.append(error_msg)
            
            # Buat hasil
            if errors:
                return {
                    'status': 'partial',
                    'message': f"Pembersihan sebagian berhasil dengan {len(errors)} error",
                    'cleaned_dirs': cleaned_dirs,
                    'errors': errors
                }
            else:
                return {
                    'status': 'success',
                    'message': f"Pembersihan {len(cleaned_dirs)} direktori berhasil",
                    'cleaned_dirs': cleaned_dirs
                }
                
        except Exception as e:
            self._log_with_emoji(f"Error saat melakukan pembersihan: {str(e)}", "error", True)
            return {'status': 'error', 'message': str(e)}
    
    def _create_empty_dataset_structure(self) -> Dict[str, Any]:
        """
        Membuat struktur dataset kosong
        
        Returns:
            Dictionary hasil pembuatan struktur
        """
        try:
            # Dapatkan path dataset
            dataset_path = self._get_dataset_path()
            self._log_with_emoji(f"Membuat struktur dataset di: {dataset_path}", "info")
            
            # Buat direktori utama jika belum ada
            dataset_path.mkdir(parents=True, exist_ok=True)
            
            # Buat direktori downloads
            downloads_dir = dataset_path / "downloads"
            downloads_dir.mkdir(exist_ok=True)
            
            # Buat direktori splits dan subdirektori
            splits_dir = dataset_path / "splits"
            splits_dir.mkdir(exist_ok=True)
            
            for split in ['train', 'val', 'test']:
                split_dir = splits_dir / split
                split_dir.mkdir(exist_ok=True)
                
                # Buat direktori images dan labels
                (split_dir / "images").mkdir(exist_ok=True)
                (split_dir / "labels").mkdir(exist_ok=True)
            
            self._log_with_emoji("Struktur dataset berhasil dibuat", "success")
            
            return {
                'status': 'success',
                'message': "Struktur dataset berhasil dibuat",
                'dataset_path': str(dataset_path)
            }
            
        except Exception as e:
            self._log_with_emoji(f"Error saat membuat struktur dataset: {str(e)}", "error", True)
            return {'status': 'error', 'message': str(e)}
