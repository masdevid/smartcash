#!/usr/bin/env python
# -*- coding: utf-8 -*-
# /Users/masdevid/Projects/smartcash/smartcash/ui/dataset/downloader/handlers/check_handler.py
# Handler untuk pengecekan dataset yang mewarisi dari BaseDatasetDownloadHandler

from pathlib import Path
from typing import Dict, Any, Optional, Union

from smartcash.ui.dataset.downloader.handlers.base_handler import BaseDatasetDownloadHandler

class DatasetCheckHandler(BaseDatasetDownloadHandler):
    """
    Handler untuk pengecekan dataset yang mewarisi dari BaseDatasetDownloadHandler
    Menyediakan fungsionalitas untuk memeriksa dataset yang sudah ada
    """
    
    def __init__(self, ui_components: Dict[str, Any], config: Dict[str, Any], logger=None):
        # Panggil konstruktor parent class
        super().__init__(ui_components, logger)
        
        # Inisialisasi properti spesifik DatasetCheckHandler
        self.config = config
        
    def check_dataset(self, button=None) -> Dict[str, Any]:
        """
        Memeriksa dataset yang sudah ada dengan progress tracking dan error handling
        
        Args:
            button: Button yang memicu pengecekan, untuk state management
            
        Returns:
            Dictionary hasil pengecekan
        """
        # Persiapkan UI untuk proses pengecekan
        if button:
            self._prepare_button_state(button, "Memeriksa...")
        
        if self.progress_tracker:
            self.progress_tracker.reset()
            self.progress_tracker.update_overall(0, "🔍 Memulai pengecekan dataset...")
        
        # Reset log accordion jika tersedia
        if 'log_accordion' in self.ui_components and hasattr(self.ui_components['log_accordion'], 'clear_logs'):
            self.ui_components['log_accordion'].clear_logs()
        
        try:
            # Mulai pengecekan
            self._log_with_emoji("Memulai pengecekan dataset...", "info")
            
            # Dapatkan path dataset
            dataset_path = self._get_dataset_path()
            self._log_with_emoji(f"Path dataset: {dataset_path}", "info")
            
            # Periksa dataset secara detail
            check_result = self._check_existing_dataset_detailed()
            
            # Tampilkan ringkasan dataset
            if check_result.get('status') == 'success':
                self._display_dataset_summary(check_result)
            
            # Update progress tracker
            if self.progress_tracker:
                if check_result.get('status') == 'success':
                    self.progress_tracker.update_overall(100, "✅ Pengecekan dataset selesai")
                else:
                    self.progress_tracker.update_overall(0, f"❌ Pengecekan gagal: {check_result.get('message', 'Unknown error')}")
            
            # Restore button state
            if button:
                self._restore_button_state(button)
                
            return check_result
            
        except Exception as e:
            # Handle error
            self._handle_error(f"Error saat memeriksa dataset: {str(e)}", button)
            return {'status': 'error', 'message': str(e)}
    
    def _check_existing_dataset_detailed(self) -> Dict[str, Any]:
        """
        Memeriksa dataset yang sudah ada dengan detail
        
        Returns:
            Dictionary hasil pengecekan dengan detail
        """
        try:
            # Update progress
            if self.progress_tracker:
                self.progress_tracker.update_overall(10, "🔍 Memeriksa struktur dataset...")
            
            # Dapatkan path dataset
            dataset_path = self._get_dataset_path()
            self._log_with_emoji(f"Memeriksa dataset di: {dataset_path}", "info")
            
            # Periksa apakah direktori dataset ada
            if not dataset_path.exists():
                self._log_with_emoji(f"Direktori dataset tidak ditemukan: {dataset_path}", "warning", True)
                return {'status': 'warning', 'message': f"Direktori dataset tidak ditemukan: {dataset_path}"}
            
            # Update progress
            if self.progress_tracker:
                self.progress_tracker.update_overall(20, "🔍 Memeriksa direktori downloads...")
            
            # Periksa direktori downloads
            downloads_dir = dataset_path / "downloads"
            downloads_result = self._check_downloads_dir(downloads_dir)
            
            # Update progress
            if self.progress_tracker:
                self.progress_tracker.update_overall(40, "🔍 Memeriksa direktori splits...")
            
            # Periksa direktori splits
            splits_dir = dataset_path / "splits"
            splits_result = self._check_splits_dir(splits_dir)
            
            # Update progress
            if self.progress_tracker:
                self.progress_tracker.update_overall(70, "🔍 Mengumpulkan statistik dataset...")
            
            # Gabungkan hasil
            result = {
                'status': 'success',
                'message': 'Dataset ditemukan',
                'dataset_path': str(dataset_path),
                'downloads': downloads_result,
                'splits': splits_result,
                'stats': {
                    'total_images': sum(split.get('images', 0) for split in splits_result.get('splits', {}).values()),
                    'total_labels': sum(split.get('labels', 0) for split in splits_result.get('splits', {}).values()),
                    'total_downloads': downloads_result.get('file_count', 0),
                    'total_download_size': downloads_result.get('total_size', 0),
                    'splits': splits_result.get('splits', {})
                }
            }
            
            # Update progress
            if self.progress_tracker:
                self.progress_tracker.update_overall(90, "✅ Pengecekan dataset selesai")
            
            return result
            
        except Exception as e:
            self._log_with_emoji(f"Error saat memeriksa dataset: {str(e)}", "error", True)
            return {'status': 'error', 'message': str(e)}
    
    def _check_downloads_dir(self, downloads_dir: Path) -> Dict[str, Any]:
        """
        Memeriksa direktori downloads
        
        Args:
            downloads_dir: Path direktori downloads
            
        Returns:
            Dictionary hasil pengecekan
        """
        try:
            self._log_with_emoji(f"Memeriksa direktori downloads: {downloads_dir}", "info")
            
            # Periksa apakah direktori ada
            if not downloads_dir.exists():
                self._log_with_emoji(f"Direktori downloads tidak ditemukan: {downloads_dir}", "warning")
                return {'status': 'warning', 'message': 'Direktori downloads tidak ditemukan', 'file_count': 0, 'total_size': 0}
            
            # Hitung jumlah file dan ukuran total
            file_count, total_size = self._count_files_in_directory(downloads_dir)
            
            # Format ukuran file
            formatted_size = self._format_file_size(total_size)
            
            # Log hasil
            if file_count > 0:
                self._log_with_emoji(f"Direktori downloads berisi {file_count} file dengan total ukuran {formatted_size}", "info")
            else:
                self._log_with_emoji(f"Direktori downloads kosong", "warning")
            
            # Periksa file-file di direktori downloads
            files = list(downloads_dir.glob("*"))
            for i, file_path in enumerate(files[:10]):  # Batasi hanya 10 file untuk efisiensi
                try:
                    file_size = file_path.stat().st_size
                    formatted_file_size = self._format_file_size(file_size)
                    self._log_with_emoji(f"File: {file_path.name} ({formatted_file_size})", "info")
                except Exception as e:
                    self._log_with_emoji(f"Error saat memeriksa file {file_path.name}: {str(e)}", "warning")
            
            # Jika ada lebih dari 10 file, tampilkan pesan
            if len(files) > 10:
                self._log_with_emoji(f"... dan {len(files) - 10} file lainnya", "info")
            
            return {
                'status': 'success' if file_count > 0 else 'warning',
                'message': f"Direktori downloads berisi {file_count} file" if file_count > 0 else "Direktori downloads kosong",
                'file_count': file_count,
                'total_size': total_size,
                'formatted_size': formatted_size
            }
            
        except Exception as e:
            self._log_with_emoji(f"Error saat memeriksa direktori downloads: {str(e)}", "error")
            return {'status': 'error', 'message': str(e), 'file_count': 0, 'total_size': 0}
    
    def _check_splits_dir(self, splits_dir: Path) -> Dict[str, Any]:
        """
        Memeriksa direktori splits
        
        Args:
            splits_dir: Path direktori splits
            
        Returns:
            Dictionary hasil pengecekan
        """
        try:
            self._log_with_emoji(f"Memeriksa direktori splits: {splits_dir}", "info")
            
            # Periksa apakah direktori ada
            if not splits_dir.exists():
                self._log_with_emoji(f"Direktori splits tidak ditemukan: {splits_dir}", "warning")
                return {'status': 'warning', 'message': 'Direktori splits tidak ditemukan', 'splits': {}}
            
            # Periksa subdirektori (train, val, test)
            splits = {}
            expected_splits = ['train', 'val', 'test']
            
            for split_name in expected_splits:
                split_dir = splits_dir / split_name
                split_result = self._check_split_directory(split_dir, split_name)
                splits[split_name] = split_result
            
            # Hitung total
            total_images = sum(split.get('images', 0) for split in splits.values())
            total_labels = sum(split.get('labels', 0) for split in splits.values())
            
            # Log hasil
            if total_images > 0:
                self._log_with_emoji(f"Total dataset: {total_images} gambar, {total_labels} label", "info")
            else:
                self._log_with_emoji(f"Dataset kosong", "warning")
            
            return {
                'status': 'success' if total_images > 0 else 'warning',
                'message': f"Dataset berisi {total_images} gambar" if total_images > 0 else "Dataset kosong",
                'splits': splits,
                'total_images': total_images,
                'total_labels': total_labels
            }
            
        except Exception as e:
            self._log_with_emoji(f"Error saat memeriksa direktori splits: {str(e)}", "error")
            return {'status': 'error', 'message': str(e), 'splits': {}}
    
    def _check_split_directory(self, split_dir: Path, split_name: str) -> Dict[str, Any]:
        """
        Memeriksa direktori split tunggal
        
        Args:
            split_dir: Path direktori split
            split_name: Nama split (train, val, test)
            
        Returns:
            Dictionary hasil pengecekan
        """
        try:
            # Periksa apakah direktori ada
            if not split_dir.exists():
                return {'status': 'warning', 'message': f"Direktori {split_name} tidak ditemukan", 'images': 0, 'labels': 0}
            
            # Periksa direktori images dan labels
            images_dir = split_dir / "images"
            labels_dir = split_dir / "labels"
            
            # Hitung jumlah file
            images_count = 0
            labels_count = 0
            
            if images_dir.exists():
                images_count, _ = self._count_files_in_directory(images_dir)
            
            if labels_dir.exists():
                labels_count, _ = self._count_files_in_directory(labels_dir)
            
            # Log hasil
            if images_count > 0 or labels_count > 0:
                self._log_with_emoji(f"Split {split_name}: {images_count} gambar, {labels_count} label", "info")
            
            return {
                'status': 'success' if images_count > 0 else 'warning',
                'message': f"Split {split_name} berisi {images_count} gambar" if images_count > 0 else f"Split {split_name} kosong",
                'images': images_count,
                'labels': labels_count,
                'images_dir_exists': images_dir.exists(),
                'labels_dir_exists': labels_dir.exists()
            }
            
        except Exception as e:
            self._log_with_emoji(f"Error saat memeriksa split {split_name}: {str(e)}", "warning")
            return {'status': 'error', 'message': str(e), 'images': 0, 'labels': 0}
    
    def _display_dataset_summary(self, check_result: Dict[str, Any]) -> None:
        """
        Menampilkan ringkasan dataset
        
        Args:
            check_result: Hasil pengecekan dataset
        """
        try:
            # Dapatkan statistik
            stats = check_result.get('stats', {})
            total_images = stats.get('total_images', 0)
            total_labels = stats.get('total_labels', 0)
            total_downloads = stats.get('total_downloads', 0)
            total_download_size = stats.get('total_download_size', 0)
            
            # Format ukuran download
            formatted_download_size = self._format_file_size(total_download_size)
            
            # Buat pesan ringkasan
            summary = [
                "📊 Ringkasan Dataset:",
                f"📂 Path: {check_result.get('dataset_path')}",
                f"🖼️ Total Gambar: {total_images:,}",
                f"🏷️ Total Label: {total_labels:,}",
                f"📦 Total Download: {total_downloads:,} file ({formatted_download_size})"
            ]
            
            # Tambahkan detail per split
            splits = stats.get('splits', {})
            if splits:
                summary.append("\n📊 Detail per Split:")
                for split_name, split_data in splits.items():
                    img_count = split_data.get('images', 0)
                    label_count = split_data.get('labels', 0)
                    img_percent = (img_count / total_images * 100) if total_images > 0 else 0
                    summary.append(f"📊 {split_name}: {img_count:,} gambar ({img_percent:.1f}%), {label_count:,} label")
            
            # Gabungkan pesan
            summary_message = "\n".join(summary)
            
            # Log ringkasan
            self.logger.info(summary_message)
            
            # Tambahkan ke log accordion jika tersedia
            if 'log_accordion' in self.ui_components and hasattr(self.ui_components['log_accordion'], 'add_log'):
                self.ui_components['log_accordion'].add_log(summary_message, 'success')
                
                # Expand log accordion
                if hasattr(self.ui_components['log_accordion'], 'selected_index'):
                    self.ui_components['log_accordion'].selected_index = 0
                    
        except Exception as e:
            self._log_with_emoji(f"Error saat menampilkan ringkasan dataset: {str(e)}", "warning")
