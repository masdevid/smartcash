#!/usr/bin/env python
# -*- coding: utf-8 -*-
# /Users/masdevid/Projects/smartcash/smartcash/ui/dataset/downloader/utils/progress_mapping.py
# Modul untuk mapping progress step ke persentase dan emoji untuk download handler

from typing import Dict, Tuple, Any

class ProgressMapping:
    """
    Kelas untuk mapping progress step ke persentase dan emoji
    Digunakan untuk standardisasi progress reporting di UI
    """
    
    # Emoji mapping untuk berbagai tahap
    EMOJI_MAPPING = {
        # Tahap inisialisasi
        'init': '🚀',
        'start': '🚀',
        'prepare': '🔧',
        'config': '⚙️',
        'validate': '✓',
        'validation': '✓',
        
        # Tahap metadata
        'metadata': '📋',
        'info': '📋',
        'parameters': '📝',
        
        # Tahap backup
        'backup': '💾',
        'backup_start': '💾',
        'backup_complete': '💾✓',
        
        # Tahap download
        'download': '📥',
        'download_start': '📥',
        'download_progress': '📥',
        'download_complete': '📥✓',
        
        # Tahap ekstraksi
        'extract': '📦',
        'extract_start': '📦',
        'extract_progress': '📦',
        'extract_complete': '📦✓',
        
        # Tahap organisasi
        'organize': '📂',
        'organize_start': '📂',
        'organize_scan': '🔍',
        'organize_scan_complete': '🔍✓',
        'organize_process': '📂',
        'organize_complete': '📂✓',
        
        # Tahap validasi
        'validate_start': '🔍',
        'validate_progress': '🔍',
        'validate_issues': '⚠️',
        'validate_complete': '✓',
        
        # Tahap cleanup
        'cleanup': '🧹',
        'cleanup_start': '🧹',
        'cleanup_complete': '🧹✓',
        
        # Status akhir
        'complete': '✅',
        'success': '✅',
        'done': '✅',
        
        # Error dan warning
        'error': '❌',
        'warning': '⚠️',
        'fail': '❌',
        'failed': '❌',
        
        # Default
        'default': '🔄'
    }
    
    # Progress range mapping untuk tahap utama
    # Format: (start_percentage, end_percentage)
    PROGRESS_RANGES = {
        # Tahap inisialisasi (0-10%)
        'init': (0, 5),
        'start': (0, 5),
        'prepare': (0, 5),
        'config': (0, 5),
        'validate_params': (5, 10),
        
        # Tahap metadata (10-20%)
        'metadata': (10, 20),
        'info': (10, 15),
        'parameters': (15, 20),
        
        # Tahap backup (20-30%)
        'backup': (20, 30),
        'backup_start': (20, 22),
        'backup_progress': (22, 28),
        'backup_complete': (28, 30),
        
        # Tahap download (30-60%)
        'download': (30, 60),
        'download_start': (30, 32),
        'download_progress': (32, 58),
        'download_complete': (58, 60),
        
        # Tahap ekstraksi (60-70%)
        'extract': (60, 70),
        'extract_start': (60, 62),
        'extract_progress': (62, 68),
        'extract_complete': (68, 70),
        
        # Tahap organisasi (70-85%)
        'organize': (70, 85),
        'organize_start': (70, 72),
        'organize_scan': (72, 75),
        'organize_scan_complete': (75, 77),
        'organize_process': (77, 83),
        'organize_complete': (83, 85),
        
        # Tahap validasi (85-95%)
        'validate': (85, 95),
        'validate_start': (85, 87),
        'validate_progress': (87, 92),
        'validate_issues': (92, 93),
        'validate_complete': (93, 95),
        
        # Tahap cleanup (95-98%)
        'cleanup': (95, 98),
        'cleanup_start': (95, 96),
        'cleanup_complete': (96, 98),
        
        # Status akhir (98-100%)
        'complete': (98, 100),
        'success': (98, 100),
        'done': (98, 100),
        
        # Error dan warning (tetap di persentase saat ini)
        'error': None,
        'warning': None,
        'fail': None,
        'failed': None,
        
        # Default
        'default': (0, 100)
    }
    
    @classmethod
    def get_emoji(cls, step: str) -> str:
        """
        Mendapatkan emoji untuk step tertentu
        
        Args:
            step: Nama step
            
        Returns:
            Emoji yang sesuai dengan step
        """
        step_lower = step.lower()
        
        # Cek apakah ada emoji khusus untuk step ini
        for key, emoji in cls.EMOJI_MAPPING.items():
            if key in step_lower:
                return emoji
        
        # Jika tidak ada yang cocok, gunakan default
        return cls.EMOJI_MAPPING['default']
    
    @classmethod
    def get_progress_range(cls, step: str) -> Tuple[int, int]:
        """
        Mendapatkan range persentase untuk step tertentu
        
        Args:
            step: Nama step
            
        Returns:
            Tuple (start_percentage, end_percentage)
        """
        step_lower = step.lower()
        
        # Cek apakah ada range khusus untuk step ini
        for key, range_value in cls.PROGRESS_RANGES.items():
            if key in step_lower and range_value is not None:
                return range_value
        
        # Jika tidak ada yang cocok, gunakan default
        return cls.PROGRESS_RANGES['default']
    
    @classmethod
    def calculate_percentage(cls, step: str, current: int, total: int) -> int:
        """
        Menghitung persentase progress berdasarkan step, current, dan total
        
        Args:
            step: Nama step
            current: Nilai progress saat ini
            total: Nilai total progress
            
        Returns:
            Persentase progress (0-100)
        """
        # Dapatkan range untuk step ini
        start_pct, end_pct = cls.get_progress_range(step)
        
        # Jika total adalah 0, kembalikan end_pct
        if total <= 0:
            return end_pct
        
        # Hitung persentase dalam range
        progress_in_range = current / total
        percentage = start_pct + (progress_in_range * (end_pct - start_pct))
        
        # Pastikan persentase dalam batas 0-100
        return max(0, min(100, int(percentage)))
    
    @classmethod
    def format_message(cls, step: str, message: str) -> str:
        """
        Format pesan dengan emoji yang sesuai
        
        Args:
            step: Nama step
            message: Pesan original
            
        Returns:
            Pesan yang sudah diformat dengan emoji
        """
        emoji = cls.get_emoji(step)
        
        # Jika pesan sudah memiliki emoji di awal, jangan tambahkan lagi
        if message and any(message.startswith(e) for e in cls.EMOJI_MAPPING.values()):
            return message
        
        return f"{emoji} {message}"
    
    @classmethod
    def is_error_step(cls, step: str) -> bool:
        """
        Cek apakah step adalah error step
        
        Args:
            step: Nama step
            
        Returns:
            True jika step adalah error step
        """
        step_lower = step.lower()
        return any(error_key in step_lower for error_key in ['error', 'fail', 'failed'])
    
    @classmethod
    def is_warning_step(cls, step: str) -> bool:
        """
        Cek apakah step adalah warning step
        
        Args:
            step: Nama step
            
        Returns:
            True jika step adalah warning step
        """
        step_lower = step.lower()
        return 'warning' in step_lower or 'issues' in step_lower
