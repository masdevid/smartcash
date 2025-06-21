"""
File: smartcash/dataset/downloader/progress_tracker.py
Deskripsi: Backend progress tracker dengan mapping tahapan yang konsisten (mengganti progress_mapping.py)
"""

from typing import Dict, Any, Callable, Optional
from enum import Enum

class DownloadStage(Enum):
    """Enum untuk tahapan download termasuk UUID renaming"""
    INIT = "init"
    METADATA = "metadata"
    BACKUP = "backup"
    DOWNLOAD = "download"
    EXTRACT = "extract"
    ORGANIZE = "organize"
    UUID_RENAME = "uuid_rename"  # 🆕 NEW: Tahapan UUID renaming
    VALIDATE = "validate"
    CLEANUP = "cleanup"
    COMPLETE = "complete"

class DownloadProgressTracker:
    """Backend progress tracker untuk download operations dengan UUID renaming support"""
    
    # Progress mapping untuk each stage (start%, end%) - UPDATED dengan UUID rename
    STAGE_PROGRESS_MAP = {
        DownloadStage.INIT: (0, 10),
        DownloadStage.METADATA: (10, 20),
        DownloadStage.BACKUP: (20, 25),
        DownloadStage.DOWNLOAD: (25, 55),
        DownloadStage.EXTRACT: (55, 65),
        DownloadStage.ORGANIZE: (65, 75),
        DownloadStage.UUID_RENAME: (75, 85),  # 🆕 NEW: UUID renaming stage
        DownloadStage.VALIDATE: (85, 92),
        DownloadStage.CLEANUP: (92, 96),
        DownloadStage.COMPLETE: (96, 100)
    }
    
    # Stage emojis - UPDATED dengan UUID rename
    STAGE_EMOJIS = {
        DownloadStage.INIT: "🚀",
        DownloadStage.METADATA: "📋",
        DownloadStage.BACKUP: "💾",
        DownloadStage.DOWNLOAD: "📥",
        DownloadStage.EXTRACT: "📦",
        DownloadStage.ORGANIZE: "🗂️",
        DownloadStage.UUID_RENAME: "🔄",  # 🆕 NEW: UUID renaming emoji
        DownloadStage.VALIDATE: "✅",
        DownloadStage.CLEANUP: "🧹",
        DownloadStage.COMPLETE: "✅"
    }
    
    # Stage descriptions - INTEGRATED dari progress_mapping.py
    STAGE_DESCRIPTIONS = {
        DownloadStage.INIT: "Inisialisasi",
        DownloadStage.METADATA: "Mengambil metadata",
        DownloadStage.BACKUP: "Membuat backup",
        DownloadStage.DOWNLOAD: "Mengunduh dataset",
        DownloadStage.EXTRACT: "Mengekstrak file",
        DownloadStage.ORGANIZE: "Mengorganisasi struktur",
        DownloadStage.UUID_RENAME: "Penamaan ulang dengan UUID",  # 🆕 NEW
        DownloadStage.VALIDATE: "Memvalidasi hasil",
        DownloadStage.CLEANUP: "Membersihkan file sementara",
        DownloadStage.COMPLETE: "Selesai"
    }
    
    def __init__(self, callback: Optional[Callable[[str, int, int, str], None]] = None):
        self.callback = callback
        self.current_stage = None
        self.stage_progress = 0
        self.overall_progress = 0
        
    def set_callback(self, callback: Callable[[str, int, int, str], None]) -> None:
        """Set progress callback dengan signature (step, current, total, message)"""
        self.callback = callback
    
    def start_stage(self, stage: DownloadStage, message: str = "") -> None:
        """Start new stage"""
        self.current_stage = stage
        self.stage_progress = 0
        
        start_progress, _ = self.STAGE_PROGRESS_MAP[stage]
        self.overall_progress = start_progress
        
        emoji = self.STAGE_EMOJIS[stage]
        formatted_message = f"{emoji} {message}" if message else f"{emoji} {stage.value.title()}"
        
        self._notify(stage.value, start_progress, 100, formatted_message)
    
    def update_stage(self, progress_percent: int, message: str = "") -> None:
        """Update current stage progress (0-100)"""
        if not self.current_stage:
            return
            
        self.stage_progress = max(0, min(100, progress_percent))
        
        # Map stage progress ke overall progress
        start_progress, end_progress = self.STAGE_PROGRESS_MAP[self.current_stage]
        stage_range = end_progress - start_progress
        
        self.overall_progress = start_progress + (self.stage_progress / 100 * stage_range)
        
        emoji = self.STAGE_EMOJIS[self.current_stage]
        formatted_message = f"{emoji} {message}" if message else f"{emoji} {self.current_stage.value.title()}: {progress_percent}%"
        
        self._notify(self.current_stage.value, int(self.overall_progress), 100, formatted_message)
    
    def complete_stage(self, message: str = "") -> None:
        """Complete current stage"""
        if not self.current_stage:
            return
            
        self.stage_progress = 100
        _, end_progress = self.STAGE_PROGRESS_MAP[self.current_stage]
        self.overall_progress = end_progress
        
        emoji = self.STAGE_EMOJIS[self.current_stage]
        formatted_message = f"{emoji} {message}" if message else f"{emoji} {self.current_stage.value.title()} selesai"
        
        self._notify(self.current_stage.value, int(self.overall_progress), 100, formatted_message)
    
    def error(self, message: str) -> None:
        """Report error"""
        self._notify("error", 0, 100, f"❌ {message}")
    
    def complete_all(self, message: str = "Download selesai") -> None:
        """Complete all operations"""
        self.current_stage = DownloadStage.COMPLETE
        self.overall_progress = 100
        self._notify("complete", 100, 100, f"✅ {message}")
    
    def _notify(self, step: str, current: int, total: int, message: str) -> None:
        """Send notification via callback"""
        if self.callback:
            try:
                self.callback(step, current, total, message)
            except Exception:
                pass  # Silent fail untuk prevent callback issues
    
    def get_status(self) -> Dict[str, Any]:
        """Get current tracker status"""
        return {
            'current_stage': self.current_stage.value if self.current_stage else None,
            'stage_progress': self.stage_progress,
            'overall_progress': self.overall_progress,
            'has_callback': self.callback is not None
        }

def create_progress_tracker(callback: Optional[Callable] = None) -> DownloadProgressTracker:
    """Factory untuk DownloadProgressTracker"""
    return DownloadProgressTracker(callback)