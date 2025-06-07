"""
File: smartcash/dataset/downloader/cleanup_behavior.py
Deskripsi: Analysis dan fix untuk cleanup behavior pada downloader module
"""

from typing import Dict, Any, List
from pathlib import Path
import shutil
import os
from smartcash.common.logger import get_logger

class DownloaderCleanupBehavior:
    """Analisis dan dokumentasi cleanup behavior untuk downloader module"""
    
    def __init__(self, logger=None):
        self.logger = logger or get_logger(__name__)
    
    def analyze_current_cleanup_process(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analisis proses cleanup saat ini pada downloader.
        
        Returns:
            Dictionary berisi analysis cleanup behavior
        """
        data_dir = Path(config.get('data', {}).get('dir', 'data'))
        
        analysis = {
            'cleanup_phases': self._get_cleanup_phases(),
            'target_directories': self._get_cleanup_targets(data_dir),
            'backup_behavior': self._analyze_backup_behavior(config),
            'safety_measures': self._get_safety_measures(),
            'recommendations': self._get_cleanup_recommendations()
        }
        
        return analysis
    
    def _get_cleanup_phases(self) -> List[Dict[str, str]]:
        """Dokumentasi fase-fase cleanup dalam download process"""
        return [
            {
                'phase': '1_pre_download',
                'description': 'Backup existing dataset (jika backup_existing=True)',
                'target': 'Seluruh direktori data/',
                'behavior': 'Copy ke backup directory, dataset lama tetap ada'
            },
            {
                'phase': '2_download_extract', 
                'description': 'Download dan extract ke temporary directory',
                'target': 'data/downloads/ (temporary)',
                'behavior': 'Clean extraction, tidak affect existing data'
            },
            {
                'phase': '3_organize_cleanup',
                'description': 'Hapus dataset lama sebelum organize yang baru',
                'target': 'data/train/, data/valid/, data/test/',
                'behavior': '🚨 PERMANEN DELETE - tidak bisa di-recover kecuali ada backup'
            },
            {
                'phase': '4_uuid_rename',
                'description': 'UUID renaming pada dataset baru',
                'target': 'Files dalam train/valid/test',
                'behavior': 'Rename in-place, preserve file content'
            },
            {
                'phase': '5_temp_cleanup',
                'description': 'Cleanup temporary download files',
                'target': 'data/downloads/',
                'behavior': 'Remove temporary files (aman)'
            }
        ]
    
    def _get_cleanup_targets(self, data_dir: Path) -> Dict[str, Dict[str, Any]]:
        """Identifikasi target directories yang akan di-cleanup"""
        targets = {}
        
        for split in ['train', 'valid', 'test']:
            split_dir = data_dir / split
            targets[split] = {
                'path': str(split_dir),
                'exists': split_dir.exists(),
                'cleanup_timing': 'before_organize',
                'recovery_possible': 'only_with_backup',
                'safety_level': 'HIGH_RISK'
            }
            
            if split_dir.exists():
                # Count existing files
                images_dir = split_dir / 'images'
                labels_dir = split_dir / 'labels'
                
                img_count = len(list(images_dir.glob('*.*'))) if images_dir.exists() else 0
                label_count = len(list(labels_dir.glob('*.txt'))) if labels_dir.exists() else 0
                
                targets[split].update({
                    'image_count': img_count,
                    'label_count': label_count,
                    'total_files': img_count + label_count
                })
        
        return targets
    
    def _analyze_backup_behavior(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Analisis backup behavior configuration"""
        download_config = config.get('download', {})
        
        return {
            'backup_enabled': download_config.get('backup_existing', False),
            'backup_location': 'data/backup/downloads/',
            'backup_scope': 'complete_dataset',
            'backup_timing': 'before_cleanup',
            'restore_procedure': 'manual_copy_back',
            'recommendation': '🔥 WAJIB AKTIFKAN backup_existing=True untuk safety!'
        }
    
    def _get_safety_measures(self) -> List[Dict[str, str]]:
        """Safety measures yang perlu diimplementasikan"""
        return [
            {
                'measure': 'backup_verification',
                'description': 'Verify backup success sebelum cleanup',
                'implementation': 'Check backup file count vs original'
            },
            {
                'measure': 'confirmation_dialog',
                'description': 'Show detailed confirmation dengan cleanup info',
                'implementation': 'Display file counts dan irreversible warning'
            },
            {
                'measure': 'rollback_capability', 
                'description': 'Quick rollback dari backup jika needed',
                'implementation': 'Provide restore function in UI'
            },
            {
                'measure': 'dry_run_mode',
                'description': 'Test mode untuk preview changes',
                'implementation': 'Show what will be deleted without actual deletion'
            }
        ]
    
    def _get_cleanup_recommendations(self) -> List[str]:
        """Rekomendasi untuk improve cleanup safety"""
        return [
            "🔒 Selalu aktifkan backup_existing=True secara default",
            "📋 Tampilkan detailed preview dari files yang akan dihapus",
            "⚠️ Gunakan confirmation dialog dengan explicit warning",
            "🔄 Implement rollback function untuk emergency restore",
            "📊 Log detailed cleanup statistics untuk audit trail",
            "🛡️ Tambahkan file integrity check post-cleanup",
            "⏱️ Implement timeout protection untuk large datasets"
        ]
    
    def implement_safe_cleanup_procedure(self, config: Dict[str, Any], 
                                       ui_components: Dict[str, Any]) -> bool:
        """
        Implement safe cleanup procedure dengan proper safeguards.
        
        Returns:
            Boolean indicating success of safe cleanup setup
        """
        try:
            # 1. Force enable backup jika ada existing data
            self._ensure_backup_enabled(config, ui_components)
            
            # 2. Create detailed cleanup plan
            cleanup_plan = self._create_cleanup_plan(config)
            
            # 3. Show confirmation dengan detailed info
            self._show_detailed_confirmation(cleanup_plan, ui_components)
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error setting up safe cleanup: {str(e)}")
            return False
    
    def _ensure_backup_enabled(self, config: Dict[str, Any], ui_components: Dict[str, Any]):
        """Ensure backup enabled untuk safety"""
        data_dir = Path(config.get('data', {}).get('dir', 'data'))
        has_existing = any((data_dir / split).exists() for split in ['train', 'valid', 'test'])
        
        if has_existing:
            # Force enable backup
            config.setdefault('download', {})['backup_existing'] = True
            
            # Update UI checkbox jika ada
            backup_checkbox = ui_components.get('backup_checkbox')
            if backup_checkbox and hasattr(backup_checkbox, 'value'):
                backup_checkbox.value = True
            
            self.logger.warning("🔒 Backup otomatis diaktifkan karena ada dataset existing")
    
    def _create_cleanup_plan(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create detailed cleanup plan untuk user preview"""
        data_dir = Path(config.get('data', {}).get('dir', 'data'))
        
        plan = {
            'will_be_deleted': [],
            'will_be_backed_up': [],
            'safety_status': 'SAFE' if config.get('download', {}).get('backup_existing') else 'RISKY'
        }
        
        for split in ['train', 'valid', 'test']:
            split_dir = data_dir / split
            if split_dir.exists():
                # Count files
                total_files = sum(1 for _ in split_dir.rglob('*.*'))
                plan['will_be_deleted'].append({
                    'directory': str(split_dir),
                    'file_count': total_files,
                    'size_estimate': self._estimate_directory_size(split_dir)
                })
                
                if config.get('download', {}).get('backup_existing'):
                    plan['will_be_backed_up'].append({
                        'source': str(split_dir),
                        'backup_location': f"data/backup/downloads/{split}",
                        'recovery_possible': True
                    })
        
        return plan
    
    def _estimate_directory_size(self, directory: Path) -> str:
        """Estimate directory size untuk user info"""
        try:
            total_size = sum(f.stat().st_size for f in directory.rglob('*.*') if f.is_file())
            
            # Convert to human readable
            for unit in ['B', 'KB', 'MB', 'GB']:
                if total_size < 1024.0:
                    return f"{total_size:.1f} {unit}"
                total_size /= 1024.0
            return f"{total_size:.1f} TB"
            
        except Exception:
            return "Unknown size"
    
    def _show_detailed_confirmation(self, cleanup_plan: Dict[str, Any], ui_components: Dict[str, Any]):
        """Show detailed confirmation dengan cleanup plan"""
        from smartcash.ui.dataset.downloader.utils.dialog_utils import show_download_confirmation_with_cleanup_info
        
        # Calculate totals
        total_files = sum(item['file_count'] for item in cleanup_plan['will_be_deleted'])
        total_dirs = len(cleanup_plan['will_be_deleted'])
        
        # Show enhanced confirmation
        show_download_confirmation_with_cleanup_info(
            ui_components, 
            total_files,
            {'cleanup_plan': cleanup_plan, 'safety_status': cleanup_plan['safety_status']},
            lambda btn: self._execute_safe_download(ui_components, cleanup_plan)
        )
    
    def _execute_safe_download(self, ui_components: Dict[str, Any], cleanup_plan: Dict[str, Any]):
        """Execute download dengan safe cleanup berdasarkan plan"""
        logger = ui_components.get('logger')
        if logger:
            safety_emoji = "🔒" if cleanup_plan['safety_status'] == 'SAFE' else "⚠️"
            logger.info(f"{safety_emoji} Memulai download dengan safety level: {cleanup_plan['safety_status']}")
        
        # Proceed dengan actual download implementation
        # This would call the original download handler
        pass

# Quick analysis function untuk debugging
def analyze_downloader_cleanup_behavior() -> str:
    """Quick analysis function untuk understand cleanup behavior"""
    analyzer = DownloaderCleanupBehavior()
    
    # Mock config untuk analysis
    mock_config = {
        'data': {'dir': 'data'},
        'download': {'backup_existing': False}
    }
    
    analysis = analyzer.analyze_current_cleanup_process(mock_config)
    
    report = [
        "📋 DOWNLOADER CLEANUP BEHAVIOR ANALYSIS",
        "=" * 50,
        "",
        "🔍 CLEANUP PHASES:",
        *[f"  {phase['phase']}: {phase['description']}" for phase in analysis['cleanup_phases']],
        "",
        "⚠️ HIGH-RISK PHASE: #3 organize_cleanup",
        "   • Target: data/train/, data/valid/, data/test/", 
        "   • Behavior: PERMANENT DELETE existing dataset",
        "   • Recovery: Only possible with backup enabled",
        "",
        "🔒 SAFETY RECOMMENDATIONS:",
        *[f"  • {rec}" for rec in analysis['recommendations']],
        "",
        "🚨 JAWABAN PERTANYAAN:",
        "   Q: Apakah cleanup existing data setelah download?",
        "   A: YA - Fase #3 menghapus dataset lama SEBELUM organize yang baru",
        "      Tanpa backup_existing=True, data HILANG PERMANEN!"
    ]
    
    return '\n'.join(report)