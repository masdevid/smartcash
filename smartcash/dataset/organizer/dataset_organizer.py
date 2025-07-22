"""
File: smartcash/dataset/organizer/dataset_organizer.py
Deskripsi: Complete enhanced dataset organizer dengan UUID renaming integration dan one-liner style
"""

import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Callable
from smartcash.common.logger import get_logger
from smartcash.common.environment import get_environment_manager
from smartcash.dataset.utils.path_validator import get_path_validator
from smartcash.common.utils.file_naming_manager import FileNamingManager
from smartcash.dataset.downloader.file_processor import create_file_processor

class DatasetOrganizer:
    """Complete enhanced service untuk organize dataset dengan UUID renaming support."""
    
    def __init__(self, logger=None):
        self.logger = logger or get_logger()
        self.env_manager = get_environment_manager()
        self.path_validator = get_path_validator(logger)
        self._progress_callback: Optional[Callable] = None
        
        # Enhanced initialization dengan UUID support
        self.file_naming_manager = FileNamingManager(logger=logger)
        self.file_processor = create_file_processor(logger)
        self.enable_uuid_by_default = True
    
    def set_progress_callback(self, callback: Callable[[str, int, int, str], None]) -> None:
        """Set progress callback dengan one-liner propagation"""
        self._progress_callback = callback
        if hasattr(self.file_processor, 'set_progress_callback'):
            self.file_processor.set_progress_callback(callback)
    
    def organize_dataset(self, source_dir: str, remove_source: bool = True) -> Dict[str, Any]:
        """Enhanced organize dataset dengan UUID support (backward compatible)"""
        return self.organize_with_uuid_renaming(source_dir, remove_source, self.enable_uuid_by_default)
    
    def organize_with_uuid_renaming(self, source_dir: str, remove_source: bool = True, 
                                   enable_uuid: bool = True) -> Dict[str, Any]:
        """Enhanced organize dataset dengan UUID renaming integration dan gradual progress"""
        source_path = Path(source_dir)
        
        if not source_path.exists():
            return {'status': 'error', 'message': f'Source tidak ditemukan: {source_dir}'}
        
        self.logger.info(f"📁 Mengorganisir dataset dengan UUID: {source_dir}")
        self._notify_progress("organize", 0, 100, "Memulai organisasi dataset")
        
        try:
            # Step 1: Detect dan validate structure (0-15%)
            self._notify_progress("organize", 5, 100, "Mendeteksi struktur dataset")
            
            target_paths = self.path_validator.get_dataset_paths()
            splits_found = self.path_validator.detect_available_splits(str(source_path))
            
            if not splits_found:
                return {'status': 'error', 'message': 'Tidak ada split dataset yang valid ditemukan'}
            
            self.logger.info(f"📊 Splits ditemukan: {', '.join(splits_found)}")
            self.logger.info(f"🎯 Target base: {target_paths['data_root']}")
            
            # Step 2: Prepare directories (15-25%)
            self._notify_progress("organize", 20, 100, "Menyiapkan direktori target")
            self._prepare_target_directories(target_paths)
            
            # Step 3: Execute organization dengan UUID (25-85%)
            organize_result = self._execute_organization_with_progress(
                source_path, target_paths, enable_uuid, 25, 85
            )
            
            if organize_result['status'] != 'success':
                return {'status': 'error', 'message': organize_result['message']}
            
            # Step 4: Cleanup source jika diminta (85-95%)
            if remove_source:
                self._notify_progress("organize", 90, 100, "Membersihkan source directory")
                self._cleanup_source(source_path)
            
            # Step 5: Final validation dan completion (95-100%)
            final_result = self._finalize_organization_result(
                organize_result, target_paths, splits_found, enable_uuid
            )
            
            self._notify_progress("organize", 100, 100, f"Organisasi selesai: {final_result['total_images']} gambar")
            return final_result
            
        except Exception as e:
            error_msg = f"Error organizasi dataset: {str(e)}"
            self.logger.error(f"❌ {error_msg}")
            return {'status': 'error', 'message': error_msg}
    
    def _execute_organization_with_progress(self, source_path: Path, target_paths: Dict[str, str], 
                                          enable_uuid: bool, start_progress: int, end_progress: int) -> Dict[str, Any]:
        """Execute organization dengan progress tracking yang detailed"""
        try:
            target_dir = Path(target_paths['data_root'])
            
            # Progress callback untuk nested operations
            def nested_progress_callback(step: str, current: int, total: int, message: str):
                # Map nested progress ke overall progress range
                nested_progress = start_progress + int(((current / 100) * (end_progress - start_progress)))
                self._notify_progress("organize", nested_progress, 100, message)
            
            # Set nested callback
            if hasattr(self.file_processor, 'set_progress_callback'):
                self.file_processor.set_progress_callback(nested_progress_callback)
            
            # Execute appropriate organization method
            if enable_uuid:
                self._notify_progress("organize", start_progress + 5, 100, "🔤 Mengorganisir dengan UUID renaming")
                organize_result = self.file_processor.organize_dataset_with_renaming(source_path, target_dir)
            else:
                self._notify_progress("organize", start_progress + 5, 100, "📁 Mengorganisir tanpa UUID renaming")
                organize_result = self.file_processor.organize_dataset(source_path, target_dir)
            
            return organize_result
            
        except Exception as e:
            return {'status': 'error', 'message': f'Organization execution error: {str(e)}'}
    
    def _finalize_organization_result(self, organize_result: Dict[str, Any], target_paths: Dict[str, str], 
                                    splits_found: list, enable_uuid: bool) -> Dict[str, Any]:
        """Finalize organization result dengan enhanced statistics"""
        total_images = organize_result.get('total_images', 0)
        total_labels = organize_result.get('total_labels', 0)
        uuid_renamed = organize_result.get('uuid_renamed', False) and enable_uuid
        
        success_message = f"Dataset berhasil diorganisir ke {target_paths['data_root']}"
        if uuid_renamed:
            success_message += " dengan UUID format"
        
        # Enhanced logging dengan statistics
        self.logger.success(
            f"✅ {success_message}\n"
            f"   • Total gambar: {total_images}\n"
            f"   • Total label: {total_labels}\n"
            f"   • UUID format: {'Ya' if uuid_renamed else 'Tidak'}\n"
            f"   • Splits: {', '.join(splits_found)}"
        )
        
        result = {
            'status': 'success',
            'message': success_message,
            'total_images': total_images,
            'total_labels': total_labels,
            'splits': organize_result.get('splits', {}),
            'target_paths': target_paths,
            'uuid_renamed': uuid_renamed,
            'splits_processed': splits_found
        }
        
        # Add UUID statistics jika tersedia
        if uuid_renamed and hasattr(self.file_processor, 'get_naming_statistics'):
            naming_stats = self.file_processor.get_naming_statistics()
            result['naming_stats'] = naming_stats
            
            if naming_stats.get('total_files', 0) > 0:
                self.logger.info(f"🔤 UUID Statistics: {naming_stats}")
        
        return result
    
    def _prepare_target_directories(self, target_paths: Dict[str, str]) -> None:
        """Prepare target directories dengan one-liner creation"""
        for split in ['train', 'valid', 'test']:
            if split in target_paths:
                target_path = Path(target_paths[split])
                target_path.mkdir(parents=True, exist_ok=True)
                (target_path / 'images').mkdir(exist_ok=True)
                (target_path / 'labels').mkdir(exist_ok=True)
    
    def _move_split_with_progress(self, source_path: Path, split_name: str, 
                                 target_path_str: str, start_progress: int, end_progress: int) -> Dict[str, Any]:
        """Enhanced move split dengan UUID awareness dan fixed path detection"""
        target_path = Path(target_path_str)
        
        try:
            # Get actual source path menggunakan path validator
            source_split_path = self.path_validator.get_split_path(str(source_path), split_name)
            
            if not source_split_path.exists():
                return {'status': 'error', 'message': f'Split directory tidak ditemukan: {source_split_path}'}
            
            source_images = source_split_path / 'images'
            source_labels = source_split_path / 'labels'
            
            image_count = len(list(source_images.glob('*.*'))) if source_images.exists() else 0
            label_count = len(list(source_labels.glob('*.txt'))) if source_labels.exists() else 0
            
            total_files = image_count + label_count
            current_file = 0
            
            # Move images dengan progress
            if source_images.exists() and image_count > 0:
                target_images = target_path / 'images'
                target_images.mkdir(parents=True, exist_ok=True)
                
                for img_file in source_images.glob('*.*'):
                    shutil.copy2(img_file, target_images / img_file.name)
                    current_file += 1
                    
                    # Update progress secara gradual
                    if current_file % max(1, total_files // 10) == 0:
                        file_progress = int((current_file / total_files) * (end_progress - start_progress))
                        current_progress = start_progress + file_progress
                        self._notify_progress("organize", current_progress, 100, 
                                            f"Menyalin {split_name}: {current_file}/{total_files}")
            
            # Move labels dengan progress
            if source_labels.exists() and label_count > 0:
                target_labels = target_path / 'labels'
                target_labels.mkdir(parents=True, exist_ok=True)
                
                for label_file in source_labels.glob('*.txt'):
                    shutil.copy2(label_file, target_labels / label_file.name)
                    current_file += 1
                    
                    # Update progress secara gradual
                    if current_file % max(1, total_files // 10) == 0:
                        file_progress = int((current_file / total_files) * (end_progress - start_progress))
                        current_progress = start_progress + file_progress
                        self._notify_progress("organize", current_progress, 100, 
                                            f"Menyalin {split_name}: {current_file}/{total_files}")
            
            self._copy_additional_files(source_split_path, target_path)
            
            self.logger.info(f"📁 Split {split_name}: {image_count} gambar, {label_count} label → {target_path}")
            
            return {
                'status': 'success',
                'images': image_count,
                'labels': label_count,
                'path': str(target_path)
            }
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def _copy_additional_files(self, source_dir: Path, target_dir: Path) -> None:
        """Copy file tambahan dengan one-liner implementation"""
        additional_files = ['data.yaml', 'dataset.yaml', 'classes.txt', 'README.md']
        
        [shutil.copy2(source_file, target_dir / filename) if (source_file := source_dir / filename).exists() else None
         for filename in additional_files]
    
    def _cleanup_source(self, source_path: Path) -> None:
        """Cleanup source directory dengan one-liner safe execution"""
        try:
            if source_path.exists():
                shutil.rmtree(source_path, ignore_errors=True)
                self.logger.info(f"🗑️ Source directory dihapus: {source_path}")
        except Exception as e:
            self.logger.warning(f"⚠️ Gagal hapus source: {str(e)}")
    
    def check_organized_dataset(self) -> Dict[str, Any]:
        """Enhanced check status dataset dengan UUID awareness"""
        target_paths = self.path_validator.get_dataset_paths()
        validation_result = self.path_validator.validate_dataset_structure(target_paths['data_root'])
        
        # Enhanced validation dengan UUID check
        uuid_info = {}
        if hasattr(self.file_processor, 'validate_dataset_structure'):
            uuid_validation = self.file_processor.validate_dataset_structure(Path(target_paths['data_root']))
            uuid_info = {
                'uuid_format': uuid_validation.get('uuid_format', False),
                'uuid_consistent': uuid_validation.get('uuid_format', False)
            }
        
        return {
            'is_organized': validation_result['valid'] and validation_result['total_images'] > 0,
            'total_images': validation_result['total_images'],
            'total_labels': validation_result['total_labels'],
            'splits': validation_result['splits'],
            'issues': validation_result['issues'],
            **uuid_info
        }
    
    def check_uuid_consistency(self, data_dir: str = None) -> Dict[str, Any]:
        """Enhanced check UUID consistency dalam dataset yang sudah diorganisir"""
        target_paths = self.path_validator.get_dataset_paths()
        data_dir = data_dir or target_paths['data_root']
        
        self.logger.info(f"🔍 Checking UUID consistency: {data_dir}")
        
        validation_result = self.file_processor.validate_dataset_structure(Path(data_dir))
        
        consistency_result = {
            'uuid_consistent': validation_result.get('uuid_format', False),
            'total_files': validation_result.get('total_images', 0) + validation_result.get('total_labels', 0),
            'issues': validation_result.get('issues', []),
            'splits': validation_result.get('splits', {}),
            'message': 'UUID consistency check completed'
        }
        
        # Enhanced reporting
        if consistency_result['uuid_consistent']:
            self.logger.success(f"✅ UUID consistency valid: {consistency_result['total_files']} files")
        else:
            self.logger.warning(f"⚠️ UUID consistency issues found: {len(consistency_result['issues'])} issues")
        
        return consistency_result
    
    def batch_rename_to_uuid(self, data_dir: str = None, backup: bool = True) -> Dict[str, Any]:
        """Enhanced batch rename existing dataset ke UUID format"""
        from smartcash.dataset.organizer.dataset_file_renamer import create_dataset_renamer
        
        target_paths = self.path_validator.get_dataset_paths()
        data_dir = data_dir or target_paths['data_root']
        
        self.logger.info(f"🔄 Memulai batch rename ke UUID format: {data_dir}")
        
        config = {'enable_uuid': True, 'backup': backup}
        renamer = create_dataset_renamer(config)
        
        def progress_callback(percentage, message):
            self._notify_progress("rename", percentage, 100, message)
        
        result = renamer.batch_rename_dataset(data_dir, backup, progress_callback)
        
        if result['status'] == 'success':
            self.logger.success(f"✅ Batch rename selesai: {result['renamed_files']} files")
            
            # Update UUID registry dari naming manager
            if hasattr(renamer, 'naming_manager'):
                self.file_naming_manager.uuid_registry.update(renamer.naming_manager.uuid_registry)
        else:
            self.logger.error(f"❌ Batch rename gagal: {result['message']}")
        
        return result
    
    def get_uuid_preview(self, data_dir: str = None, limit: int = 10) -> Dict[str, Any]:
        """Get preview dari rename operations tanpa execute"""
        from smartcash.dataset.organizer.dataset_file_renamer import create_dataset_renamer
        
        target_paths = self.path_validator.get_dataset_paths()
        data_dir = data_dir or target_paths['data_root']
        
        config = {'enable_uuid': True}
        renamer = create_dataset_renamer(config)
        
        return renamer.get_rename_preview(data_dir, limit)
    
    def cleanup_all_dataset_folders(self) -> Dict[str, Any]:
        """Enhanced cleanup dengan UUID awareness dan gradual progress tracking"""
        target_paths = self.path_validator.get_dataset_paths()
        
        cleanup_stats = {
            'total_files_removed': 0,
            'folders_cleaned': [],
            'errors': [],
            'uuid_files_found': 0,
            'legacy_files_found': 0
        }
        
        # Count total files untuk progress calculation
        total_files_to_remove = 0
        folders_to_clean = []
        
        self._notify_progress("cleanup", 5, 100, "Menghitung file yang akan dihapus")
        
        # Count files in splits dengan UUID detection
        for split in ['train', 'valid', 'test']:
            split_path = Path(target_paths[split])
            if split_path.exists():
                try:
                    uuid_count, legacy_count, total_count = self._count_files_by_type(split_path)
                    if total_count > 0:
                        total_files_to_remove += total_count
                        folders_to_clean.append((split, split_path, total_count, uuid_count, legacy_count))
                        cleanup_stats['uuid_files_found'] += uuid_count
                        cleanup_stats['legacy_files_found'] += legacy_count
                except Exception as e:
                    cleanup_stats['errors'].append(f"Error counting {split}: {str(e)}")
        
        # Count downloads folder
        downloads_path = Path(target_paths.get('downloads', f"{target_paths['data_root']}/downloads"))
        if downloads_path.exists():
            try:
                _, _, file_count = self._count_files_by_type(downloads_path)
                if file_count > 0:
                    total_files_to_remove += file_count
                    folders_to_clean.append(('downloads', downloads_path, file_count, 0, 0))
            except Exception as e:
                cleanup_stats['errors'].append(f"Error counting downloads: {str(e)}")
        
        if total_files_to_remove == 0:
            self._notify_progress("cleanup", 100, 100, "Tidak ada file untuk dihapus")
            return {
                'status': 'empty',
                'message': 'Tidak ada file untuk dihapus di folder dataset',
                'stats': cleanup_stats
            }
        
        # Enhanced cleanup dengan gradual progress dan UUID awareness
        files_removed = 0
        
        for i, folder_info in enumerate(folders_to_clean):
            folder_name, folder_path, file_count = folder_info[:3]
            uuid_count = folder_info[3] if len(folder_info) > 3 else 0
            legacy_count = folder_info[4] if len(folder_info) > 4 else 0
            
            start_progress = 10 + (i * 80 // len(folders_to_clean))
            end_progress = 10 + ((i + 1) * 80 // len(folders_to_clean))
            
            self._notify_progress("cleanup", start_progress, 100, f"Menghapus folder {folder_name}")
            
            try:
                # Delete files dengan progress tracking
                current_file = 0
                for file_path in folder_path.rglob('*'):
                    if file_path.is_file():
                        try:
                            file_path.unlink()
                            current_file += 1
                            files_removed += 1
                            
                            # Update progress setiap 10% dari folder ini
                            if current_file % max(1, file_count // 10) == 0:
                                folder_progress = int((current_file / file_count) * (end_progress - start_progress))
                                current_progress = start_progress + folder_progress
                                self._notify_progress("cleanup", current_progress, 100, 
                                                    f"Menghapus {folder_name}: {current_file}/{file_count}")
                        except Exception:
                            pass
                
                # Remove empty directories
                try:
                    if folder_path.exists():
                        shutil.rmtree(folder_path, ignore_errors=True)
                        folder_summary = f"{folder_name} ({file_count} files"
                        if uuid_count > 0:
                            folder_summary += f", {uuid_count} UUID"
                        if legacy_count > 0:
                            folder_summary += f", {legacy_count} legacy"
                        folder_summary += ")"
                        
                        cleanup_stats['folders_cleaned'].append(folder_summary)
                        self.logger.info(f"🗑️ Cleaned {folder_summary}")
                except Exception as e:
                    cleanup_stats['errors'].append(f"Error cleaning {folder_name}: {str(e)}")
                    
            except Exception as e:
                cleanup_stats['errors'].append(f"Error processing {folder_name}: {str(e)}")
        
        cleanup_stats['total_files_removed'] = files_removed
        
        self._notify_progress("cleanup", 100, 100, f"Cleanup selesai: {files_removed} file dihapus")
        
        # Enhanced reporting dengan UUID info
        if cleanup_stats['uuid_files_found'] > 0:
            self.logger.info(f"🔤 UUID files removed: {cleanup_stats['uuid_files_found']}")
        if cleanup_stats['legacy_files_found'] > 0:
            self.logger.info(f"📁 Legacy files removed: {cleanup_stats['legacy_files_found']}")
        
        return {
            'status': 'success', 
            'message': f"Berhasil menghapus {files_removed} file",
            'stats': cleanup_stats
        }
    
    def _count_files_by_type(self, directory: Path) -> tuple:
        """Count files by type (UUID, legacy, total) dengan one-liner detection"""
        try:
            all_files = [f for f in directory.rglob('*') if f.is_file()]
            
            uuid_count = sum(1 for f in all_files if self.file_naming_manager.validate_filename_format(f.name)['valid'])
            legacy_count = len(all_files) - uuid_count
            total_count = len(all_files)
            
            return uuid_count, legacy_count, total_count
        except Exception:
            return 0, 0, 0
    
    def _notify_progress(self, step: str, current: int, total: int, message: str) -> None:
        """Progress notification dengan safe execution"""
        if self._progress_callback:
            try:
                self._progress_callback(step, current, total, message)
            except Exception:
                pass
    
    def get_organizer_info(self) -> Dict[str, Any]:
        """Get comprehensive organizer information dengan UUID support"""
        return {
            'uuid_support': True,
            'uuid_enabled_by_default': self.enable_uuid_by_default,
            'has_file_processor': hasattr(self, 'file_processor'),
            'has_naming_manager': hasattr(self, 'file_naming_manager'),
            'environment': {
                'is_colab': self.env_manager.is_colab,
                'drive_mounted': self.env_manager.is_drive_mounted
            },
            'registry_size': len(self.file_naming_manager.uuid_registry) if hasattr(self, 'file_naming_manager') else 0
        }

# Enhanced factory functions dengan UUID support
def create_dataset_organizer(logger=None, enable_uuid: bool = True) -> DatasetOrganizer:
    """Enhanced factory untuk DatasetOrganizer dengan UUID support option"""
    organizer = DatasetOrganizer(logger)
    organizer.enable_uuid_by_default = enable_uuid
    return organizer

def create_uuid_enabled_organizer(logger=None) -> DatasetOrganizer:
    """One-liner factory untuk UUID-enabled organizer"""
    return create_dataset_organizer(logger, enable_uuid=True)

def create_legacy_organizer(logger=None) -> DatasetOrganizer:
    """One-liner factory untuk legacy organizer (tanpa UUID)"""
    return create_dataset_organizer(logger, enable_uuid=False)

# One-liner utilities
organize_with_uuid = lambda source_dir, logger=None: create_uuid_enabled_organizer(logger).organize_dataset(source_dir)
organize_without_uuid = lambda source_dir, logger=None: create_legacy_organizer(logger).organize_dataset(source_dir)
check_dataset_uuid_consistency = lambda data_dir=None, logger=None: create_dataset_organizer(logger).check_uuid_consistency(data_dir)
preview_uuid_rename = lambda data_dir=None, logger=None: create_dataset_organizer(logger).get_uuid_preview(data_dir)