"""
File: smartcash/dataset/organizer/dataset_file_renamer.py
Deskripsi: Service untuk batch rename file existing ke format UUID consistency dengan research naming
"""

import os
import shutil
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import time

from smartcash.common.logger import get_logger
from smartcash.common.utils.file_naming_manager import FileNamingManager
from smartcash.common.worker_utils import get_optimal_worker_count

class DatasetFileRenamer:
    """Service untuk batch rename dataset ke UUID consistency format"""
    
    def __init__(self, config: Dict[str, Any], logger=None):
        self.config = config
        self.logger = logger or get_logger()
        self.naming_manager = FileNamingManager(config)
        self.stats = defaultdict(int)
        
    def batch_rename_dataset(self, data_dir: str = "data", backup: bool = True, 
                           progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """Batch rename seluruh dataset ke format UUID consistency"""
        start_time = time.time()
        self.logger.info(f"🔄 Memulai batch rename dataset: {data_dir}")
        
        try:
            # Phase 1: Discovery dan validation (0-15%)
            self._update_progress(5, "Scanning dataset structure", progress_callback)
            
            discovery_result = self._discover_dataset_files(data_dir)
            if not discovery_result['valid']:
                return self._error_result(f"Dataset discovery failed: {discovery_result['message']}")
            
            total_files = discovery_result['total_files']
            self.logger.info(f"📊 Ditemukan {total_files} file untuk rename")
            
            # Phase 2: Backup jika diminta (15-25%)
            if backup:
                self._update_progress(20, "Creating backup", progress_callback)
                backup_result = self._create_backup(data_dir)
                if not backup_result['success']:
                    self.logger.warning(f"⚠️ Backup failed: {backup_result['message']}")
            
            # Phase 3: Generate rename mapping (25-40%)
            self._update_progress(30, "Generating UUID mapping", progress_callback)
            rename_mapping = self._generate_rename_mapping(discovery_result['files'])
            
            # Phase 4: Execute rename (40-90%)
            self._update_progress(45, f"Renaming {len(rename_mapping)} files", progress_callback)
            rename_result = self._execute_batch_rename(rename_mapping, progress_callback)
            
            # Phase 5: Validation (90-100%)
            self._update_progress(95, "Validating renamed files", progress_callback)
            validation_result = self._validate_rename_result(data_dir)
            
            processing_time = time.time() - start_time
            
            return {
                'status': 'success',
                'total_files': total_files,
                'renamed_files': rename_result['success_count'],
                'failed_files': rename_result['error_count'],
                'processing_time': processing_time,
                'uuid_registry_size': len(self.naming_manager.uuid_registry),
                'validation': validation_result,
                'backup_created': backup
            }
            
        except Exception as e:
            error_msg = f"Batch rename error: {str(e)}"
            self.logger.error(f"❌ {error_msg}")
            return self._error_result(error_msg)
    
    def _discover_dataset_files(self, data_dir: str) -> Dict[str, Any]:
        """Discover semua file yang perlu direname dengan split awareness"""
        try:
            data_path = Path(data_dir)
            if not data_path.exists():
                return {'valid': False, 'message': f'Directory tidak ditemukan: {data_dir}'}
            
            files_to_rename = []
            
            # Strategy 1: Split-based discovery (train/valid/test)
            for split in ['train', 'valid', 'test']:
                split_path = data_path / split
                if split_path.exists():
                    split_files = self._discover_split_files(split_path, split)
                    files_to_rename.extend(split_files)
            
            # Strategy 2: Flat structure discovery (fallback)
            if not files_to_rename:
                flat_files = self._discover_flat_files(data_path)
                files_to_rename.extend(flat_files)
            
            return {
                'valid': True,
                'total_files': len(files_to_rename),
                'files': files_to_rename,
                'structure_type': 'split_based' if any('split' in f for f in files_to_rename) else 'flat'
            }
            
        except Exception as e:
            return {'valid': False, 'message': f'Discovery error: {str(e)}'}
    
    def _discover_split_files(self, split_path: Path, split: str) -> List[Dict[str, Any]]:
        """Discover files dalam split structure"""
        split_files = []
        
        # Images directory
        images_dir = split_path / 'images'
        if images_dir.exists():
            for img_file in images_dir.glob('*.*'):
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp'] and not self._is_already_renamed(img_file.name):
                    split_files.append({
                        'path': str(img_file),
                        'type': 'image',
                        'split': split,
                        'current_name': img_file.name,
                        'class_id': self._extract_class_from_label(img_file, split_path / 'labels')
                    })
        
        # Labels directory
        labels_dir = split_path / 'labels'
        if labels_dir.exists():
            for label_file in labels_dir.glob('*.txt'):
                if not self._is_already_renamed(label_file.name):
                    split_files.append({
                        'path': str(label_file),
                        'type': 'label',
                        'split': split,
                        'current_name': label_file.name,
                        'class_id': None  # Labels follow images
                    })
        
        return split_files
    
    def _discover_flat_files(self, data_path: Path) -> List[Dict[str, Any]]:
        """Discover files dalam flat structure"""
        flat_files = []
        
        # Direct images
        for img_file in data_path.glob('**/*.*'):
            if (img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp'] and 
                not self._is_already_renamed(img_file.name) and img_file.is_file()):
                flat_files.append({
                    'path': str(img_file),
                    'type': 'image',
                    'split': 'train',  # Default split
                    'current_name': img_file.name,
                    'class_id': self._extract_class_from_label(img_file, img_file.parent)
                })
        
        # Direct labels
        for label_file in data_path.glob('**/*.txt'):
            if not self._is_already_renamed(label_file.name) and label_file.is_file():
                flat_files.append({
                    'path': str(label_file),
                    'type': 'label',
                    'split': 'train',
                    'current_name': label_file.name,
                    'class_id': None
                })
        
        return flat_files
    
    def _is_already_renamed(self, filename: str) -> bool:
        """Check apakah file sudah menggunakan UUID format"""
        parsed = self.naming_manager.parse_existing_filename(filename)
        return parsed is not None
    
    def _extract_class_from_label(self, img_file: Path, labels_dir: Path) -> Optional[str]:
        """Extract primary class dari corresponding label file"""
        try:
            label_path = labels_dir / f"{img_file.stem}.txt"
            if not label_path.exists():
                return None
            
            class_counts = defaultdict(int)
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        try:
                            class_id = str(int(float(parts[0])))
                            class_counts[class_id] += 1
                        except (ValueError, IndexError):
                            continue
            
            return max(class_counts.items(), key=lambda x: x[1])[0] if class_counts else None
            
        except Exception:
            return None
    
    def _generate_rename_mapping(self, files: List[Dict[str, Any]]) -> Dict[str, str]:
        """Generate mapping old_path -> new_path dengan UUID consistency"""
        rename_mapping = {}
        image_uuid_map = {}  # stem -> uuid mapping untuk labels
        
        # Phase 1: Process images first untuk generate UUID
        for file_info in [f for f in files if f['type'] == 'image']:
            old_path = file_info['path']
            current_name = file_info['current_name']
            class_id = file_info['class_id']
            
            # Generate file info dengan UUID
            file_naming_info = self.naming_manager.generate_file_info(current_name, class_id, 'raw')
            new_filename = file_naming_info.get_filename()
            new_path = str(Path(old_path).parent / new_filename)
            
            rename_mapping[old_path] = new_path
            image_uuid_map[Path(current_name).stem] = file_naming_info.uuid
        
        # Phase 2: Process labels dengan UUID dari corresponding images
        for file_info in [f for f in files if f['type'] == 'label']:
            old_path = file_info['path']
            current_name = file_info['current_name']
            stem = Path(current_name).stem
            
            # Find corresponding UUID dari image
            corresponding_uuid = image_uuid_map.get(stem)
            if corresponding_uuid:
                # Use same UUID dan nominal info
                img_info = next((f for f in files if f['type'] == 'image' and Path(f['current_name']).stem == stem), None)
                if img_info:
                    file_naming_info = self.naming_manager.generate_file_info(img_info['current_name'], img_info['class_id'], 'raw')
                    new_filename = f"{Path(file_naming_info.get_filename()).stem}.txt"
                    new_path = str(Path(old_path).parent / new_filename)
                    rename_mapping[old_path] = new_path
        
        self.logger.info(f"🎯 Generated {len(rename_mapping)} rename mappings")
        return rename_mapping
    
    def _execute_batch_rename(self, rename_mapping: Dict[str, str], progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """Execute batch rename dengan parallel processing"""
        total_files = len(rename_mapping)
        if total_files == 0:
            return {'success_count': 0, 'error_count': 0, 'errors': []}
        
        results = {'success_count': 0, 'error_count': 0, 'errors': []}
        max_workers = min(get_optimal_worker_count('io'), 8)
        
        def rename_single_file(old_new_pair: Tuple[str, str]) -> Dict[str, Any]:
            old_path, new_path = old_new_pair
            try:
                if Path(old_path).exists() and old_path != new_path:
                    shutil.move(old_path, new_path)
                    return {'status': 'success', 'old': old_path, 'new': new_path}
                return {'status': 'skipped', 'reason': 'same_path_or_missing'}
            except Exception as e:
                return {'status': 'error', 'old': old_path, 'error': str(e)}
        
        # Process dengan ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_pair = {executor.submit(rename_single_file, pair): pair for pair in rename_mapping.items()}
            
            completed = 0
            for future in as_completed(future_to_pair):
                result = future.result()
                completed += 1
                
                if result['status'] == 'success':
                    results['success_count'] += 1
                elif result['status'] == 'error':
                    results['error_count'] += 1
                    results.setdefault('errors', []).append(result)
                
                # Progress update setiap 5%
                if completed % max(1, total_files // 20) == 0:
                    progress = 45 + int((completed / total_files) * 45)
                    self._update_progress(progress, f"Renamed: {completed}/{total_files}", progress_callback)
        
        self.logger.info(f"✅ Rename completed: {results['success_count']} success, {results['error_count']} errors")
        return results
    
    def _create_backup(self, data_dir: str) -> Dict[str, Any]:
        """Create backup dari dataset sebelum rename"""
        try:
            data_path = Path(data_dir)
            backup_path = data_path.parent / f"{data_path.name}_backup_{int(time.time())}"
            
            shutil.copytree(data_path, backup_path)
            
            return {
                'success': True,
                'backup_path': str(backup_path),
                'message': f'Backup created: {backup_path}'
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f'Backup failed: {str(e)}'
            }
    
    def _validate_rename_result(self, data_dir: str) -> Dict[str, Any]:
        """Validate hasil rename dengan UUID consistency check"""
        try:
            validation_result = {'uuid_consistent': True, 'issues': [], 'renamed_files': 0}
            
            # Scan renamed files
            for file_path in Path(data_dir).rglob('*.*'):
                if file_path.is_file() and file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.txt']:
                    parsed = self.naming_manager.parse_existing_filename(file_path.name)
                    if parsed:
                        validation_result['renamed_files'] += 1
                    elif not file_path.name.startswith('.'):  # Ignore hidden files
                        validation_result['issues'].append(f'Not renamed: {file_path.name}')
                        validation_result['uuid_consistent'] = False
            
            # Check image-label pairing
            pairing_issues = self._validate_image_label_pairing(data_dir)
            validation_result['issues'].extend(pairing_issues)
            
            if pairing_issues:
                validation_result['uuid_consistent'] = False
            
            return validation_result
            
        except Exception as e:
            return {'uuid_consistent': False, 'issues': [f'Validation error: {str(e)}'], 'renamed_files': 0}
    
    def _validate_image_label_pairing(self, data_dir: str) -> List[str]:
        """Validate image-label UUID pairing"""
        issues = []
        
        try:
            for split in ['train', 'valid', 'test']:
                split_path = Path(data_dir) / split
                if not split_path.exists():
                    continue
                
                images_dir = split_path / 'images'
                labels_dir = split_path / 'labels'
                
                if images_dir.exists() and labels_dir.exists():
                    # Check pairing
                    for img_file in images_dir.glob('rp_*.*'):
                        img_stem = img_file.stem
                        label_file = labels_dir / f"{img_stem}.txt"
                        
                        if not label_file.exists():
                            issues.append(f'Missing label for: {img_file.name}')
        
        except Exception as e:
            issues.append(f'Pairing validation error: {str(e)}')
        
        return issues
    
    def get_rename_preview(self, data_dir: str, limit: int = 10) -> Dict[str, Any]:
        """Preview rename mapping tanpa execute"""
        try:
            discovery_result = self._discover_dataset_files(data_dir)
            if not discovery_result['valid']:
                return {'status': 'error', 'message': discovery_result['message']}
            
            rename_mapping = self._generate_rename_mapping(discovery_result['files'])
            
            # Ambil sample untuk preview
            preview_items = list(rename_mapping.items())[:limit]
            
            return {
                'status': 'success',
                'total_files': len(rename_mapping),
                'preview': [
                    {
                        'old_name': Path(old).name,
                        'new_name': Path(new).name,
                        'old_path': old,
                        'new_path': new
                    } for old, new in preview_items
                ],
                'structure_type': discovery_result['structure_type']
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'Preview error: {str(e)}'}
    
    def rollback_rename(self, backup_path: str, target_dir: str) -> Dict[str, Any]:
        """Rollback rename dari backup"""
        try:
            backup_path_obj = Path(backup_path)
            target_path_obj = Path(target_dir)
            
            if not backup_path_obj.exists():
                return {'status': 'error', 'message': f'Backup tidak ditemukan: {backup_path}'}
            
            # Remove current dan restore dari backup
            if target_path_obj.exists():
                shutil.rmtree(target_path_obj)
            
            shutil.copytree(backup_path_obj, target_path_obj)
            
            return {
                'status': 'success',
                'message': f'Rollback berhasil dari: {backup_path}',
                'restored_to': str(target_path_obj)
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'Rollback error: {str(e)}'}
    
    def _update_progress(self, percentage: int, message: str, progress_callback: Optional[callable]) -> None:
        """Update progress dengan callback"""
        if progress_callback:
            try:
                progress_callback(percentage, message)
            except Exception:
                pass
    
    def _error_result(self, message: str) -> Dict[str, Any]:
        """Generate error result"""
        return {
            'status': 'error',
            'message': message,
            'renamed_files': 0,
            'failed_files': 0
        }

# Factory function dan utilities
def create_dataset_renamer(config: Dict[str, Any]) -> DatasetFileRenamer:
    """Factory untuk DatasetFileRenamer"""
    return DatasetFileRenamer(config)

# One-liner utilities
preview_rename = lambda data_dir, config, limit=10: create_dataset_renamer(config).get_rename_preview(data_dir, limit)
execute_rename = lambda data_dir, config, backup=True: create_dataset_renamer(config).batch_rename_dataset(data_dir, backup)
validate_uuid_consistency = lambda data_dir, config: create_dataset_renamer(config)._validate_rename_result(data_dir)