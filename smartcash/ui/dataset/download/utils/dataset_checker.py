"""
File: smartcash/ui/dataset/download/utils/dataset_checker.py
Deskripsi: Utility untuk check status dataset dengan comprehensive analysis
"""

from typing import Dict, Any, List
from pathlib import Path
from smartcash.common.constants.paths import get_paths_for_environment
from smartcash.common.environment import get_environment_manager

def check_complete_dataset_status() -> Dict[str, Any]:
    """
    Check status lengkap dataset termasuk final structure dan downloads folder.
    
    Returns:
        Dictionary berisi status lengkap dataset
    """
    env_manager = get_environment_manager()
    paths = get_paths_for_environment(
        is_colab=env_manager.is_colab,
        is_drive_mounted=env_manager.is_drive_mounted
    )
    
    result = {
        'final_dataset': check_final_dataset_structure(paths),
        'downloads_folder': check_downloads_folder(paths),
        'storage_info': {
            'type': 'Google Drive' if env_manager.is_drive_mounted else 'Local Storage',
            'base_path': env_manager.drive_path if env_manager.is_drive_mounted else '/content',
            'persistent': env_manager.is_drive_mounted
        },
        'summary': {}
    }
    
    # Generate summary
    result['summary'] = _generate_dataset_summary(result)
    
    return result

def check_final_dataset_structure(paths: Dict[str, str]) -> Dict[str, Any]:
    """Check struktur final dataset (train/valid/test)."""
    final_stats = {
        'exists': False,
        'total_images': 0,
        'total_labels': 0,
        'splits': {},
        'base_dir': paths['data_root'],
        'issues': []
    }
    
    for split in ['train', 'valid', 'test']:
        split_info = _analyze_split_directory(paths[split], split)
        final_stats['splits'][split] = split_info
        final_stats['total_images'] += split_info['images']
        final_stats['total_labels'] += split_info['labels']
        
        # Collect issues
        if split_info['issues']:
            final_stats['issues'].extend(split_info['issues'])
    
    final_stats['exists'] = final_stats['total_images'] > 0
    
    return final_stats

def check_downloads_folder(paths: Dict[str, str]) -> Dict[str, Any]:
    """Check status downloads folder."""
    downloads_stats = {
        'exists': False,
        'total_files': 0,
        'total_size_mb': 0,
        'path': paths['downloads'],
        'contents': []
    }
    
    try:
        downloads_path = Path(paths['downloads'])
        
        if downloads_path.exists():
            files = list(downloads_path.rglob('*.*'))
            downloads_stats['total_files'] = len(files)
            downloads_stats['exists'] = len(files) > 0
            
            # Analyze contents
            downloads_stats['contents'] = _analyze_downloads_contents(downloads_path)
            
            # Calculate total size
            try:
                total_size = sum(f.stat().st_size for f in files if f.is_file())
                downloads_stats['total_size_mb'] = round(total_size / (1024 * 1024), 2)
            except Exception:
                downloads_stats['total_size_mb'] = 0
                
    except Exception:
        pass
    
    return downloads_stats

def _analyze_split_directory(split_path_str: str, split_name: str) -> Dict[str, Any]:
    """Analyze single split directory dengan detail."""
    split_info = {
        'exists': False,
        'images': 0,
        'labels': 0,
        'path': split_path_str,
        'images_path': '',
        'labels_path': '',
        'issues': [],
        'file_types': {'images': set(), 'labels': set()}
    }
    
    try:
        split_path = Path(split_path_str)
        
        if not split_path.exists():
            split_info['issues'].append(f"❌ Split {split_name}: Directory tidak ditemukan")
            return split_info
        
        images_dir = split_path / 'images'
        labels_dir = split_path / 'labels'
        
        split_info['images_path'] = str(images_dir)
        split_info['labels_path'] = str(labels_dir)
        
        # Analyze images directory
        if images_dir.exists():
            img_files = list(images_dir.glob('*.*'))
            split_info['images'] = len(img_files)
            split_info['exists'] = len(img_files) > 0
            
            # Collect image file types
            for img_file in img_files[:10]:  # Sample first 10 files
                split_info['file_types']['images'].add(img_file.suffix.lower())
                
            if split_info['images'] == 0:
                split_info['issues'].append(f"⚠️ Split {split_name}: Folder images kosong")
        else:
            split_info['issues'].append(f"❌ Split {split_name}: Folder images tidak ditemukan")
        
        # Analyze labels directory
        if labels_dir.exists():
            label_files = list(labels_dir.glob('*.txt'))
            split_info['labels'] = len(label_files)
            
            # Collect label file types
            for label_file in label_files[:10]:  # Sample first 10 files
                split_info['file_types']['labels'].add(label_file.suffix.lower())
                
            if split_info['labels'] == 0:
                split_info['issues'].append(f"⚠️ Split {split_name}: Folder labels kosong")
        else:
            split_info['issues'].append(f"❌ Split {split_name}: Folder labels tidak ditemukan")
        
        # Check image-label matching
        if split_info['images'] > 0 and split_info['labels'] > 0:
            mismatch = abs(split_info['images'] - split_info['labels'])
            if mismatch > 0:
                split_info['issues'].append(
                    f"⚠️ Split {split_name}: Mismatch gambar ({split_info['images']}) vs label ({split_info['labels']})"
                )
        
        # Convert sets to lists for JSON serialization
        split_info['file_types']['images'] = list(split_info['file_types']['images'])
        split_info['file_types']['labels'] = list(split_info['file_types']['labels'])
        
    except Exception as e:
        split_info['issues'].append(f"❌ Split {split_name}: Error analisis - {str(e)}")
    
    return split_info

def _analyze_downloads_contents(downloads_path: Path) -> List[Dict[str, Any]]:
    """Analyze contents dari downloads folder."""
    contents = []
    
    try:
        # List immediate subdirectories
        for item in downloads_path.iterdir():
            if item.is_dir():
                # Count files in subdirectory
                try:
                    file_count = len(list(item.rglob('*.*')))
                    folder_size_mb = 0
                    
                    # Calculate folder size
                    try:
                        folder_size = sum(f.stat().st_size for f in item.rglob('*') if f.is_file())
                        folder_size_mb = round(folder_size / (1024 * 1024), 2)
                    except Exception:
                        pass
                    
                    contents.append({
                        'name': item.name,
                        'type': 'directory',
                        'files': file_count,
                        'size_mb': folder_size_mb
                    })
                except Exception:
                    contents.append({
                        'name': item.name,
                        'type': 'directory',
                        'files': 0,
                        'size_mb': 0
                    })
                    
            elif item.is_file():
                # Individual file
                try:
                    file_size_mb = round(item.stat().st_size / (1024 * 1024), 2)
                    contents.append({
                        'name': item.name,
                        'type': 'file',
                        'size_mb': file_size_mb
                    })
                except Exception:
                    contents.append({
                        'name': item.name,
                        'type': 'file',
                        'size_mb': 0
                    })
                    
    except Exception:
        pass
    
    return contents

def _generate_dataset_summary(status_data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate summary dari status data."""
    final = status_data['final_dataset']
    downloads = status_data['downloads_folder']
    storage = status_data['storage_info']
    
    summary = {
        'status': 'unknown',
        'message': '',
        'recommendations': [],
        'statistics': {
            'total_images': final['total_images'],
            'total_labels': final['total_labels'],
            'active_splits': len([s for s in final['splits'].values() if s['exists']]),
            'downloads_files': downloads['total_files'],
            'storage_type': storage['type']
        }
    }
    
    # Determine overall status
    if final['exists'] and final['total_images'] > 0:
        summary['status'] = 'ready'
        summary['message'] = f"✅ Dataset siap: {final['total_images']} gambar dalam {summary['statistics']['active_splits']} split"
        
        # Add training readiness check
        train_split = final['splits'].get('train', {})
        if train_split.get('exists', False) and train_split.get('images', 0) > 0:
            summary['recommendations'].append("🚀 Dataset siap untuk training")
        else:
            summary['recommendations'].append("⚠️ Split training tidak ditemukan atau kosong")
            
    elif downloads['exists'] and downloads['total_files'] > 0:
        summary['status'] = 'downloaded'
        summary['message'] = f"📥 Dataset terdownload tapi belum diorganisir: {downloads['total_files']} file"
        summary['recommendations'].append("📁 Jalankan organisasi dataset untuk memindahkan ke struktur final")
        
    else:
        summary['status'] = 'empty'
        summary['message'] = "❌ Dataset tidak ditemukan"
        summary['recommendations'].append("📥 Download dataset terlebih dahulu")
    
    # Storage recommendations
    if not storage['persistent']:
        summary['recommendations'].append("💾 Hubungkan Google Drive untuk penyimpanan permanen")
    
    # Issue-based recommendations
    if final['issues']:
        summary['recommendations'].append("🔧 Perbaiki issues yang ditemukan dalam struktur dataset")
    
    return summary

def get_dataset_readiness_score(status_data: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate dataset readiness score untuk training."""
    final = status_data['final_dataset']
    
    score_breakdown = {
        'train_split': 0,
        'valid_split': 0,
        'label_matching': 0,
        'file_integrity': 0,
        'overall_score': 0
    }
    
    # Train split (40% weight)
    train_split = final['splits'].get('train', {})
    if train_split.get('exists', False) and train_split.get('images', 0) > 100:
        score_breakdown['train_split'] = 40
    elif train_split.get('images', 0) > 0:
        score_breakdown['train_split'] = 20
    
    # Valid split (30% weight)
    valid_split = final['splits'].get('valid', {})
    if valid_split.get('exists', False) and valid_split.get('images', 0) > 20:
        score_breakdown['valid_split'] = 30
    elif valid_split.get('images', 0) > 0:
        score_breakdown['valid_split'] = 15
    
    # Label matching (20% weight)
    total_mismatches = sum(len(split.get('issues', [])) for split in final['splits'].values())
    if total_mismatches == 0:
        score_breakdown['label_matching'] = 20
    elif total_mismatches < 3:
        score_breakdown['label_matching'] = 10
    
    # File integrity (10% weight)
    if final['total_images'] > 0 and final['total_labels'] > 0:
        score_breakdown['file_integrity'] = 10
    
    # Calculate overall score
    score_breakdown['overall_score'] = sum(score_breakdown.values()) - score_breakdown['overall_score']
    
    # Determine readiness level
    overall_score = score_breakdown['overall_score']
    if overall_score >= 80:
        readiness_level = "Siap Training"
        readiness_color = "success"
    elif overall_score >= 60:
        readiness_level = "Perlu Perbaikan Minor"
        readiness_color = "warning"
    elif overall_score >= 30:
        readiness_level = "Perlu Perbaikan Major"
        readiness_color = "warning"
    else:
        readiness_level = "Belum Siap"
        readiness_color = "danger"
    
    return {
        'score_breakdown': score_breakdown,
        'overall_score': overall_score,
        'readiness_level': readiness_level,
        'readiness_color': readiness_color,
        'max_score': 100
    }