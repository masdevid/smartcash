"""
File: smartcash/dataset/utils/debug_helper.py
Deskripsi: Debug helper untuk troubleshoot "Tidak ada file gambar ditemukan" dengan comprehensive analysis
"""

import os
import glob
from pathlib import Path
from typing import Dict, Any, List
from smartcash.common.logger import get_logger

def debug_image_detection(data_dir: str, logger=None) -> Dict[str, Any]:
    """
    Comprehensive debug untuk image detection issues.
    
    Args:
        data_dir: Directory yang akan di-debug
        logger: Logger untuk output
        
    Returns:
        Debug information dictionary
    """
    logger = logger or get_logger(__name__)
    
    logger.info(f"🔍 Debug Image Detection untuk: {data_dir}")
    
    debug_info = {
        'input_path': data_dir,
        'resolved_paths': [],
        'directory_structure': {},
        'file_counts': {},
        'sample_files': {},
        'recommendations': []
    }
    
    # 1. Test path resolution
    logger.info("📍 Testing path resolution...")
    search_bases = [
        '/content/drive/MyDrive/SmartCash',
        '/content/drive/MyDrive', 
        '/content/SmartCash',
        '/content',
        os.getcwd()
    ]
    
    for base in search_bases:
        test_path = os.path.join(base, data_dir) if not os.path.isabs(data_dir) else data_dir
        exists = os.path.exists(test_path)
        is_dir = os.path.isdir(test_path) if exists else False
        
        debug_info['resolved_paths'].append({
            'base': base,
            'full_path': test_path,
            'exists': exists,
            'is_directory': is_dir
        })
        
        logger.info(f"   {'✅' if exists else '❌'} {test_path}")
        
        if exists and is_dir:
            debug_info['primary_path'] = test_path
            break
    
    # 2. Analyze directory structure
    primary_path = debug_info.get('primary_path')
    if primary_path:
        logger.info(f"📁 Analyzing directory structure: {primary_path}")
        debug_info['directory_structure'] = _analyze_directory_structure(primary_path, logger)
    else:
        logger.error("❌ Tidak ada path yang valid ditemukan!")
        debug_info['recommendations'].append("Path tidak ditemukan - periksa path input")
        return debug_info
    
    # 3. Count files by type and location
    logger.info("📊 Counting files by type...")
    debug_info['file_counts'] = _count_files_by_type(primary_path, logger)
    
    # 4. Find sample files
    logger.info("📋 Finding sample files...")
    debug_info['sample_files'] = _find_sample_files(primary_path, logger)
    
    # 5. Generate recommendations
    debug_info['recommendations'] = _generate_recommendations(debug_info, logger)
    
    # 6. Summary
    total_images = sum(debug_info['file_counts'].get('images', {}).values())
    total_labels = sum(debug_info['file_counts'].get('labels', {}).values())
    
    logger.info(f"📋 Debug Summary:")
    logger.info(f"   • Total images found: {total_images}")
    logger.info(f"   • Total labels found: {total_labels}")
    logger.info(f"   • Directories scanned: {len(debug_info['directory_structure'])}")
    logger.info(f"   • Recommendations: {len(debug_info['recommendations'])}")
    
    return debug_info

def _analyze_directory_structure(path: str, logger) -> Dict[str, Any]:
    """Analyze directory structure untuk debugging"""
    structure = {}
    
    try:
        for root, dirs, files in os.walk(path):
            rel_path = os.path.relpath(root, path)
            if rel_path == '.':
                rel_path = 'root'
            
            structure[rel_path] = {
                'subdirectories': dirs,
                'file_count': len(files),
                'has_images': any(Path(f).suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp'] for f in files),
                'has_labels': any(f.endswith('.txt') for f in files),
                'sample_files': files[:5] if files else []
            }
            
            if len(files) > 0:
                logger.info(f"   📂 {rel_path}: {len(files)} files ({'images' if structure[rel_path]['has_images'] else 'no images'})")
    
    except Exception as e:
        logger.error(f"❌ Error analyzing structure: {str(e)}")
        structure['error'] = str(e)
    
    return structure

def _count_files_by_type(path: str, logger) -> Dict[str, Any]:
    """Count files by type dan location"""
    counts = {
        'images': {},
        'labels': {},
        'total_files': 0
    }
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    
    try:
        for root, dirs, files in os.walk(path):
            rel_path = os.path.relpath(root, path)
            if rel_path == '.':
                rel_path = 'root'
            
            image_count = sum(1 for f in files if Path(f).suffix.lower() in image_extensions)
            label_count = sum(1 for f in files if f.endswith('.txt'))
            
            if image_count > 0:
                counts['images'][rel_path] = image_count
            if label_count > 0:
                counts['labels'][rel_path] = label_count
            
            counts['total_files'] += len(files)
    
    except Exception as e:
        logger.error(f"❌ Error counting files: {str(e)}")
        counts['error'] = str(e)
    
    return counts

def _find_sample_files(path: str, logger) -> Dict[str, List[str]]:
    """Find sample files untuk inspection"""
    samples = {
        'images': [],
        'labels': [],
        'other': []
    }
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    
    try:
        # Use glob untuk comprehensive search
        for ext in image_extensions:
            pattern = os.path.join(path, '**', f'*{ext}')
            found_files = glob.glob(pattern, recursive=True)
            samples['images'].extend(found_files[:10])  # Max 10 samples
        
        # Find labels
        label_pattern = os.path.join(path, '**', '*.txt')
        found_labels = glob.glob(label_pattern, recursive=True)
        samples['labels'].extend(found_labels[:10])
        
        # Find other files
        all_pattern = os.path.join(path, '**', '*.*')
        all_files = glob.glob(all_pattern, recursive=True)
        other_files = [f for f in all_files if not any(f.endswith(ext) for ext in image_extensions + ['.txt'])]
        samples['other'].extend(other_files[:5])
        
    except Exception as e:
        logger.error(f"❌ Error finding samples: {str(e)}")
        samples['error'] = str(e)
    
    return samples

def _generate_recommendations(debug_info: Dict[str, Any], logger) -> List[str]:
    """Generate actionable recommendations"""
    recommendations = []
    
    total_images = sum(debug_info['file_counts'].get('images', {}).values())
    total_labels = sum(debug_info['file_counts'].get('labels', {}).values())
    
    # Path recommendations
    if not debug_info.get('primary_path'):
        recommendations.append("❌ Path tidak ditemukan - pastikan data sudah diupload ke Google Drive")
        recommendations.append("💡 Coba: upload ke /content/drive/MyDrive/SmartCash/data/")
        return recommendations
    
    # Image recommendations
    if total_images == 0:
        recommendations.append("❌ Tidak ada gambar ditemukan")
        recommendations.append("💡 Pastikan file gambar berformat: .jpg, .jpeg, .png, .bmp")
        recommendations.append("💡 Cek apakah ada di subdirectory 'images' atau 'train/images'")
    elif total_images < 10:
        recommendations.append(f"⚠️ Hanya {total_images} gambar ditemukan - mungkin terlalu sedikit")
    else:
        recommendations.append(f"✅ Ditemukan {total_images} gambar")
    
    # Label recommendations  
    if total_labels == 0:
        recommendations.append("⚠️ Tidak ada file label (.txt) ditemukan")
        recommendations.append("💡 Augmentasi akan berjalan tanpa bounding boxes")
    elif total_labels < total_images:
        recommendations.append(f"⚠️ Label tidak lengkap: {total_labels} label vs {total_images} gambar")
    else:
        recommendations.append(f"✅ Ditemukan {total_labels} file label")
    
    # Structure recommendations
    structure = debug_info.get('directory_structure', {})
    has_yolo_structure = any('images' in dirs.get('subdirectories', []) for dirs in structure.values())
    
    if has_yolo_structure:
        recommendations.append("✅ Struktur YOLO terdeteksi (images/labels directory)")
    else:
        recommendations.append("ℹ️ Struktur flat terdeteksi - akan dicari secara recursive")
    
    # Sample file recommendations
    sample_images = debug_info.get('sample_files', {}).get('images', [])
    if sample_images:
        recommendations.append(f"🔍 Sample image pertama: {Path(sample_images[0]).name}")
    
    return recommendations

def quick_debug(data_dir: str = 'data'):
    """Quick debug function untuk Colab cell"""
    from IPython.display import display, HTML
    
    logger = get_logger("debug")
    debug_info = debug_image_detection(data_dir, logger)
    
    # Create HTML summary
    html_content = f"""
    <div style="padding: 15px; border: 1px solid #ddd; border-radius: 5px; font-family: monospace;">
        <h3>🔍 Debug Image Detection Results</h3>
        <p><strong>Input Path:</strong> {debug_info['input_path']}</p>
        <p><strong>Primary Path:</strong> {debug_info.get('primary_path', 'Not found')}</p>
        
        <h4>📊 File Counts:</h4>
        <ul>
            <li>Images: {sum(debug_info['file_counts'].get('images', {}).values())}</li>
            <li>Labels: {sum(debug_info['file_counts'].get('labels', {}).values())}</li>
            <li>Total Files: {debug_info['file_counts'].get('total_files', 0)}</li>
        </ul>
        
        <h4>💡 Recommendations:</h4>
        <ul>
    """
    
    for rec in debug_info['recommendations']:
        html_content += f"<li>{rec}</li>"
    
    html_content += """
        </ul>
    </div>
    """
    
    display(HTML(html_content))
    return debug_info

# One-liner utilities
debug_path = lambda path: debug_image_detection(path)
check_images = lambda path='data': len(glob.glob(os.path.join(path, '**', '*.jpg'), recursive=True))
list_directories = lambda path='data': [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))] if os.path.exists(path) else []