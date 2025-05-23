"""
File: smartcash/ui/dataset/preprocessing/handlers/check_dataset_handler.py
Deskripsi: Handler untuk check dataset dengan summary lokasi dan statistik data menggunakan explorer service
"""

from typing import Dict, Any
from pathlib import Path
from smartcash.ui.components.status_panel import update_status_panel


def setup_check_dataset_handler(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Setup handler untuk check dataset functionality dengan explorer integration."""
    logger = ui_components.get('logger')
    
    def _check_dataset_structure() -> Dict[str, Any]:
        """Check struktur dataset dan return summary dengan explorer service."""
        data_dir = Path(ui_components.get('data_dir', 'data'))
        preprocessed_dir = Path(ui_components.get('preprocessed_dir', 'data/preprocessed'))
        
        result = {
            'raw_exists': data_dir.exists(),
            'preprocessed_exists': preprocessed_dir.exists(),
            'raw_stats': {},
            'preprocessed_stats': {},
            'issues': [],
            'class_analysis': {},
            'total_summary': {}
        }
        
        # Check raw dataset dengan detail analysis
        if result['raw_exists']:
            total_images, total_labels = 0, 0
            
            for split in ['train', 'val', 'test']:
                split_dir = data_dir / split
                images_dir = split_dir / 'images'
                labels_dir = split_dir / 'labels'
                
                if split_dir.exists():
                    image_count = len(list(images_dir.glob('*.*'))) if images_dir.exists() else 0
                    label_count = len(list(labels_dir.glob('*.*'))) if labels_dir.exists() else 0
                    
                    result['raw_stats'][split] = {
                        'images': image_count,
                        'labels': label_count,
                        'complete': image_count > 0 and label_count > 0,
                        'images_path': str(images_dir),
                        'labels_path': str(labels_dir)
                    }
                    
                    total_images += image_count
                    total_labels += label_count
                    
                    # Check for issues
                    if image_count == 0:
                        result['issues'].append(f"❌ Split {split}: Tidak ada gambar")
                    if label_count == 0:
                        result['issues'].append(f"❌ Split {split}: Tidak ada label")
                    if image_count != label_count:
                        result['issues'].append(f"⚠️ Split {split}: Jumlah gambar ({image_count}) ≠ label ({label_count})")
                else:
                    result['issues'].append(f"❌ Split {split}: Direktori tidak ditemukan")
            
            result['total_summary']['raw'] = {'images': total_images, 'labels': total_labels}
            
            # Analisis kelas jika ada data valid
            if total_images > 0:
                result['class_analysis'] = _analyze_class_distribution(data_dir)
        else:
            result['issues'].append("❌ Direktori dataset raw tidak ditemukan")
        
        # Check preprocessed dataset
        if result['preprocessed_exists']:
            total_processed = 0
            for split in ['train', 'val', 'test']:
                split_dir = preprocessed_dir / split
                if split_dir.exists():
                    processed_count = len(list(split_dir.glob('**/*.jpg')))
                    result['preprocessed_stats'][split] = {
                        'processed': processed_count,
                        'path': str(split_dir)
                    }
                    total_processed += processed_count
            
            result['total_summary']['preprocessed'] = {'processed': total_processed}
        
        return result
    
    def _analyze_class_distribution(data_dir: Path) -> Dict[str, Any]:
        """Analisis distribusi kelas menggunakan explorer service."""
        try:
            from smartcash.dataset.services.explorer.explorer_service import ExplorerService
            
            # Setup config minimal untuk explorer
            config = {
                'data': {'dir': str(data_dir)},
                'classes': {}  # Akan dideteksi otomatis
            }
            
            explorer = ExplorerService(config, str(data_dir), logger)
            
            # Analisis split train (paling representatif)
            class_result = explorer.analyze_class_distribution('train')
            
            if class_result.get('status') == 'success':
                return {
                    'total_classes': class_result.get('class_count', 0),
                    'class_counts': class_result.get('counts', {}),
                    'imbalance_score': class_result.get('imbalance_score', 0)
                }
        except ImportError:
            if logger:
                logger.debug("🔍 Explorer service tidak tersedia untuk analisis kelas")
        except Exception as e:
            if logger:
                logger.debug(f"🔍 Error analisis kelas: {str(e)}")
        
        return {}
    
    def _format_dataset_summary(check_result: Dict[str, Any]) -> str:
        """Format hasil check menjadi summary log yang informatif."""
        lines = ["📊 Dataset Check Summary", "=" * 50]
        
        # Raw dataset info
        lines.append("\n📁 Raw Dataset:")
        if check_result['raw_exists']:
            raw_summary = check_result['total_summary'].get('raw', {})
            total_images = raw_summary.get('images', 0)
            total_labels = raw_summary.get('labels', 0)
            
            lines.append(f"   📍 Lokasi: {ui_components.get('data_dir', 'data')}")
            lines.append(f"   📊 Total: {total_images:,} gambar, {total_labels:,} label")
            
            # Per split breakdown
            for split, stats in check_result['raw_stats'].items():
                status = "✅" if stats['complete'] else "❌"
                percentage = (stats['images'] / max(total_images, 1)) * 100 if total_images > 0 else 0
                lines.append(f"   {status} {split}: {stats['images']:,} gambar ({percentage:.1f}%), {stats['labels']:,} label")
            
            # Class analysis jika tersedia
            class_analysis = check_result.get('class_analysis', {})
            if class_analysis:
                total_classes = class_analysis.get('total_classes', 0)
                imbalance_score = class_analysis.get('imbalance_score', 0)
                lines.append(f"   🏷️ Kelas: {total_classes} jenis")
                if imbalance_score > 5:
                    lines.append(f"   ⚠️ Ketidakseimbangan kelas: {imbalance_score:.1f}/10 (tinggi)")
                else:
                    lines.append(f"   ✅ Keseimbangan kelas: {imbalance_score:.1f}/10 (baik)")
        else:
            lines.append("   ❌ Tidak ditemukan")
        
        # Preprocessed dataset info
        lines.append("\n🔧 Preprocessed Dataset:")
        if check_result['preprocessed_exists']:
            lines.append(f"   📍 Lokasi: {ui_components.get('preprocessed_dir', 'data/preprocessed')}")
            
            if check_result['preprocessed_stats']:
                preprocessed_summary = check_result['total_summary'].get('preprocessed', {})
                total_processed = preprocessed_summary.get('processed', 0)
                lines.append(f"   📊 Total: {total_processed:,} gambar terproses")
                
                for split, stats in check_result['preprocessed_stats'].items():
                    count = stats['processed']
                    status = "✅" if count > 0 else "⚪"
                    percentage = (count / max(total_processed, 1)) * 100 if total_processed > 0 else 0
                    lines.append(f"   {status} {split}: {count:,} gambar ({percentage:.1f}%)")
            else:
                lines.append("   ⚪ Belum ada data preprocessed")
        else:
            lines.append("   ⚪ Direktori belum dibuat")
        
        # Issues section
        if check_result['issues']:
            lines.append(f"\n⚠️ Issues Ditemukan ({len(check_result['issues'])}):")
            for issue in check_result['issues']:
                lines.append(f"   • {issue}")
        else:
            lines.append("\n✅ Tidak ada issues ditemukan")
        
        # Smart recommendations
        lines.append("\n💡 Rekomendasi:")
        if not check_result['raw_exists']:
            lines.append("   • 📥 Download dataset terlebih dahulu")
        elif check_result['issues']:
            critical_issues = [i for i in check_result['issues'] if '❌' in i]
            if critical_issues:
                lines.append("   • 🔧 Perbaiki critical issues sebelum preprocessing")
            else:
                lines.append("   • ⚠️ Perhatikan warning issues, dataset masih bisa diproses")
        elif not any(stats.get('processed', 0) > 0 for stats in check_result['preprocessed_stats'].values()):
            lines.append("   • 🚀 Dataset siap untuk preprocessing")
        else:
            lines.append("   • 🔄 Dataset sudah dipreprocessing, gunakan force reprocess jika perlu update")
        
        # Performance recommendations
        raw_total = check_result['total_summary'].get('raw', {}).get('images', 0)
        if raw_total > 10000:
            recommended_workers = min(8, max(4, raw_total // 2500))
            lines.append(f"   • ⚡ Dataset besar ({raw_total:,} gambar), gunakan {recommended_workers} workers untuk performa optimal")
        
        return "\n".join(lines)
    
    def _on_check_dataset_click(b):
        """Handler untuk tombol check dataset."""
        if logger:
            logger.info("🔍 Memulai pengecekan dataset...")
        
        update_status_panel(ui_components['status_panel'], "Memeriksa struktur dan statistik dataset...", "info")
        
        try:
            # Perform comprehensive dataset check
            check_result = _check_dataset_structure()
            
            # Generate detailed summary
            summary = _format_dataset_summary(check_result)
            
            # Log summary to UI
            if logger:
                logger.info(summary)
            
            # Update status panel dengan smart status
            raw_total = check_result['total_summary'].get('raw', {}).get('images', 0)
            critical_issues = [i for i in check_result['issues'] if '❌' in i]
            
            if critical_issues:
                status_msg = f"Dataset ditemukan dengan {len(critical_issues)} critical issues"
                update_status_panel(ui_components['status_panel'], status_msg, "error")
            elif check_result['issues']:
                status_msg = f"Dataset valid dengan {len(check_result['issues'])} warnings"
                update_status_panel(ui_components['status_panel'], status_msg, "warning")
            elif check_result['raw_exists'] and raw_total > 0:
                status_msg = f"Dataset siap: {raw_total:,} gambar tersedia untuk preprocessing"
                update_status_panel(ui_components['status_panel'], status_msg, "success")
            else:
                update_status_panel(ui_components['status_panel'], "Dataset tidak ditemukan atau kosong", "error")
                
        except Exception as e:
            error_msg = f"Error saat check dataset: {str(e)}"
            if logger:
                logger.error(f"❌ {error_msg}")
            update_status_panel(ui_components['status_panel'], error_msg, "error")
    
    # Setup event handler
    ui_components['check_button'].on_click(_on_check_dataset_click)
    
    if logger:
        logger.debug("✅ Check dataset handler setup selesai")
    
    return ui_components