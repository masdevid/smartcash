"""
File: smartcash/dataset/utils/progress_utils.py
Deskripsi: Utilitas untuk pelaporan progress dan notifikasi observer
"""

from typing import Dict, Any, Callable, Optional, List

def update_progress(
    callback: Callable, 
    current: int, 
    total: int, 
    message: str = None, 
    status: str = 'info', 
    **kwargs
) -> None:
    """
    Update progress dengan callback dan notifikasi observer.
    
    Args:
        callback: Callback function untuk progress reporting
        current: Nilai progress saat ini
        total: Nilai total progress
        message: Pesan progress
        status: Status progress ('info', 'success', 'warning', 'error')
        **kwargs: Parameter tambahan untuk callback
    """
    if 'total_files_all' not in kwargs: 
        kwargs['total_files_all'] = total
    
    # Call progress callback jika ada
    if callback:
        callback(
            progress=current, 
            total=total, 
            message=message or f"Preprocessing progress: {int(current/total*100) if total > 0 else 0}%", 
            status=status, 
            **kwargs
        )
    
    # Notifikasi observer jika tidak disertakan flag suppress_notify
    if not kwargs.get('suppress_notify', False):
        try:
            from smartcash.components.observer import notify
            from smartcash.components.observer.event_topics_observer import EventTopics
            
            # Bersihkan kwargs duplikasi
            notify_kwargs = {k: v for k, v in kwargs.items() if k not in ('current_progress', 'current_total')}
            
            notify(
                event_type=EventTopics.PREPROCESSING_PROGRESS, 
                sender="dataset_preprocessor", 
                message=message or f"Preprocessing progress: {int(current/total*100) if total > 0 else 0}%", 
                progress=current, 
                total=total, 
                **notify_kwargs
            )
        except Exception: 
            pass

def process_augmentation_results(results: List[Dict], logger=None) -> Dict[str, Any]:
    """
    Proses hasil augmentasi untuk statistik konsolidasian.
    
    Args:
        results: List hasil augmentasi per file
        logger: Logger untuk mencatat aktivitas
        
    Returns:
        Dictionary berisi statistik hasil augmentasi
    """
    try:
        # Statistik dasar dengan one-liner
        stats = {
            'successful': [r for r in results if r.get('status') == 'success'],
            'failed': [r for r in results if r.get('status') != 'success'],
            'total_generated': sum(r.get('generated', 0) for r in results),
            'class_stats': defaultdict(lambda: {'files': 0, 'generated': 0})
        }
        
        # Populate class stats dengan one-liner
        [stats['class_stats'][result.get('class_id', 'unknown')].update({
            'files': stats['class_stats'][result.get('class_id', 'unknown')]['files'] + 1,
            'generated': stats['class_stats'][result.get('class_id', 'unknown')]['generated'] + result.get('generated', 0),
            'denomination': get_denomination_label(result.get('class_id', 'unknown'))
        }) for result in stats['successful']]
        
        # Tambahan statistik dengan one-liner
        stats.update({
            'total_files': len(results),
            'success_rate': len(stats['successful']) / max(1, len(results)),
            'generated_per_file': stats['total_generated'] / max(1, len(stats['successful'])),
            'unique_classes': len(stats['class_stats'])
        })
        
        return stats
    except Exception as e:
        if logger: 
            logger.error(f"❌ Error saat memproses hasil augmentasi: {str(e)}")
        return {
            'total_files': len(results),
            'total_generated': sum(r.get('generated', 0) for r in results if isinstance(r, dict)),
            'error': str(e)
        }