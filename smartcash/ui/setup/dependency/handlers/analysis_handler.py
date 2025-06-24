"""
File: smartcash/ui/setup/dependency/handlers/analysis_handler.py
Deskripsi: Handler untuk analisis packages dan compatibility checking
"""

from typing import Dict, Any, Callable, List
from concurrent.futures import ThreadPoolExecutor
from smartcash.ui.setup.dependency.utils import (
    get_selected_packages, batch_check_packages_status,
    update_status_panel, with_button_context, 
    show_progress_tracker_safe, complete_operation_with_message
)

def setup_analysis_handler(ui_components: Dict[str, Any]) -> Dict[str, Callable]:
    """Setup analysis handler untuk package analysis"""
    
    def handle_analysis():
        """Handle package analysis dengan batch processing"""
        logger = ui_components.get('logger')
        
        with with_button_context(ui_components, 'analyze_button'):
            try:
                # Extract packages untuk analysis
                selected_packages = get_selected_packages(ui_components.get('package_selector', {}))
                custom_packages = _get_custom_packages(ui_components)
                all_packages = selected_packages + custom_packages
                
                if not all_packages:
                    update_status_panel(ui_components, "⚠️ Tidak ada packages untuk dianalisis", "warning")
                    return
                
                # Get analysis settings
                config = _extract_analysis_config(ui_components)
                
                # Start analysis
                update_status_panel(ui_components, f"🔍 Menganalisis {len(all_packages)} packages...", "info")
                show_progress_tracker_safe(ui_components, "Package Analysis")
                
                if logger:
                    logger.info(f"🔍 Analyzing {len(all_packages)} packages...")
                
                # Analyze packages
                analysis_results = _analyze_packages_batch(all_packages, config, ui_components)
                
                # Generate report
                report = _generate_analysis_report(analysis_results)
                
                # Update UI dengan hasil
                _update_analysis_results(ui_components, report)
                
                complete_operation_with_message(ui_components, f"✅ Analisis selesai: {report['summary']}")
                
                if logger:
                    logger.info(f"📊 Analysis completed: {report['summary']}")
                
            except Exception as e:
                update_status_panel(ui_components, f"❌ Analysis error: {str(e)}", "error")
                if logger:
                    logger.error(f"❌ Analysis failed: {str(e)}")
    
    # Setup button handler
    analyze_button = ui_components.get('analyze_button')
    if analyze_button:
        analyze_button.on_click(lambda b: handle_analysis())
    
    return {
        'handle_analysis': handle_analysis,
        'analyze_packages_batch': lambda packages, config: _analyze_packages_batch(packages, config, ui_components)
    }

def _get_custom_packages(ui_components: Dict[str, Any]) -> List[str]:
    """Extract custom packages dari textarea"""
    try:
        widget = ui_components.get('custom_packages')
        if widget and widget.value.strip():
            return [pkg.strip() for pkg in widget.value.strip().split('\n') if pkg.strip()]
        return []
    except:
        return []

def _extract_analysis_config(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Extract analysis config dari UI"""
    from ..handlers.config_extractor import extract_dependency_config
    
    try:
        full_config = extract_dependency_config(ui_components)
        return full_config.get('analysis', {
            'check_compatibility': True,
            'batch_size': 10,
            'detailed_info': True
        })
    except:
        return {
            'check_compatibility': True,
            'batch_size': 10,
            'detailed_info': True
        }

def _analyze_packages_batch(packages: List[str], config: Dict[str, Any], ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze packages dalam batch dengan progress tracking"""
    from smartcash.ui.setup.dependency.utils import update_progress_step
    
    batch_size = config.get('batch_size', 10)
    results = {
        'installed': [],
        'not_installed': [],
        'errors': [],
        'compatibility_issues': []
    }
    
    # Process in batches
    for i in range(0, len(packages), batch_size):
        batch = packages[i:i + batch_size]
        
        # Update progress
        progress = int((i / len(packages)) * 100)
        update_progress_step(ui_components, "overall", progress, f"Analyzing batch {i//batch_size + 1}")
        
        # Check batch status
        batch_results = batch_check_packages_status(batch)
        
        # Categorize results
        for result in batch_results:
            if result['success']:
                if result['installed']:
                    results['installed'].append(result)
                else:
                    results['not_installed'].append(result)
            else:
                results['errors'].append(result)
        
        # Check compatibility jika enabled
        if config.get('check_compatibility', True):
            compatibility_issues = _check_compatibility_batch(batch, config)
            results['compatibility_issues'].extend(compatibility_issues)
    
    return results

def _check_compatibility_batch(packages: List[str], config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Check compatibility issues untuk batch packages"""
    issues = []
    
    # Simple compatibility checks
    for package in packages:
        try:
            # Check for common compatibility issues
            if 'tensorflow' in package.lower() and 'torch' in [p.lower() for p in packages]:
                issues.append({
                    'package': package,
                    'issue': 'Potential conflict with PyTorch',
                    'severity': 'warning'
                })
            
            # Check for version conflicts
            if '>=' in package and '<=' in package:
                issues.append({
                    'package': package,
                    'issue': 'Complex version constraints',
                    'severity': 'info'
                })
                
        except Exception:
            pass
    
    return issues

def _generate_analysis_report(results: Dict[str, Any]) -> Dict[str, Any]:
    """Generate comprehensive analysis report"""
    total_packages = len(results['installed']) + len(results['not_installed']) + len(results['errors'])
    
    return {
        'total_packages': total_packages,
        'installed_count': len(results['installed']),
        'not_installed_count': len(results['not_installed']),
        'error_count': len(results['errors']),
        'compatibility_issues_count': len(results['compatibility_issues']),
        'summary': f"{len(results['installed'])}/{total_packages} installed, {len(results['compatibility_issues'])} issues",
        'details': results
    }

def _update_analysis_results(ui_components: Dict[str, Any], report: Dict[str, Any]) -> None:
    """Update UI dengan analysis results"""
    try:
        logger = ui_components.get('logger')
        if logger:
            logger.info(f"📊 Analysis Report:")
            logger.info(f"   • Total packages: {report['total_packages']}")
            logger.info(f"   • Installed: {report['installed_count']}")
            logger.info(f"   • Not installed: {report['not_installed_count']}")
            logger.info(f"   • Errors: {report['error_count']}")
            logger.info(f"   • Compatibility issues: {report['compatibility_issues_count']}")
            
            # Log compatibility issues
            if report['details']['compatibility_issues']:
                logger.warning("⚠️ Compatibility Issues:")
                for issue in report['details']['compatibility_issues'][:3]:  # Show first 3
                    logger.warning(f"   • {issue['package']}: {issue['issue']}")
    except Exception:
        pass