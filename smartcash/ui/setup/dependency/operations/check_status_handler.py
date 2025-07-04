"""
File: smartcash/ui/setup/dependency/operations/check_status_handler.py
Deskripsi: Handler untuk check status operations
"""

import subprocess
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed

from smartcash.ui.core.handlers.operation_handler import OperationHandler

class CheckStatusOperationHandler(OperationHandler):
    """Handler untuk check package status operations"""
    
    def __init__(self, ui_components: Dict[str, Any], config: Dict[str, Any]):
        super().__init__('check_status', 'dependency.setup')
        self.ui_components = ui_components
        self.config = config
    
    def execute_operation(self) -> Dict[str, Any]:
        """Execute check status operation"""
        try:
            self._update_status("🔍 Memulai pengecekan status packages...")
            
            packages_to_check = self._get_packages_to_check()
            
            if not packages_to_check:
                self._update_status("⚠️ Tidak ada packages untuk dicek", "warning")
                return {'success': False, 'message': 'Tidak ada packages yang dipilih'}
            
            self._show_progress(f"Checking {len(packages_to_check)} packages", packages_to_check)
            
            # Check packages
            results = self._check_packages_status(packages_to_check)
            
            # Process results
            installed_count = sum(1 for r in results if r['installed'])
            not_installed_count = len(results) - installed_count
            
            status_summary = self._generate_status_summary(results)
            
            self._update_status(f"✅ Status check selesai: {installed_count} installed, {not_installed_count} not installed", "success")
            
            return {
                'success': True,
                'total': len(packages_to_check),
                'installed': installed_count,
                'not_installed': not_installed_count,
                'details': results,
                'summary': status_summary
            }
                
        except Exception as e:
            self.handle_error(f"Check status operation failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _get_packages_to_check(self) -> List[str]:
        """Get packages untuk dicek status"""
        packages = []
        
        # Get selected packages
        selected_packages = self.config.get('selected_packages', [])
        packages.extend(selected_packages)
        
        # Get custom packages
        custom_packages = self.config.get('custom_packages', '')
        if custom_packages:
            for line in custom_packages.split('\n'):
                line = line.strip()
                if line and not line.startswith('#'):
                    # Extract package name
                    pkg_name = line.split('==')[0].split('>=')[0].split('<=')[0].split('>')[0].split('<')[0]
                    packages.append(pkg_name.strip())
        
        return list(set(packages))  # Remove duplicates
    
    def _check_packages_status(self, packages: List[str]) -> List[Dict[str, Any]]:
        """Check status untuk semua packages"""
        results = []
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_package = {
                executor.submit(self._check_single_package_status, pkg): pkg 
                for pkg in packages
            }
            
            for future in as_completed(future_to_package):
                package = future_to_package[future]
                try:
                    result = future.result()
                    results.append(result)
                    self._update_progress_step(f"Checked: {package}")
                except Exception as e:
                    results.append({
                        'package': package,
                        'installed': False,
                        'version': None,
                        'error': str(e)
                    })
        
        return results
    
    def _check_single_package_status(self, package: str) -> Dict[str, Any]:
        """Check status untuk single package"""
        try:
            result = subprocess.run(
                ['pip', 'show', package],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                # Parse output untuk get info
                info = self._parse_package_info(result.stdout)
                
                return {
                    'package': package,
                    'installed': True,
                    'version': info.get('version'),
                    'location': info.get('location'),
                    'requires': info.get('requires'),
                    'required_by': info.get('required_by'),
                    'size': self._get_package_size(package),
                    'status': 'installed'
                }
            else:
                return {
                    'package': package,
                    'installed': False,
                    'version': None,
                    'status': 'not_installed'
                }
                
        except Exception as e:
            return {
                'package': package,
                'installed': False,
                'version': None,
                'error': str(e),
                'status': 'error'
            }
    
    def _parse_package_info(self, output: str) -> Dict[str, Any]:
        """Parse pip show output"""
        info = {}
        
        for line in output.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower()
                value = value.strip()
                
                if key == 'version':
                    info['version'] = value
                elif key == 'location':
                    info['location'] = value
                elif key == 'requires':
                    info['requires'] = value.split(', ') if value else []
                elif key == 'required-by':
                    info['required_by'] = value.split(', ') if value else []
        
        return info
    
    def _get_package_size(self, package: str) -> str:
        """Get approximate package size"""
        try:
            result = subprocess.run(
                ['pip', 'show', '-f', package],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                file_count = result.stdout.count('\n')
                # Rough size estimation
                if file_count < 50:
                    return "< 1MB"
                elif file_count < 200:
                    return "1-5MB"
                elif file_count < 500:
                    return "5-20MB"
                else:
                    return "> 20MB"
            
            return "Unknown"
            
        except Exception:
            return "Unknown"
    
    def _generate_status_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary dari status check"""
        summary = {
            'total_packages': len(results),
            'installed_packages': [],
            'not_installed_packages': [],
            'packages_with_errors': [],
            'categories': {}
        }
        
        for result in results:
            package = result['package']
            
            if result.get('error'):
                summary['packages_with_errors'].append(package)
            elif result['installed']:
                summary['installed_packages'].append({
                    'name': package,
                    'version': result.get('version'),
                    'size': result.get('size')
                })
            else:
                summary['not_installed_packages'].append(package)
            
            # Categorize by category
            category = self._get_package_category(package)
            if category not in summary['categories']:
                summary['categories'][category] = {'installed': 0, 'not_installed': 0}
            
            if result['installed']:
                summary['categories'][category]['installed'] += 1
            else:
                summary['categories'][category]['not_installed'] += 1
        
        return summary
    
    def _get_package_category(self, package: str) -> str:
        """Get category dari package"""
        from ..configs.dependency_defaults import get_default_package_categories
        
        categories = get_default_package_categories()
        
        for category_key, category_info in categories.items():
            for pkg in category_info.get('packages', []):
                if pkg['name'] == package:
                    return category_info['name']
        
        return 'Custom'
    
    def _update_status(self, message: str, status_type: str = "info") -> None:
        """Update status panel"""
        if 'status_panel' in self.ui_components:
            status_panel = self.ui_components['status_panel']
            if hasattr(status_panel, 'update_status'):
                status_panel.update_status(message, status_type)
    
    def _show_progress(self, title: str, steps: List[str]) -> None:
        """Show progress tracker"""
        if 'progress_tracker' in self.ui_components:
            progress_tracker = self.ui_components['progress_tracker']
            if hasattr(progress_tracker, 'show'):
                progress_tracker.show(title, steps)
    
    def _update_progress_step(self, message: str) -> None:
        """Update progress step"""
        if 'progress_tracker' in self.ui_components:
            progress_tracker = self.ui_components['progress_tracker']
            if hasattr(progress_tracker, 'update_step'):
                progress_tracker.update_step(message)