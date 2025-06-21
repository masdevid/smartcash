"""
File: smartcash/model/evaluation/scenario_manager.py
Deskripsi: Manager untuk research scenarios dengan position dan lighting variations
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import shutil

from smartcash.common.logger import get_logger
from smartcash.model.evaluation.scenario_augmentation import ScenarioAugmentation

class ScenarioManager:
    """Manager untuk research scenarios evaluation"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = get_logger('scenario_manager')
        self.scenario_augmentation = ScenarioAugmentation(config)
        
        # Setup directories
        data_config = self.config.get('evaluation', {}).get('data', {})
        self.test_dir = Path(data_config.get('test_dir', 'data/preprocessed/test'))
        self.evaluation_dir = Path(data_config.get('evaluation_dir', 'data/evaluation'))
        self.results_dir = Path(data_config.get('results_dir', 'data/evaluation/results'))
        
        self._ensure_directories()
    
    def setup_position_scenario(self) -> Dict[str, Any]:
        """📐 Setup position variation scenario"""
        scenario_config = self.config.get('evaluation', {}).get('scenarios', {}).get('position_variation', {})
        
        if not scenario_config.get('enabled', False):
            raise ValueError("❌ Position variation scenario tidak aktif")
        
        scenario_name = 'position_variation'
        output_dir = self.evaluation_dir / scenario_name
        
        self.logger.info(f"📐 Setting up {scenario_config.get('name', 'Position Variation')} scenario")
        
        # Generate scenario data
        result = self.scenario_augmentation.generate_position_variations(
            test_dir=str(self.test_dir),
            output_dir=str(output_dir),
            num_variations=scenario_config.get('augmentation_config', {}).get('num_variations', 5)
        )
        
        # Validate scenario data
        validation = self.scenario_augmentation.validate_scenario_data(str(output_dir))
        
        scenario_info = {
            'name': scenario_name,
            'display_name': scenario_config.get('name', 'Position Variation'),
            'enabled': True,
            'data_path': str(output_dir),
            'generation_result': result,
            'validation': validation,
            'config': scenario_config
        }
        
        if validation['valid']:
            self.logger.info(f"✅ Position scenario ready: {validation['images_count']} images")
        else:
            self.logger.warning(f"⚠️ Position scenario validation issues: {validation['issues']}")
        
        return scenario_info
    
    def setup_lighting_scenario(self) -> Dict[str, Any]:
        """💡 Setup lighting variation scenario"""
        scenario_config = self.config.get('evaluation', {}).get('scenarios', {}).get('lighting_variation', {})
        
        if not scenario_config.get('enabled', False):
            raise ValueError("❌ Lighting variation scenario tidak aktif")
        
        scenario_name = 'lighting_variation'
        output_dir = self.evaluation_dir / scenario_name
        
        self.logger.info(f"💡 Setting up {scenario_config.get('name', 'Lighting Variation')} scenario")
        
        # Generate scenario data
        result = self.scenario_augmentation.generate_lighting_variations(
            test_dir=str(self.test_dir),
            output_dir=str(output_dir),
            num_variations=scenario_config.get('augmentation_config', {}).get('num_variations', 5)
        )
        
        # Validate scenario data
        validation = self.scenario_augmentation.validate_scenario_data(str(output_dir))
        
        scenario_info = {
            'name': scenario_name,
            'display_name': scenario_config.get('name', 'Lighting Variation'),
            'enabled': True,
            'data_path': str(output_dir),
            'generation_result': result,
            'validation': validation,
            'config': scenario_config
        }
        
        if validation['valid']:
            self.logger.info(f"✅ Lighting scenario ready: {validation['images_count']} images")
        else:
            self.logger.warning(f"⚠️ Lighting scenario validation issues: {validation['issues']}")
        
        return scenario_info
    
    def generate_scenario_data(self, scenario_name: str, force_regenerate: bool = False) -> Dict[str, Any]:
        """🔄 Generate data untuk specific scenario"""
        scenario_dir = self.evaluation_dir / scenario_name
        
        # Check if data already exists
        if scenario_dir.exists() and not force_regenerate:
            validation = self.test_augmentation.validate_scenario_data(str(scenario_dir))
            if validation['valid']:
                self.logger.info(f"📁 Using existing {scenario_name} data")
                return {'status': 'existing', 'validation': validation, 'data_path': str(scenario_dir)}
        
        # Generate new data
        self.logger.info(f"🔄 Generating new {scenario_name} data")
        
        if scenario_name == 'position_variation':
            return self.setup_position_scenario()
        elif scenario_name == 'lighting_variation':
            return self.setup_lighting_scenario()
        else:
            raise ValueError(f"❌ Scenario tidak dikenal: {scenario_name}")
    
    def get_scenario_config(self, scenario_name: str) -> Dict[str, Any]:
        """⚙️ Get configuration untuk specific scenario"""
        scenarios_config = self.config.get('evaluation', {}).get('scenarios', {})
        
        if scenario_name not in scenarios_config:
            raise ValueError(f"❌ Scenario tidak ditemukan: {scenario_name}")
        
        return scenarios_config[scenario_name]
    
    def validate_scenario(self, scenario_name: str) -> Dict[str, Any]:
        """✅ Validate scenario setup dan data"""
        scenario_dir = self.evaluation_dir / scenario_name
        
        validation_result = {
            'scenario_name': scenario_name,
            'config_valid': False,
            'data_valid': False,
            'ready_for_evaluation': False,
            'issues': []
        }
        
        # Validate configuration
        try:
            scenario_config = self.get_scenario_config(scenario_name)
            if scenario_config.get('enabled', False):
                validation_result['config_valid'] = True
            else:
                validation_result['issues'].append(f"❌ Scenario {scenario_name} tidak aktif")
        except ValueError as e:
            validation_result['issues'].append(str(e))
        
        # Validate data
        if scenario_dir.exists():
            data_validation = self.test_augmentation.validate_scenario_data(str(scenario_dir))
            validation_result['data_valid'] = data_validation['valid']
            
            if not data_validation['valid']:
                validation_result['issues'].extend(data_validation['issues'])
        else:
            validation_result['issues'].append(f"❌ Data directory tidak ditemukan: {scenario_dir}")
        
        # Overall readiness
        validation_result['ready_for_evaluation'] = (
            validation_result['config_valid'] and 
            validation_result['data_valid']
        )
        
        return validation_result
    
    def cleanup_scenario(self, scenario_name: str) -> Dict[str, Any]:
        """🧹 Cleanup scenario data"""
        scenario_dir = self.evaluation_dir / scenario_name
        
        cleanup_result = {
            'scenario_name': scenario_name,
            'cleaned': False,
            'files_removed': 0,
            'size_freed_mb': 0
        }
        
        if scenario_dir.exists():
            # Calculate size before cleanup
            total_size = sum(f.stat().st_size for f in scenario_dir.rglob('*') if f.is_file())
            cleanup_result['size_freed_mb'] = round(total_size / (1024 * 1024), 2)
            
            # Count files
            file_count = len(list(scenario_dir.rglob('*')))
            cleanup_result['files_removed'] = file_count
            
            # Remove directory
            shutil.rmtree(scenario_dir)
            cleanup_result['cleaned'] = True
            
            self.logger.info(f"🧹 Cleaned {scenario_name}: {file_count} files, {cleanup_result['size_freed_mb']} MB")
        else:
            self.logger.info(f"📁 No data to clean for {scenario_name}")
        
        return cleanup_result
    
    def list_available_scenarios(self) -> List[Dict[str, Any]]:
        """📋 List semua scenario yang tersedia"""
        scenarios_config = self.config.get('evaluation', {}).get('scenarios', {})
        
        scenarios = []
        for scenario_name, scenario_config in scenarios_config.items():
            scenario_info = {
                'name': scenario_name,
                'display_name': scenario_config.get('name', scenario_name.replace('_', ' ').title()),
                'enabled': scenario_config.get('enabled', False),
                'description': self._get_scenario_description(scenario_name),
                'data_exists': (self.evaluation_dir / scenario_name).exists()
            }
            
            # Add validation if data exists
            if scenario_info['data_exists']:
                validation = self.validate_scenario(scenario_name)
                scenario_info.update({
                    'data_valid': validation['data_valid'],
                    'ready': validation['ready_for_evaluation']
                })
            else:
                scenario_info.update({
                    'data_valid': False,
                    'ready': False
                })
            
            scenarios.append(scenario_info)
        
        return scenarios
    
    def get_scenario_summary(self) -> Dict[str, Any]:
        """📊 Get summary semua scenarios"""
        scenarios = self.list_available_scenarios()
        
        summary = {
            'total_scenarios': len(scenarios),
            'enabled_scenarios': len([s for s in scenarios if s['enabled']]),
            'ready_scenarios': len([s for s in scenarios if s.get('ready', False)]),
            'scenarios_with_data': len([s for s in scenarios if s['data_exists']]),
            'scenarios': scenarios
        }
        
        return summary
    
    def prepare_all_scenarios(self, force_regenerate: bool = False) -> Dict[str, Any]:
        """🚀 Prepare semua enabled scenarios"""
        scenarios = self.list_available_scenarios()
        enabled_scenarios = [s for s in scenarios if s['enabled']]
        
        results = {
            'total_scenarios': len(enabled_scenarios),
            'successful': 0,
            'failed': 0,
            'results': {}
        }
        
        for scenario in enabled_scenarios:
            scenario_name = scenario['name']
            
            try:
                self.logger.info(f"🚀 Preparing {scenario['display_name']}")
                
                result = self.generate_scenario_data(scenario_name, force_regenerate)
                results['results'][scenario_name] = result
                results['successful'] += 1
                
            except Exception as e:
                self.logger.error(f"❌ Failed to prepare {scenario_name}: {str(e)}")
                results['results'][scenario_name] = {'status': 'failed', 'error': str(e)}
                results['failed'] += 1
        
        self.logger.info(f"✅ Scenario preparation complete: {results['successful']}/{results['total_scenarios']} successful")
        return results
    
    def _ensure_directories(self) -> None:
        """📁 Ensure evaluation directories exist"""
        directories = [self.evaluation_dir, self.results_dir]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _get_scenario_description(self, scenario_name: str) -> str:
        """📄 Get description untuk scenario"""
        descriptions = {
            'position_variation': 'Evaluasi performa deteksi dengan variasi posisi pengambilan gambar (rotasi, translasi, skala)',
            'lighting_variation': 'Evaluasi performa deteksi dengan variasi kondisi pencahayaan (brightness, contrast, gamma)'
        }
        
        return descriptions.get(scenario_name, f'Research scenario: {scenario_name}')


# Factory functions
def create_scenario_manager(config: Dict[str, Any] = None) -> ScenarioManager:
    """🏭 Factory untuk ScenarioManager"""
    return ScenarioManager(config)

def setup_research_scenarios(config: Dict[str, Any] = None, force_regenerate: bool = False) -> Dict[str, Any]:
    """🚀 One-liner untuk setup semua research scenarios"""
    manager = create_scenario_manager(config)
    return manager.prepare_all_scenarios(force_regenerate)

def get_scenarios_summary(config: Dict[str, Any] = None) -> Dict[str, Any]:
    """📊 One-liner untuk get scenarios summary"""
    return create_scenario_manager(config).get_scenario_summary()