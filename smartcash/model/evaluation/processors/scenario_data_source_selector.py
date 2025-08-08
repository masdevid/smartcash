"""
Scenario data source selector for evaluation.
Maps scenarios to specific data directories without augmentation.
"""

from pathlib import Path
from typing import Dict, Any
from smartcash.common.logger import get_logger


class ScenarioDataSourceSelector:
    """Select appropriate data sources for different evaluation scenarios"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.logger = get_logger('scenario_data_source')
        self.config = config or {}
        
        # Default scenario to data directory mapping  
        self.scenario_data_mapping = {
            'position_variation': 'data/test',
            'lighting_variation': 'data/test_lighting'
        }
    
    def get_scenario_data_directory(self, scenario_name: str) -> str:
        """Get the data directory for a specific scenario"""
        
        # Check if scenario is configured with custom data path
        scenario_config = self.config.get('evaluation', {}).get('scenarios', {}).get(scenario_name, {})
        custom_data_dir = scenario_config.get('data_dir')
        
        if custom_data_dir:
            data_dir = custom_data_dir
            self.logger.info(f"📁 Using custom data directory for {scenario_name}: {data_dir}")
        else:
            # Use default mapping
            data_dir = self.scenario_data_mapping.get(scenario_name, 'data/test')
            self.logger.info(f"📁 Using default data directory for {scenario_name}: {data_dir}")
        
        return data_dir
    
    def validate_scenario_data_directory(self, scenario_name: str, data_dir: str) -> Dict[str, Any]:
        """Validate that scenario data directory exists and contains required files"""
        
        data_path = Path(data_dir)
        validation_result = {
            'valid': False,
            'data_dir': str(data_path),
            'scenario_name': scenario_name,
            'images_count': 0,
            'labels_count': 0,
            'missing_labels': [],
            'issues': []
        }
        
        # Check if directory exists
        if not data_path.exists():
            validation_result['issues'].append(f"❌ Data directory does not exist: {data_dir}")
            return validation_result
        
        # Check for images and labels subdirectories
        images_dir = data_path / 'images'
        labels_dir = data_path / 'labels'
        
        if not images_dir.exists():
            validation_result['issues'].append(f"❌ Images directory not found: {images_dir}")
            return validation_result
            
        if not labels_dir.exists():
            validation_result['issues'].append(f"❌ Labels directory not found: {labels_dir}")
            return validation_result
        
        # Count files
        image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
        label_files = list(labels_dir.glob('*.txt'))
        
        validation_result['images_count'] = len(image_files)
        validation_result['labels_count'] = len(label_files)
        
        # Check for matching labels
        for img_file in image_files:
            label_file = labels_dir / f"{img_file.stem}.txt"
            if not label_file.exists():
                validation_result['missing_labels'].append(img_file.name)
        
        # Determine validity
        has_data = validation_result['images_count'] > 0 and validation_result['labels_count'] > 0
        has_matching_labels = len(validation_result['missing_labels']) == 0
        
        validation_result['valid'] = has_data and has_matching_labels
        
        if validation_result['valid']:
            self.logger.info(f"✅ Scenario data valid for {scenario_name}: {validation_result['images_count']} images")
        else:
            self.logger.warning(f"⚠️ Validation issues for {scenario_name}: {validation_result['issues']}")
            if validation_result['missing_labels']:
                self.logger.warning(f"⚠️ Missing labels for {len(validation_result['missing_labels'])} images")
        
        return validation_result
    
    def get_all_scenario_data_sources(self) -> Dict[str, Dict[str, Any]]:
        """Get data sources for all configured scenarios"""
        
        all_sources = {}
        
        for scenario_name in self.scenario_data_mapping.keys():
            data_dir = self.get_scenario_data_directory(scenario_name)
            validation = self.validate_scenario_data_directory(scenario_name, data_dir)
            
            all_sources[scenario_name] = {
                'data_dir': data_dir,
                'validation': validation,
                'status': 'valid' if validation['valid'] else 'invalid'
            }
        
        return all_sources
    
    def setup_scenario_data_directories(self) -> Dict[str, Any]:
        """Setup and validate all scenario data directories"""
        
        self.logger.info("🔍 Setting up scenario data directories...")
        
        all_sources = self.get_all_scenario_data_sources()
        setup_result = {
            'status': 'success',
            'scenarios_configured': len(all_sources),
            'scenarios_valid': 0,
            'scenarios_invalid': 0,
            'sources': all_sources,
            'issues': []
        }
        
        for scenario_name, source_info in all_sources.items():
            if source_info['status'] == 'valid':
                setup_result['scenarios_valid'] += 1
                self.logger.info(f"✅ {scenario_name}: {source_info['validation']['images_count']} images ready")
            else:
                setup_result['scenarios_invalid'] += 1
                setup_result['issues'].extend(source_info['validation']['issues'])
                self.logger.warning(f"⚠️ {scenario_name}: Invalid data source")
        
        if setup_result['scenarios_invalid'] > 0:
            self.logger.warning(f"⚠️ Setup completed with {setup_result['scenarios_invalid']} invalid scenarios")
        else:
            self.logger.info(f"✅ All {setup_result['scenarios_valid']} scenarios have valid data sources")
        
        return setup_result
    
    def get_scenario_data_info(self, scenario_name: str) -> Dict[str, Any]:
        """Get detailed information about a scenario's data source"""
        
        data_dir = self.get_scenario_data_directory(scenario_name)
        validation = self.validate_scenario_data_directory(scenario_name, data_dir)
        
        # Additional metadata
        data_path = Path(data_dir)
        
        info = {
            'scenario_name': scenario_name,
            'data_dir': data_dir,
            'data_path_exists': data_path.exists(),
            'validation': validation,
            'metadata': {
                'absolute_path': str(data_path.resolve()),
                'is_custom': scenario_name not in self.scenario_data_mapping or 
                            data_dir != self.scenario_data_mapping[scenario_name]
            }
        }
        
        return info


# Factory functions
def create_scenario_data_source_selector(config: Dict[str, Any] = None) -> ScenarioDataSourceSelector:
    """Factory function to create scenario data source selector"""
    return ScenarioDataSourceSelector(config)


def get_scenario_data_directory(scenario_name: str, config: Dict[str, Any] = None) -> str:
    """Quick function to get data directory for a scenario"""
    selector = create_scenario_data_source_selector(config)
    return selector.get_scenario_data_directory(scenario_name)


def validate_all_scenario_data_sources(config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Quick function to validate all scenario data sources"""
    selector = create_scenario_data_source_selector(config)
    return selector.setup_scenario_data_directories()