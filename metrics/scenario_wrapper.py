# File: src/metrics/scenario_wrapper.py
# Author: Alfrida Sabar
# Deskripsi: Wrapper untuk kompatibilitas TestScenarioRunner

from .test_scenarios import EnhancedTestScenario
class TestScenarioRunner:
    """Wrapper untuk backward compatibility"""
    def __init__(self, model, data_loader, device='cuda'):
        self.enhanced = EnhancedTestScenario(model, data_loader, device)
        
    def run_scenario(self, config):
        """Map old interface ke enhanced features"""
        results = {}
        
        if config.conditions.get('orientation'):
            results.update(self.enhanced.evaluate_orientation())
            
        if config.conditions.get('degradation'):
            results.update(self.enhanced.evaluate_degradation())
            
        if config.conditions.get('lighting') == 'low':
            results.update(self.enhanced.evaluate_feature_preservation())
            
        return results