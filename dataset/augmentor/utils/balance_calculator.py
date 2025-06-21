"""
File: smartcash/dataset/augmentor/utils/balance_calculator.py
Deskripsi: Calculator menggunakan refactored balancer module
"""
from typing import Dict, Any, List
from smartcash.dataset.augmentor.balancer import ClassBalancingStrategy, FileSelectionStrategy

class BalanceCalculator:
    """⚖️ Calculator dengan updated balancer import"""
    
    def __init__(self, config: Dict[str, Any] = None):
        from smartcash.dataset.augmentor.utils.config_validator import validate_augmentation_config, get_default_augmentation_config
        
        if config is None:
            self.config = get_default_augmentation_config()
        else:
            self.config = validate_augmentation_config(config)
        
        # Initialize dengan validated config
        self.balancer = ClassBalancingStrategy(self.config)
        self.selector = FileSelectionStrategy(self.config)
    
    def calculate_needs(self, source_files: List[str], target_count: int) -> Dict[str, int]:
        """Calculate needs menggunakan refactored balancer"""
        return self.balancer.calculate_balancing_needs_split_aware(
            self.config.get('data', {}).get('dir', 'data'),
            'train',
            target_count
        )
    
    def select_files_for_augmentation(self, source_files: List[str], 
                                    class_needs: Dict[str, int]) -> List[str]:
        """Select files menggunakan refactored selector"""
        return self.selector.select_prioritized_files_split_aware(
            self.config.get('data', {}).get('dir', 'data'),
            'train',
            class_needs
        )


def create_balance_calculator(config: Dict[str, Any]) -> BalanceCalculator:
    return BalanceCalculator(config)
