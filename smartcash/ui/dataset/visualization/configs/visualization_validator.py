"""
File: smartcash/ui/dataset/visualization/configs/visualization_validator.py
Description: Configuration validation for the visualization module
"""

def validate_visualization_config(config: Dict[str, Any]) -> tuple[bool, str]:
    """Validate visualization configuration.
    
    Args:
        config: Configuration to validate
        
    Returns:
        tuple: (is_valid, error_message)
    """
    if not isinstance(config, dict):
        return False, "Configuration must be a dictionary"
    
    # Validate required fields
    required_fields = ['splits', 'colors']
    for field in required_fields:
        if field not in config:
            return False, f"Missing required field: {field}"
    
    # Validate splits
    if not isinstance(config.get('splits'), list):
        return False, "'splits' must be a list"
    
    # Validate colors
    if not isinstance(config.get('colors'), dict):
        return False, "'colors' must be a dictionary"
    
    return True, ""
