"""
Visualization Module for SmartCash

This module provides visualization components for SmartCash dataset analysis.
"""

import pandas as pd
from typing import Optional, Dict, Any
from pathlib import Path

from .visualization_components import VisualizationUI, create_visualization_ui

def initialize_visualization_ui(
    data: Optional[Any] = None,
    data_path: Optional[str] = None,
    title: str = "Data Visualization",
    **kwargs
) -> VisualizationUI:
    """Initialize visualization UI with flexible data input options.
    
    Args:
        data: Optional pandas DataFrame or dictionary containing the data
        data_path: Optional path to a CSV or JSON file containing the data
        title: Title for the visualization
        **kwargs: Additional arguments to pass to pandas read_csv or read_json
        
    Returns:
        Initialized VisualizationUI instance
        
    Example:
        # From DataFrame
        viz = initialize_visualization_ui(data=df)
        
        # From file
        viz = initialize_visualization_ui(data_path='data.csv')
    """
    if data is None and data_path is None:
        # Create empty visualization with instructions
        return create_visualization_ui(
            pd.DataFrame({'Instruction': ['Please load some data to visualize']}),
            title=title
        )
    
    if data is not None:
        if isinstance(data, dict):
            df = pd.DataFrame(data)
        elif isinstance(data, pd.DataFrame):
            df = data
        else:
            raise ValueError("Data must be a pandas DataFrame or dictionary")
    else:
        # Load from file
        path = Path(data_path)
        if path.suffix == '.csv':
            df = pd.read_csv(path, **kwargs)
        elif path.suffix == '.json':
            df = pd.read_json(path, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
    
    return create_visualization_ui(df, title=title)

__all__ = [
    'VisualizationUI', 
    'create_visualization_ui',
    'initialize_visualization_ui'
]
