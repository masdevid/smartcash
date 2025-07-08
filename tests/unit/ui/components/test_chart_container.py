#!/usr/bin/env python3
"""
Unit tests for chart container component.
"""

import pytest
from unittest.mock import Mock, patch

from smartcash.ui.components.chart_container import ChartContainer, create_chart_container


class TestChartContainer:
    """Test chart container component."""
    
    def test_chart_container_creation(self):
        """Test creating chart container."""
        chart = ChartContainer(
            component_name="test_chart",
            title="Test Chart",
            chart_type="line",
            columns=1,
            height=300
        )
        
        assert chart._title == "Test Chart"
        assert chart._chart_type == "line"
        assert chart._columns == 1
        assert chart._height == 300
    
    def test_chart_types_available(self):
        """Test that chart types are properly defined."""
        chart = ChartContainer()
        
        expected_types = {"line", "bar", "area"}
        available_types = set(chart.CHART_TYPES.keys())
        
        assert expected_types.issubset(available_types)
    
    def test_column_limit(self):
        """Test that columns are limited to 1 or 2."""
        # Test column limiting
        chart1 = ChartContainer(columns=0)
        assert chart1._columns == 1
        
        chart2 = ChartContainer(columns=5)
        assert chart2._columns == 2
        
        chart3 = ChartContainer(columns=2)
        assert chart3._columns == 2
    
    def test_single_column_layout(self):
        """Test single column chart layout."""
        chart = ChartContainer(columns=1)
        chart.initialize()
        
        assert 'chart_1' in chart._ui_components
        assert 'chart_2' not in chart._ui_components
    
    def test_two_column_layout(self):
        """Test two column chart layout."""
        chart = ChartContainer(columns=2)
        chart.initialize()
        
        assert 'chart_1' in chart._ui_components
        assert 'chart_2' in chart._ui_components
    
    def test_chart_data_update(self):
        """Test updating chart data."""
        chart = ChartContainer(columns=2)
        chart.initialize()
        
        test_data = [0.5, 0.4, 0.3, 0.2]
        config = {"title": "Test Loss", "color": "#ff6b6b"}
        
        # Test that update doesn't raise error
        chart.update_chart("chart_1", test_data, config)
        
        # Check that data is stored
        assert chart._chart_data.get("chart_1") == test_data
        assert chart._chart_configs.get("chart_1", {}).get("title") == "Test Loss"
    
    def test_chart_config_setting(self):
        """Test setting chart configuration."""
        chart = ChartContainer()
        chart.initialize()
        
        config = {"type": "bar", "color": "#4ecdc4"}
        chart.set_chart_config("chart_1", config)
        
        assert chart._chart_configs["chart_1"] == config
    
    def test_clear_charts(self):
        """Test clearing chart data."""
        chart = ChartContainer(columns=2)
        chart.initialize()
        
        # Add some data
        chart.update_chart("chart_1", [1, 2, 3], {"title": "Test"})
        chart.update_chart("chart_2", [4, 5, 6], {"title": "Test2"})
        
        # Clear charts
        chart.clear_charts()
        
        assert len(chart._chart_data) == 0
        assert len(chart._chart_configs) == 0
    
    def test_multiple_chart_update(self):
        """Test updating multiple charts at once."""
        chart = ChartContainer(columns=2)
        chart.initialize()
        
        chart_data = {
            "chart_1": {
                "data": [0.5, 0.4, 0.3],
                "config": {"title": "Loss", "color": "#ff6b6b"}
            },
            "chart_2": {
                "data": [0.3, 0.5, 0.7],
                "config": {"title": "mAP", "color": "#4ecdc4"}
            }
        }
        
        chart.update_charts(chart_data)
        
        assert chart._chart_data["chart_1"] == [0.5, 0.4, 0.3]
        assert chart._chart_data["chart_2"] == [0.3, 0.5, 0.7]
    
    def test_chart_type_change(self):
        """Test chart type selection change."""
        chart = ChartContainer(chart_type="line")
        chart.initialize()
        
        # Simulate chart type change
        change = {'new': ('bar', 'Bar Chart')}
        chart._on_chart_type_change(change)
        
        assert chart._chart_type == "bar"


class TestChartContainerFactory:
    """Test chart container factory function."""
    
    def test_create_chart_container(self):
        """Test factory function creates valid chart container."""
        chart = create_chart_container(
            title="Training Metrics",
            chart_type="line",
            columns=2,
            height=400
        )
        
        assert isinstance(chart, ChartContainer)
        assert chart._title == "Training Metrics"
        assert chart._chart_type == "line"
        assert chart._columns == 2
        assert chart._height == 400
    
    def test_default_parameters(self):
        """Test factory function with default parameters."""
        chart = create_chart_container()
        
        assert isinstance(chart, ChartContainer)
        assert chart._title == "Metrics Chart"
        assert chart._chart_type == "line"
        assert chart._columns == 1
        assert chart._height == 400
    
    def test_component_name_uniqueness(self):
        """Test that created charts have unique component names."""
        chart1 = create_chart_container()
        chart2 = create_chart_container()
        
        assert chart1.component_name != chart2.component_name


if __name__ == "__main__":
    pytest.main([__file__, "-v"])