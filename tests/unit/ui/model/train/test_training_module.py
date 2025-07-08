#!/usr/bin/env python3
"""
Comprehensive unit tests for training module structure and components.
"""

import pytest
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from smartcash.ui.model.train.constants import (
    TrainingOperation, TrainingPhase, DEFAULT_CONFIG, UI_CONFIG
)
from smartcash.ui.model.train.services.training_service import TrainingService
from smartcash.ui.model.train.operations.start_training_operation import StartTrainingOperation
from smartcash.ui.model.train.operations.stop_training_operation import StopTrainingOperation
from smartcash.ui.model.train.operations.resume_training_operation import ResumeTrainingOperation
from smartcash.ui.model.train.handlers.training_ui_handler import TrainingUIHandler
from smartcash.ui.model.train.components.training_ui import create_training_ui
from smartcash.ui.model.train.training_initializer import TrainingInitializer, get_training_initializer


class TestTrainingModuleStructure:
    """Test training module structure and imports."""
    
    def test_constants_import(self):
        """Test that all constants are properly defined."""
        # Test enums
        assert hasattr(TrainingOperation, 'START')
        assert hasattr(TrainingOperation, 'STOP') 
        assert hasattr(TrainingOperation, 'RESUME')
        
        assert hasattr(TrainingPhase, 'IDLE')
        assert hasattr(TrainingPhase, 'TRAINING')
        assert hasattr(TrainingPhase, 'COMPLETED')
        
        # Test configurations
        assert 'training' in DEFAULT_CONFIG
        assert 'optimizer' in DEFAULT_CONFIG
        assert 'scheduler' in DEFAULT_CONFIG
        
        assert 'title' in UI_CONFIG
        assert 'subtitle' in UI_CONFIG
        assert 'icon' in UI_CONFIG
        assert 'theme' in UI_CONFIG
    
    def test_service_import(self):
        """Test training service import and basic functionality."""
        service = TrainingService()
        assert service is not None
        assert hasattr(service, 'validate_backend_availability')
        assert hasattr(service, 'get_current_status')
        assert hasattr(service, 'get_metrics_for_charts')
    
    def test_operations_import(self):
        """Test operation handlers import."""
        start_op = StartTrainingOperation()
        stop_op = StopTrainingOperation()
        resume_op = ResumeTrainingOperation()
        
        assert start_op is not None
        assert stop_op is not None
        assert resume_op is not None
        
        # Test that they have required methods
        assert hasattr(start_op, 'execute_operation')
        assert hasattr(stop_op, 'execute_operation')
        assert hasattr(resume_op, 'execute_operation')
    
    def test_ui_components_import(self):
        """Test UI components creation."""
        ui_components = create_training_ui(DEFAULT_CONFIG)
        
        required_components = [
            'header_container', 'config_summary', 'form_container',
            'action_container', 'operation_container', 'chart_container',
            'footer_container', 'ui', 'main_container'
        ]
        
        for component in required_components:
            assert component in ui_components, f"Missing component: {component}"
    
    def test_initializer_import(self):
        """Test training initializer."""
        initializer = get_training_initializer()
        assert isinstance(initializer, TrainingInitializer)
        assert hasattr(initializer, 'initialize_module')
        assert hasattr(initializer, 'create_ui_components')


class TestTrainingService:
    """Test training service functionality."""
    
    def test_service_creation(self):
        """Test creating training service."""
        service = TrainingService()
        assert service.current_phase == TrainingPhase.IDLE
        assert not service.is_training
        assert service.current_config is not None
    
    def test_backend_validation(self):
        """Test backend availability validation."""
        service = TrainingService()
        status = service.validate_backend_availability()
        
        assert isinstance(status, dict)
        assert 'available' in status
        assert 'message' in status
    
    def test_current_status(self):
        """Test getting current status."""
        service = TrainingService()
        status = service.get_current_status()
        
        assert 'phase' in status
        assert 'is_training' in status
        assert 'current_metrics' in status
        assert 'training_history' in status
    
    def test_metrics_for_charts(self):
        """Test getting metrics formatted for charts."""
        service = TrainingService()
        metrics = service.get_metrics_for_charts()
        
        assert 'epochs' in metrics
        assert 'loss_chart' in metrics
        assert 'performance_chart' in metrics
        
        # Test chart structure
        loss_chart = metrics['loss_chart']
        assert 'title' in loss_chart
        assert 'data' in loss_chart
        assert 'color' in loss_chart
    
    @pytest.mark.asyncio
    async def test_simulation_training(self):
        """Test simulation training mode."""
        service = TrainingService()
        service.backend_service = None  # Force simulation mode
        
        progress_updates = []
        log_messages = []
        
        def progress_callback(percent, message):
            progress_updates.append((percent, message))
        
        def log_callback(message):
            log_messages.append(message)
        
        # Test short simulation training
        result = await service._run_simulation_training(
            epochs=3,
            progress_callback=progress_callback,
            log_callback=log_callback
        )
        
        assert result['success'] is True
        assert result['simulation'] is True
        assert len(progress_updates) > 0
        assert len(log_messages) > 0
        
        # Check metrics were updated
        metrics = service.get_metrics_for_charts()
        assert len(metrics['epochs']) == 3


class TestTrainingOperations:
    """Test training operations."""
    
    @pytest.mark.asyncio
    async def test_start_operation(self):
        """Test start training operation."""
        start_op = StartTrainingOperation()
        
        config = DEFAULT_CONFIG.copy()
        config['training']['epochs'] = 2  # Short test
        
        progress_msgs = []
        log_msgs = []
        
        def progress_callback(percent, message):
            progress_msgs.append((percent, message))
        
        def log_callback(message):
            log_msgs.append(message)
        
        result = await start_op.execute_operation(
            config=config,
            progress_callback=progress_callback,
            log_callback=log_callback
        )
        
        assert 'success' in result
        assert 'operation_id' in result
        assert len(progress_msgs) > 0
        assert len(log_msgs) > 0
    
    @pytest.mark.asyncio
    async def test_stop_operation(self):
        """Test stop training operation."""
        stop_op = StopTrainingOperation()
        
        result = await stop_op.execute_operation(
            progress_callback=lambda p, m: None,
            log_callback=lambda m: None
        )
        
        assert 'success' in result
        assert 'message' in result
    
    @pytest.mark.asyncio
    async def test_resume_operation(self):
        """Test resume training operation."""
        resume_op = ResumeTrainingOperation()
        
        # Create dummy checkpoint
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp_file:
            tmp_file.write(b"dummy checkpoint")
            checkpoint_path = tmp_file.name
        
        try:
            result = await resume_op.execute_operation(
                checkpoint_path=checkpoint_path,
                additional_epochs=5,
                progress_callback=lambda p, m: None,
                log_callback=lambda m: None
            )
            
            assert 'success' in result
            assert 'message' in result
        finally:
            Path(checkpoint_path).unlink()


class TestTrainingUIHandler:
    """Test training UI handler."""
    
    def test_handler_creation(self):
        """Test creating UI handler."""
        ui_components = create_training_ui(DEFAULT_CONFIG)
        handler = TrainingUIHandler(ui_components)
        
        assert handler is not None
        assert hasattr(handler, 'training_service')
        assert hasattr(handler, 'current_config')
    
    def test_config_extraction(self):
        """Test configuration extraction from UI."""
        ui_components = create_training_ui(DEFAULT_CONFIG)
        handler = TrainingUIHandler(ui_components)
        
        config = handler._extract_config_from_ui()
        assert isinstance(config, dict)
        assert 'training' in config
    
    def test_training_status(self):
        """Test getting training status."""
        ui_components = create_training_ui(DEFAULT_CONFIG)
        handler = TrainingUIHandler(ui_components)
        
        status = handler.get_training_status()
        assert 'training_active' in status
        assert 'current_config' in status
        assert 'service_status' in status
    
    def test_config_summary_generation(self):
        """Test configuration summary HTML generation."""
        ui_components = create_training_ui(DEFAULT_CONFIG)
        handler = TrainingUIHandler(ui_components)
        
        summary_html = handler._generate_config_summary_html()
        assert isinstance(summary_html, str)
        assert 'Training Configuration' in summary_html


class TestTrainingInitializer:
    """Test training initializer."""
    
    def test_initializer_creation(self):
        """Test creating initializer."""
        initializer = TrainingInitializer()
        assert initializer is not None
        assert hasattr(initializer, 'default_config')
    
    def test_ui_components_creation(self):
        """Test UI components creation."""
        initializer = TrainingInitializer()
        ui_components = initializer.create_ui_components()
        
        required_components = [
            'ui', 'main_container', 'ui_handler', 'chart_container'
        ]
        
        for component in required_components:
            assert component in ui_components, f"Missing component: {component}"
    
    def test_module_initialization(self):
        """Test module initialization."""
        initializer = TrainingInitializer()
        
        # Test initialization
        initialized_components = initializer.initialize_module()
        assert 'ui_handler' in initialized_components
        assert 'chart_container' in initialized_components


if __name__ == "__main__":
    pytest.main([__file__, "-v"])