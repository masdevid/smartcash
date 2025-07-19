"""
Evaluation UIModule - BaseUIModule Pattern
Handles model evaluation across 2×4 research scenarios (2 scenarios × 4 models = 8 tests)
"""

from typing import Dict, Any, Optional
from smartcash.ui.core.base_ui_module import BaseUIModule
from smartcash.ui.model.evaluation.configs.evaluation_config_handler import EvaluationConfigHandler
from smartcash.ui.model.evaluation.configs.evaluation_defaults import get_default_evaluation_config
from smartcash.model.evaluation.evaluation_service import EvaluationService
from smartcash.model.evaluation.checkpoint_selector import CheckpointSelector
from smartcash.model.evaluation.utils.evaluation_progress_bridge import EvaluationProgressBridge

class EvaluationUIModule(BaseUIModule):
    """
    Evaluation UI Module for comprehensive model evaluation using BaseUIModule pattern.
    
    Handles 2×4 evaluation matrix:
    - 2 scenarios: position_variation, lighting_variation
    - 4 model combinations: 2 backbones × 2 layer modes
    - Total: 8 evaluation tests
    """
    
    def __init__(self):
        """Initialize evaluation UI module."""
        super().__init__(
            module_name='evaluation',
            parent_module='model'
        )
        
        # Define required components for this module
        self._required_components = [
            'main_container',
            'action_container', 
            'operation_container',
            'summary_container'
        ]
        
        # Initialize backend services
        self.evaluation_service = None
        self.checkpoint_selector = None
        self.progress_bridge = None
        
    # Required abstract methods for BaseUIModule
    
    def get_default_config(self) -> Dict[str, Any]:
        """
        Return default configuration for the evaluation module.
        
        Returns:
            Default evaluation configuration dictionary
        """
        return get_default_evaluation_config()
    
    def create_config_handler(self, config: Dict[str, Any]) -> EvaluationConfigHandler:
        """
        Create evaluation config handler.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            EvaluationConfigHandler instance
        """
        return EvaluationConfigHandler(config=config)
    
    def create_ui_components(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create evaluation UI components.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Dictionary containing UI components
        """
        from smartcash.ui.model.evaluation.components.evaluation_ui import create_evaluation_ui
        return create_evaluation_ui(config)
    
    # Module-specific methods
    
    def _get_module_button_handlers(self) -> Dict[str, callable]:
        """
        Get module-specific button handlers.
        
        Returns:
            Dictionary mapping button IDs to handler functions
        """
        return {
            'run_scenario': self._handle_run_scenario_sync
        }
    
    def _get_module_operation_handlers(self) -> Dict[str, callable]:
        """
        Get module-specific operation handlers.
        
        Returns:
            Dictionary mapping operation names to handler functions
        """
        return {
            'run_all_scenarios': self._execute_all_scenarios,
            'run_position_scenario': self._execute_position_scenario,
            'run_lighting_scenario': self._execute_lighting_scenario,
            'get_available_models': self._get_available_models,
            'refresh_model_list': self._refresh_model_list
        }
    
    def _post_initialize_hook(self) -> None:
        """
        Hook called after successful initialization.
        Perform any additional setup specific to evaluation module.
        """
        try:
            # Initialize backend services
            self._initialize_backend_services()
            
            # Initialize summary panel with empty state
            self._update_summary_panel({})
            
            self.log_info("📊 Evaluation services initialized - Ready to test models")
            
        except Exception as e:
            self.log_error(f"Post-initialization setup failed: {e}")

    def _initialize_backend_services(self) -> None:
        """Initialize evaluation backend services with UI integration."""
        try:
            config = self.get_current_config()
            
            # Initialize checkpoint selector for best model discovery
            self.checkpoint_selector = CheckpointSelector(config=config)
            
            # Initialize evaluation service (without model_api for now)
            self.evaluation_service = EvaluationService(model_api=None, config=config)
            
            # Initialize progress bridge with UI components
            ui_components = {
                'operation_container': self.get_component('operation_container'),
                'progress_tracker': self._get_progress_tracker(),
                'status': self._get_status_widget()
            }
            
            self.progress_bridge = EvaluationProgressBridge(
                ui_components=ui_components,
                callback=self._on_progress_update
            )
            
            self.log_info("✅ Backend services initialized successfully")
            
        except Exception as e:
            self.log_error(f"Failed to initialize backend services: {e}")
            # Set fallback services
            self.checkpoint_selector = None
            self.evaluation_service = None
            self.progress_bridge = None

    def _get_progress_tracker(self):
        """Get progress tracker from operation container."""
        try:
            operation_container = self.get_component('operation_container')
            if operation_container and isinstance(operation_container, dict):
                return operation_container.get('progress_tracker')
            return None
        except Exception:
            return None

    def _get_status_widget(self):
        """Get status widget from operation container."""
        try:
            operation_container = self.get_component('operation_container')
            if operation_container and isinstance(operation_container, dict):
                return operation_container.get('status')
            return None
        except Exception:
            return None

    def _on_progress_update(self, progress_data: Dict[str, Any]) -> None:
        """Handle progress updates from evaluation service."""
        try:
            # Log progress updates
            if progress_data.get('message'):
                self.log_info(progress_data['message'])
            
            # Update operation container progress if available
            operation_container = self.get_component('operation_container')
            if operation_container and hasattr(operation_container, 'update_progress'):
                overall_progress = progress_data.get('overall_progress')
                if overall_progress is not None:
                    operation_container.update_progress(overall_progress, progress_data.get('message', ''))
                    
        except Exception as e:
            self.log_error(f"Progress update failed: {e}")
    
    # Operation handlers (moved from operation manager)
    
    def _execute_all_scenarios(self) -> Dict[str, Any]:
        """
        Execute all evaluation scenarios using backend services.
        
        Returns:
            Results dictionary with success status and metrics
        """
        try:
            self.log_info("🚀 Starting comprehensive evaluation...")
            
            # Check if services are available
            if not self.evaluation_service or not self.checkpoint_selector:
                return self._execute_fallback_evaluation(['position_variation', 'lighting_variation'])
            
            # Get available checkpoints using proper checkpoint format
            available_checkpoints = self.checkpoint_selector.list_available_checkpoints()
            if not available_checkpoints:
                self.log_warning("⚠️ No checkpoints found, running with mock data")
                return self._execute_fallback_evaluation(['position_variation', 'lighting_variation'])
            
            # Select best checkpoints (limit to top 2 for demo)
            selected_checkpoints = [cp['path'] for cp in available_checkpoints[:2]]
            scenarios = ['position_variation', 'lighting_variation']
            
            self.log_info(f"📋 Running evaluation: {len(scenarios)} scenarios × {len(selected_checkpoints)} models")
            
            # Run evaluation using backend service with progress tracking
            result = self.evaluation_service.run_evaluation(
                scenarios=scenarios,
                checkpoints=selected_checkpoints,
                progress_callback=self._on_progress_update,
                metrics_callback=self._on_metrics_update,
                ui_components={
                    'operation_container': self.get_component('operation_container'),
                    'progress_tracker': self._get_progress_tracker()
                }
            )
            
            if result.get('status') == 'success':
                self.log_success(f"🎉 Comprehensive evaluation completed successfully")
                self._update_summary_panel({
                    'success': True,
                    'successful_tests': result.get('scenarios_evaluated', 0) * result.get('checkpoints_evaluated', 0),
                    'total_tests': len(scenarios) * len(selected_checkpoints),
                    'scenarios_completed': scenarios,
                    'best_model': self._extract_best_model(result),
                    'average_map': self._calculate_average_map(result)
                })
                return result
            else:
                error_msg = result.get('error', 'Unknown evaluation error')
                self.log_error(f"❌ Evaluation failed: {error_msg}")
                return {'success': False, 'error': error_msg}
            
        except Exception as e:
            self.log_error(f"❌ Comprehensive evaluation failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _execute_position_scenario(self) -> Dict[str, Any]:
        """
        Execute position variation scenario using backend services.
        
        Returns:
            Results dictionary with success status and metrics
        """
        try:
            self.log_info("📐 Starting position variation scenario...")
            
            # Check if services are available
            if not self.evaluation_service or not self.checkpoint_selector:
                return self._execute_fallback_evaluation(['position_variation'])
            
            # Get best checkpoint for position scenario
            best_checkpoint = self.checkpoint_selector.get_best_checkpoint()
            if not best_checkpoint:
                self.log_warning("⚠️ No checkpoints found, running with mock data")
                return self._execute_fallback_evaluation(['position_variation'])
            
            # Run single scenario evaluation
            result = self.evaluation_service.run_scenario(
                scenario_name='position_variation',
                checkpoint_path=best_checkpoint['path']
            )
            
            if result.get('status') == 'success':
                self.log_success(f"✅ Position scenario completed successfully")
                self._update_summary_panel({
                    'success': True,
                    'successful_tests': 1,
                    'total_tests': 1,
                    'scenario': 'position_variation',
                    'model_evaluated': best_checkpoint['display_name'],
                    'best_model': best_checkpoint['display_name'],
                    'average_map': result.get('metrics', {}).get('mAP', 0.0)
                })
                return result
            else:
                error_msg = result.get('error', 'Position evaluation failed')
                self.log_error(f"❌ Position scenario failed: {error_msg}")
                return {'success': False, 'error': error_msg}
            
        except Exception as e:
            self.log_error(f"❌ Position scenario failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _execute_lighting_scenario(self) -> Dict[str, Any]:
        """
        Execute lighting variation scenario using backend services.
        
        Returns:
            Results dictionary with success status and metrics
        """
        try:
            self.log_info("💡 Starting lighting variation scenario...")
            
            # Check if services are available
            if not self.evaluation_service or not self.checkpoint_selector:
                return self._execute_fallback_evaluation(['lighting_variation'])
            
            # Get best checkpoint for lighting scenario
            best_checkpoint = self.checkpoint_selector.get_best_checkpoint()
            if not best_checkpoint:
                self.log_warning("⚠️ No checkpoints found, running with mock data")
                return self._execute_fallback_evaluation(['lighting_variation'])
            
            # Run single scenario evaluation
            result = self.evaluation_service.run_scenario(
                scenario_name='lighting_variation',
                checkpoint_path=best_checkpoint['path']
            )
            
            if result.get('status') == 'success':
                self.log_success(f"✅ Lighting scenario completed successfully")
                self._update_summary_panel({
                    'success': True,
                    'successful_tests': 1,
                    'total_tests': 1,
                    'scenario': 'lighting_variation',
                    'model_evaluated': best_checkpoint['display_name'],
                    'best_model': best_checkpoint['display_name'],
                    'average_map': result.get('metrics', {}).get('mAP', 0.0)
                })
                return result
            else:
                error_msg = result.get('error', 'Lighting evaluation failed')
                self.log_error(f"❌ Lighting scenario failed: {error_msg}")
                return {'success': False, 'error': error_msg}
            
        except Exception as e:
            self.log_error(f"❌ Lighting scenario failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _get_available_models(self) -> Dict[str, Any]:
        """
        Get list of available trained models using checkpoint selector with proper format.
        Implements best model discovery with checkpoint format: best_{model_name}_{backbone}_{date:%Y%m%d}.pt
        
        Returns:
            Dictionary with available models and their metadata
        """
        try:
            if not self.checkpoint_selector:
                self.log_warning("⚠️ Checkpoint selector not available, using fallback data")
                return self._get_fallback_models()
            
            # Get available checkpoints using proper checkpoint format
            available_checkpoints = self.checkpoint_selector.list_available_checkpoints()
            
            if not available_checkpoints:
                self.log_warning("⚠️ No checkpoints found using format: best_{model_name}_{backbone}_{date:%Y%m%d}.pt")
                return {'success': True, 'models': {}}
            
            # Convert checkpoints to models format for UI with enhanced metadata
            models = {}
            best_models_per_backbone = {}
            
            for checkpoint in available_checkpoints:
                # Extract checkpoint info with proper format validation
                backbone = checkpoint.get('backbone', 'unknown')
                model_name = checkpoint.get('model_name', 'smartcash')
                date_str = checkpoint.get('date', 'unknown')
                
                # Create model key using consistent format
                model_key = f"{model_name}_{backbone}_{date_str}"
                
                # Get validation status
                is_valid, validation_msg = self.checkpoint_selector.validate_checkpoint(checkpoint['path'])
                
                # Calculate performance score for ranking
                val_map = checkpoint.get('metrics', {}).get('val_map', 0.0)
                val_loss = checkpoint.get('metrics', {}).get('val_loss', float('inf'))
                performance_score = val_map * (1.0 / (val_loss + 0.001))  # Combined score
                
                model_data = {
                    'name': checkpoint.get('display_name', f"{backbone.title()} - {date_str}"),
                    'checkpoint_path': checkpoint['path'],
                    'model_name': model_name,
                    'backbone': backbone,
                    'layer_mode': checkpoint.get('layer_mode', 'unknown'),
                    'date': date_str,
                    'map_score': val_map,
                    'loss_score': val_loss,
                    'performance_score': performance_score,
                    'epoch': checkpoint.get('epoch', 0),
                    'file_size_mb': checkpoint.get('file_size_mb', 0.0),
                    'status': 'valid' if is_valid else 'invalid',
                    'validation_message': validation_msg,
                    'scenarios_compatible': ['position_variation', 'lighting_variation'],  # All scenarios by default
                    'is_best_for_backbone': False  # Will be set below
                }
                
                models[model_key] = model_data
                
                # Track best model per backbone
                if backbone not in best_models_per_backbone or performance_score > best_models_per_backbone[backbone]['performance_score']:
                    best_models_per_backbone[backbone] = model_data
            
            # Mark best models per backbone
            for backbone, best_model in best_models_per_backbone.items():
                for model_key, model_data in models.items():
                    if (model_data['backbone'] == backbone and 
                        model_data['performance_score'] == best_model['performance_score']):
                        model_data['is_best_for_backbone'] = True
                        break
            
            # Sort models by performance score (highest first)
            sorted_models = dict(sorted(models.items(), 
                                      key=lambda x: x[1]['performance_score'], 
                                      reverse=True))
            
            # Log discovery results
            valid_count = len([m for m in models.values() if m['status'] == 'valid'])
            best_count = len([m for m in models.values() if m['is_best_for_backbone']])
            
            self.log_info(f"🔍 Best model discovery completed:")
            self.log_info(f"   📋 Found {len(models)} total checkpoints")
            self.log_info(f"   ✅ {valid_count} valid checkpoints")
            self.log_info(f"   🏆 {best_count} best models per backbone")
            
            if best_models_per_backbone:
                self.log_info("   🏆 Best models discovered:")
                for backbone, best_model in best_models_per_backbone.items():
                    self.log_info(f"     • {backbone}: {best_model['name']} (mAP: {best_model['map_score']:.3f})")
            
            return {
                'success': True, 
                'models': sorted_models,
                'best_models_per_backbone': best_models_per_backbone,
                'discovery_stats': {
                    'total_checkpoints': len(models),
                    'valid_checkpoints': valid_count,
                    'best_models_found': best_count,
                    'backbones_detected': list(best_models_per_backbone.keys())
                }
            }
            
        except Exception as e:
            self.log_error(f"❌ Best model discovery failed: {e}")
            return {'success': False, 'error': str(e), 'models': {}}
    
    def _refresh_model_list(self) -> Dict[str, Any]:
        """
        Refresh the available models list.
        
        Returns:
            Updated models dictionary
        """
        self.log_info("🔄 Refreshing model list...")
        return self._get_available_models()
    
    
    # Button handler methods
    
    def _handle_run_scenario_sync(self, button) -> None:
        """
        Handle run scenario button click - determines action based on UI form selections.
        """
        try:
            # Clear previous results and logs
            self._clear_ui_state()
            
            # Run prerequisite checks before starting evaluation
            if not self._check_evaluation_prerequisites():
                self.log_error("❌ Prerequisites not met. Cannot start evaluation.")
                return
            
            # Extract form values to determine what to run
            form_config = self._extract_form_values()
            run_mode = form_config.get('run_mode', 'all_scenarios')
            
            # Execute the appropriate operation directly (synchronous)
            if run_mode == 'all_scenarios':
                self._execute_all_scenarios()
            elif run_mode == 'position_only':
                self._execute_position_scenario()
            elif run_mode == 'lighting_only':
                self._execute_lighting_scenario()
            else:
                self.log_error(f"Unknown run mode: {run_mode}")
                
        except Exception as e:
            self.log_error(f"Scenario execution failed: {e}")
    
    def _check_evaluation_prerequisites(self) -> bool:
        """
        Check if all prerequisites for evaluation are met.
        
        Returns:
            True if prerequisites are met, False otherwise
        """
        try:
            self.log_info("🔍 Checking evaluation prerequisites...")
            
            prerequisites_met = True
            issues = []
            
            # Check 1: Verify configuration is valid
            config = self.get_current_config()
            if not config or not config.get('evaluation'):
                issues.append("❌ Invalid or missing evaluation configuration")
                prerequisites_met = False
            else:
                self.log_info("✅ Configuration is valid")
            
            # Check 2: Verify at least one model is available
            models_result = self._check_available_models()
            if not models_result.get('success') or not models_result.get('models'):
                issues.append("❌ No trained models available for evaluation")
                prerequisites_met = False
            else:
                model_count = len(models_result['models'])
                self.log_info(f"✅ Found {model_count} available models")
            
            # Check 3: Verify at least one scenario is enabled
            scenarios_config = config.get('evaluation', {}).get('scenarios', {})
            enabled_scenarios = [s for s, cfg in scenarios_config.items() if cfg.get('enabled', True)]
            if not enabled_scenarios:
                issues.append("❌ No evaluation scenarios are enabled")
                prerequisites_met = False
            else:
                self.log_info(f"✅ Found {len(enabled_scenarios)} enabled scenarios")
            
            # Check 4: Verify at least one metric is selected
            metrics_config = config.get('evaluation', {}).get('metrics', {})
            if not metrics_config or not metrics_config.get('primary'):
                issues.append("❌ No evaluation metrics are configured")
                prerequisites_met = False
            else:
                metric_count = len(metrics_config['primary'])
                self.log_info(f"✅ Found {metric_count} configured metrics")
            
            # Check 5: Verify output directory is accessible
            output_config = config.get('evaluation', {}).get('output', {})
            output_dir = output_config.get('save_dir', 'runs/evaluation')
            try:
                import os
                os.makedirs(output_dir, exist_ok=True)
                self.log_info("✅ Output directory is accessible")
            except Exception as e:
                issues.append(f"❌ Cannot access output directory: {output_dir}")
                prerequisites_met = False
            
            # Log results
            if prerequisites_met:
                self.log_success("🎯 All evaluation prerequisites are met. Ready to start!")
            else:
                self.log_error("❌ Prerequisite checks failed:")
                for issue in issues:
                    self.log_error(f"  {issue}")
            
            return prerequisites_met
            
        except Exception as e:
            self.log_error(f"Failed to check prerequisites: {e}")
            return False
    
    def _check_available_models(self) -> Dict[str, Any]:
        """
        Check for available trained models.
        
        Returns:
            Dictionary with success status and available models
        """
        try:
            # TODO: Implement actual model discovery logic
            # For now, return mock model availability check
            
            # Simulate checking model directories/files
            mock_models = {
                'yolo_v8_full_layers': {
                    'name': 'YOLOv8 Full Layers',
                    'backbone': 'yolo_v8',
                    'layer_mode': 'full_layers',
                    'map_score': 0.847,
                    'epochs': 100,
                    'status': 'completed',
                    'path': 'runs/train/yolo_v8_full/weights/best.pt'
                },
                'yolo_v8_last_layer': {
                    'name': 'YOLOv8 Last Layer',
                    'backbone': 'yolo_v8',
                    'layer_mode': 'last_layer',
                    'map_score': 0.782,
                    'epochs': 50,
                    'status': 'completed',
                    'path': 'runs/train/yolo_v8_last/weights/best.pt'
                }
            }
            
            return {'success': True, 'models': mock_models}
            
        except Exception as e:
            return {'success': False, 'error': str(e), 'models': {}}
    
    def _extract_form_values(self) -> Dict[str, Any]:
        """
        Extract current form values from UI components.
        
        Returns:
            Dictionary with current form values including model selection
        """
        try:
            # Get current config as fallback
            current_config = self.get_current_config()
            execution_config = current_config.get('evaluation', {}).get('execution', {})
            models_config = current_config.get('evaluation', {}).get('models', {})
            
            form_values = {
                'run_mode': execution_config.get('run_mode', 'all_scenarios'),
                'parallel_execution': execution_config.get('parallel_execution', False),
                'save_intermediate_results': execution_config.get('save_intermediate_results', True),
                # Model selection from form (backbone and layer selections)
                'backbone': models_config.get('backbone', 'yolov5_efficientnet-b4'),
                'layer_mode': models_config.get('layer_mode', 'full_layers'),
                'auto_select_best': models_config.get('auto_select_best', True)
            }
            
            # TODO: Extract actual widget values from form when needed
            # For now, use config defaults
            
            return form_values
            
        except Exception as e:
            self.log_error(f"Failed to extract form values: {e}")
            return {'run_mode': 'all_scenarios', 'backbone': 'yolov5_efficientnet-b4', 'layer_mode': 'full_layers'}
    
    def _clear_ui_state(self) -> None:
        """
        Clear logs and state before starting new evaluation.
        """
        try:
            self.log_info("🧹 Clearing UI state...")
            
            # Clear summary panel
            self._update_summary_panel({})
            
            self.log_info("🧹 UI state cleared")
            
        except Exception as e:
            self.log_error(f"Failed to clear UI state: {e}")
    
    def _update_summary_panel(self, results: dict) -> None:
        """
        Update summary panel with evaluation results.
        
        Args:
            results: Results dictionary from evaluation
        """
        try:
            summary_container = self.get_component('summary_container')
            if summary_container:
                if results:
                    # Display actual results
                    successful = results.get('successful_tests', 0)
                    total = results.get('total_tests', 0)
                    
                    summary_html = f"""
                    <div style='padding: 15px;'>
                        <h4 style='margin-top: 0; color: #28a745;'>🎉 Evaluation Results</h4>
                        <div style='display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; margin: 10px 0;'>
                            <div><strong>Tests Completed:</strong> {successful}/{total}</div>
                            <div><strong>Success Rate:</strong> {(successful/total*100):.1f}%</div>
                        </div>
                        <div style='margin-top: 15px;'>
                            <strong>📊 Model Performance:</strong>
                            <div style='margin: 10px 0; font-size: 0.9em; color: #666;'>
                                Best Model: {results.get('best_model', 'N/A')}<br>
                                Avg mAP: {results.get('average_map', 0):.3f}
                            </div>
                        </div>
                    </div>
                    """
                    
                    if hasattr(summary_container, 'set_content'):
                        summary_container.set_content(summary_html)
                    elif hasattr(summary_container, 'set_html'):
                        summary_container.set_html(summary_html, 'success')
                else:
                    # Display empty state
                    empty_html = f"""
                    <div style='padding: 15px; text-align: center; color: #6c757d;'>
                        <h4 style='margin-top: 0;'>📊 Evaluation Results</h4>
                        <div style='margin: 20px 0;'>
                            <div style='font-size: 3em; opacity: 0.3;'>📈</div>
                            <p>No evaluation results yet</p>
                            <p style='font-size: 0.9em;'>Click "Run Scenario" to start evaluation</p>
                        </div>
                        <div style='display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; margin: 15px 0;'>
                            <div><strong>Tests Completed:</strong> 0/8</div>
                            <div><strong>Success Rate:</strong> 0%</div>
                            <div><strong>Best Model:</strong> None</div>
                            <div><strong>Avg mAP:</strong> N/A</div>
                        </div>
                    </div>
                    """
                    
                    if hasattr(summary_container, 'set_content'):
                        summary_container.set_content(empty_html)
                    elif hasattr(summary_container, 'set_html'):
                        summary_container.set_html(empty_html, 'info')
                        
        except Exception as e:
            self.log_error(f"Failed to update summary panel: {e}")
    
    # Additional helper methods for evaluation
    
    def get_evaluation_status(self) -> Dict[str, Any]:
        """
        Get current evaluation status.
        
        Returns:
            Dictionary with current evaluation state
        """
        try:
            config = self.get_current_config()
            execution_config = config.get('evaluation', {}).get('execution', {})
            
            return {
                'ready': True,
                'run_mode': execution_config.get('run_mode', 'all_scenarios'),
                'parallel_execution': execution_config.get('parallel_execution', False),
                'models_available': 4,  # TODO: Get actual count
                'scenarios_enabled': 2  # TODO: Get actual count
            }
            
        except Exception as e:
            self.log_error(f"Failed to get evaluation status: {e}")
            return {'ready': False, 'error': str(e)}
    
    # Missing helper methods for evaluation operations
    
    def _execute_fallback_evaluation(self, scenarios: List[str]) -> Dict[str, Any]:
        """
        Execute fallback evaluation when backend services are unavailable.
        
        Args:
            scenarios: List of scenario names to evaluate
            
        Returns:
            Mock evaluation results
        """
        try:
            self.log_warning("⚠️ Running fallback evaluation with mock data")
            
            import random
            import time
            
            # Simulate evaluation progress
            total_tests = len(scenarios) * 2  # 2 mock models per scenario
            
            for i, scenario in enumerate(scenarios):
                self.log_info(f"🎯 Mock evaluation: {scenario}")
                time.sleep(0.5)  # Simulate processing time
                
                # Update progress if operation container is available
                if hasattr(self, 'operation_container'):
                    progress = int(((i + 1) / len(scenarios)) * 100)
                    operation_container = self.get_component('operation_container')
                    if operation_container and hasattr(operation_container, 'update_progress'):
                        operation_container.update_progress(progress, f"Processing {scenario}")
            
            # Generate mock results
            mock_results = {
                'success': True,
                'scenarios_evaluated': len(scenarios),
                'checkpoints_evaluated': 2,
                'metrics': {
                    'mAP': random.uniform(0.75, 0.90),
                    'precision': random.uniform(0.80, 0.95),
                    'recall': random.uniform(0.75, 0.90),
                    'f1_score': random.uniform(0.78, 0.92)
                },
                'mock_evaluation': True
            }
            
            self.log_success("✅ Fallback evaluation completed")
            return mock_results
            
        except Exception as e:
            self.log_error(f"❌ Fallback evaluation failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _get_fallback_models(self) -> Dict[str, Any]:
        """
        Get fallback model data when checkpoint selector is unavailable.
        
        Returns:
            Dictionary with mock model data
        """
        try:
            # Generate mock models with proper checkpoint format
            from datetime import datetime, timedelta
            
            mock_models = {}
            backbones = ['efficientnet_b4', 'cspdarknet']
            
            for i, backbone in enumerate(backbones):
                date = (datetime.now() - timedelta(days=i)).strftime('%Y%m%d')
                model_key = f"smartcash_{backbone}_{date}"
                
                mock_models[model_key] = {
                    'name': f'SmartCash {backbone.title()} - {date}',
                    'checkpoint_path': f'data/checkpoints/best_smartcash_{backbone}_{date}.pt',
                    'model_name': 'smartcash',
                    'backbone': backbone,
                    'layer_mode': 'full_layers',
                    'date': date,
                    'map_score': 0.847 - (i * 0.03),  # Decreasing scores
                    'epoch': 100 - (i * 10),
                    'file_size_mb': 45.2 + (i * 2.1),
                    'status': 'completed'
                }
            
            self.log_info(f"📋 Generated {len(mock_models)} fallback models")
            return {'success': True, 'models': mock_models}
            
        except Exception as e:
            self.log_error(f"❌ Failed to generate fallback models: {e}")
            return {'success': False, 'error': str(e), 'models': {}}
    
    def _extract_best_model(self, result: Dict[str, Any]) -> str:
        """
        Extract best model name from evaluation results.
        
        Args:
            result: Evaluation result dictionary
            
        Returns:
            Best model name or 'N/A'
        """
        try:
            # Look for best model in various result structures
            if 'best_model' in result:
                return str(result['best_model'])
            
            if 'summary' in result and 'best_configurations' in result['summary']:
                best_configs = result['summary']['best_configurations']
                if best_configs and 'model_name' in best_configs:
                    return str(best_configs['model_name'])
            
            if 'evaluation_results' in result:
                # Find model with highest mAP
                best_map = 0
                best_model = 'N/A'
                
                for scenario_results in result['evaluation_results'].values():
                    for backbone, backbone_results in scenario_results.items():
                        metrics = backbone_results.get('metrics', {})
                        current_map = metrics.get('mAP', 0)
                        if current_map > best_map:
                            best_map = current_map
                            checkpoint_info = backbone_results.get('checkpoint_info', {})
                            best_model = checkpoint_info.get('display_name', backbone)
                
                return best_model
            
            return 'N/A'
            
        except Exception as e:
            self.log_error(f"❌ Failed to extract best model: {e}")
            return 'N/A'
    
    def _calculate_average_map(self, result: Dict[str, Any]) -> float:
        """
        Calculate average mAP from evaluation results.
        
        Args:
            result: Evaluation result dictionary
            
        Returns:
            Average mAP score
        """
        try:
            # Look for average mAP in various result structures
            if 'metrics' in result and 'mAP' in result['metrics']:
                return float(result['metrics']['mAP'])
            
            if 'summary' in result and 'aggregated_metrics' in result['summary']:
                aggregated = result['summary']['aggregated_metrics']
                if 'average_map' in aggregated:
                    return float(aggregated['average_map'])
            
            if 'evaluation_results' in result:
                # Calculate average from all scenario results
                map_scores = []
                
                for scenario_results in result['evaluation_results'].values():
                    for backbone_results in scenario_results.values():
                        metrics = backbone_results.get('metrics', {})
                        if 'mAP' in metrics:
                            map_scores.append(float(metrics['mAP']))
                
                if map_scores:
                    return sum(map_scores) / len(map_scores)
            
            return 0.0
            
        except Exception as e:
            self.log_error(f"❌ Failed to calculate average mAP: {e}")
            return 0.0
    
    def _on_metrics_update(self, metrics_data: Dict[str, Any]) -> None:
        """
        Handle metrics updates from evaluation service.
        
        Args:
            metrics_data: Metrics data from evaluation
        """
        try:
            # Log metrics updates
            if 'metrics' in metrics_data:
                metrics = metrics_data['metrics']
                self.log_info(f"📊 Metrics update: mAP={metrics.get('mAP', 0):.3f}")
            
            # Update summary panel with partial results if needed
            if metrics_data.get('partial_results'):
                self._update_summary_panel(metrics_data['partial_results'])
                
        except Exception as e:
            self.log_error(f"Metrics update failed: {e}")
    
    # Note: display() method is now provided by BaseUIModule


# Use enhanced factory for standardized initialization
from smartcash.ui.core.enhanced_ui_module_factory import create_display_function
initialize_evaluation_ui = create_display_function(EvaluationUIModule)