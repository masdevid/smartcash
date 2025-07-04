"""Ensure that EnvConfig UI renders only a single LogAccordion instance."""

import pytest


def test_env_config_renders_single_log_accordion():
    """Test that the environment config UI uses a single LogAccordion instance."""
    from smartcash.ui.setup.env_config.components.ui_components import create_env_config_ui
    from unittest.mock import patch, MagicMock
    import ipywidgets as widgets
    
    # Create a real LogAccordion instance for testing
    from smartcash.ui.components.log_accordion import LogAccordion
    real_log_accordion = LogAccordion(module_name="Environment")
    
    # Setup all the mocks
    with patch('ipywidgets.VBox') as mock_vbox, \
         patch('ipywidgets.HBox') as mock_hbox, \
         patch('ipywidgets.Button') as mock_button, \
         patch('ipywidgets.Output') as mock_output, \
         patch('ipywidgets.Accordion') as mock_accordion, \
         patch('smartcash.ui.components.header_container.create_header_container') as mock_header, \
         patch('smartcash.ui.components.action_container.create_action_container') as mock_action, \
         patch('smartcash.ui.components.form_container.create_form_container') as mock_form, \
         patch('smartcash.ui.components.summary_container.create_summary_container') as mock_summary, \
         patch('smartcash.ui.components.progress_tracker.ProgressTracker') as mock_progress, \
         patch('smartcash.ui.setup.env_config.components.env_info_panel.create_env_info_panel') as mock_env_info, \
         patch('smartcash.ui.setup.env_config.components.tips_panel.create_tips_requirements') as mock_tips, \
         patch('smartcash.ui.components.footer_container.FooterContainer', autospec=True) as mock_footer_class, \
         patch('smartcash.ui.components.main_container.MainContainer') as mock_main, \
         patch('smartcash.ui.setup.env_config.components.setup_summary.create_setup_summary') as mock_setup_summary:
        
        # Configure the widget mocks to return proper widget instances
        mock_vbox.return_value = MagicMock()
        mock_vbox.return_value.children = tuple()
        mock_hbox.return_value = MagicMock()
        mock_hbox.return_value.children = tuple()
        mock_button.return_value = MagicMock()
        mock_button.return_value._click_callbacks = []
        mock_output.return_value = MagicMock()
        mock_accordion.return_value = MagicMock()
        
        # Mock the header container
        mock_header.return_value = MagicMock()
        
        # Mock the action container
        mock_action_instance = MagicMock()
        mock_action_instance['container'] = MagicMock()
        mock_action_instance['buttons'] = {'setup': MagicMock()}
        mock_action_instance['buttons']['setup']._click_callbacks = []
        mock_action_instance['dialog_area'] = MagicMock()
        mock_action_instance['show_dialog'] = MagicMock()
        mock_action_instance['show_info'] = MagicMock()
        mock_action_instance['clear_dialog'] = MagicMock()
        mock_action_instance['is_dialog_visible'] = MagicMock()
        mock_action.return_value = mock_action_instance
        
        # Mock the form container
        mock_form_instance = MagicMock()
        mock_form_instance['container'] = MagicMock()
        mock_form_instance['container'].children = tuple()
        mock_form.return_value = mock_form_instance
        
        # Mock the footer container
        mock_footer_instance = MagicMock()
        mock_footer_instance.container = MagicMock()
        # Use the real LogAccordion instance for the footer
        mock_footer_instance.log_accordion = real_log_accordion
        mock_footer_instance.log_output = MagicMock()
        mock_footer_instance.log = MagicMock()
        
        # Configure the FooterContainer class mock to return our instance
        mock_footer_class.return_value = mock_footer_instance
        
        # Trigger UI creation (no heavy initialization logic)
        ui_components = create_env_config_ui()
        
        # Verify that the log_accordion in the UI components is the same instance as our real_log_accordion
        assert ui_components['log_accordion'] is real_log_accordion, \
            "The log_accordion in UI components should be the same instance as the one in the footer container"
        
        # Also verify that log_output points to the same instance
        assert ui_components['log_output'] is real_log_accordion, \
            "The log_output in UI components should be the same instance as the log_accordion"
