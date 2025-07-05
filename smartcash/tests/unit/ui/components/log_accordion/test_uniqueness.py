"""Tests ensuring that only a single LogAccordion instance is created per footer container.

The FooterContainer now uses the modern LogAccordion implementation directly instead of
the legacy procedural `create_log_accordion` helper. This test verifies that repeated
creation of footer containers with the same parameters creates separate LogAccordion
instances as expected, avoiding any shared state issues.
"""

import pytest


def test_no_duplicate_log_accordion():
    """Each footer container should have its own LogAccordion instance."""
    # Arrange
    from smartcash.ui.components.footer_container import create_footer_container

    # Act – create two footers with log accordions
    footer1 = create_footer_container(show_progress=False, show_info=False, show_logs=True)
    footer2 = create_footer_container(show_progress=False, show_info=False, show_logs=True)

    # Assert - each footer should have its own log accordion instance
    assert footer1.log_accordion is not None, "First footer should have a log accordion"
    assert footer2.log_accordion is not None, "Second footer should have a log accordion"
    
    # Each footer should have its own unique log accordion instance
    assert footer1.log_accordion is not footer2.log_accordion, "Each footer should have its own log accordion instance"
    
    # Both log accordions should be properly initialized
    assert hasattr(footer1.log_accordion, 'log'), "First log accordion should have log method"
    assert hasattr(footer2.log_accordion, 'log'), "Second log accordion should have log method"
