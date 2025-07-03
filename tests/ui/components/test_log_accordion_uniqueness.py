"""Tests ensuring that only a single legacy LogAccordion instance exists.

The FooterContainer (and other components) rely on the legacy procedural
`create_log_accordion` helper which stores references to created accordions in
an internal registry (`_log_accordions`).  This test verifies that repeated
creation of components that request the *same* accordion does **not** lead to
multiple entries being registered – guarding against UI duplication and memory
leaks.
"""

import pytest


def test_no_duplicate_log_accordion():
    """Repeated footer creation should not register duplicate log accordions."""
    # Arrange
    from smartcash.ui.components.footer_container import create_footer_container
    from smartcash.ui.components.log_accordion import legacy as legacy_log

    # Ensure a clean slate
    legacy_log._log_accordions.clear()

    # Act – create two footers that each request the default log accordion
    create_footer_container(show_progress=False, show_info=False, show_tips=False, show_logs=True)
    first_count = len(legacy_log._log_accordions)

    create_footer_container(show_progress=False, show_info=False, show_tips=False, show_logs=True)
    second_count = len(legacy_log._log_accordions)

    # Assert – the registry should still only contain ONE entry
    assert first_count == 1, "Expected exactly one log accordion to be registered after first creation"
    assert second_count == 1, "Duplicate log accordion detected – registry size increased after second creation"
