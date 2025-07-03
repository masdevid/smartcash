"""Tests for FooterContainer log display functionality."""

import pytest
import ipywidgets as widgets

from smartcash.ui.components.footer_container import FooterContainer


def _get_entries_container(footer_container: FooterContainer) -> widgets.Widget:
    """Helper to retrieve the entries_container VBox from the footer container's log accordion.

    The widget hierarchy created by ``create_log_accordion`` is:

        Accordion
            Box (log_container)
                VBox (entries_container) <- we want this
                ... other children (e.g. scroll spacer)

    Args:
        footer_container: FooterContainer instance with logs enabled.

    Returns:
        ipywidgets.VBox containing the individual log entry widgets.
    """
    accordion = footer_container.log_accordion
    assert isinstance(accordion, widgets.Accordion), "Log accordion should be an ipywidgets.Accordion"
    # The first child of the accordion is the log_container (a Box)
    assert len(accordion.children) > 0, "Accordion should have at least one child"
    log_container = accordion.children[0]
    assert isinstance(log_container, widgets.Box), "First child of accordion should be the log container Box"
    # The first child of the log_container is the entries_container (a VBox)
    assert len(log_container.children) > 0, "Log container should have children"
    entries_container = log_container.children[0]
    return entries_container


def test_footer_container_logs_are_displayed():
    """Ensure that calling FooterContainer.log() adds an entry to the log accordion UI."""
    # Create a footer container with only logs enabled to minimise widget hierarchy complexity
    fc = FooterContainer(
        show_progress=False,
        show_logs=True,
        show_info=False,
        show_tips=False,
        log_module_name="TestModule",
        log_height="100px",
    )

    entries_container = _get_entries_container(fc)
    initial_count = len(entries_container.children)

    # Add a log entry via the public API
    test_message = "This is a test log message"
    fc.log(test_message, level="info")

    # The UI should now contain one additional child widget representing the new log entry.
    updated_count = len(entries_container.children)
    assert updated_count == initial_count + 1, (
        "Log accordion did not update after adding a log entry. "
        f"Count before: {initial_count}, after: {updated_count}"
    )

    # Optionally, verify the text content of the last log entry widget
    last_entry_widget = entries_container.children[-1]
    if hasattr(last_entry_widget, "value"):
        # HTML widgets expose their HTML via the `value` attribute
        assert test_message in last_entry_widget.value, (
            "The rendered log entry does not contain the expected message"
        )
