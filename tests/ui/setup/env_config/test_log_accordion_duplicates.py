"""Ensure that EnvConfig UI renders only a single LogAccordion instance."""

import pytest


def test_env_config_renders_single_log_accordion():
    """EnvConfigInitializer should result in exactly one LogAccordion registry entry."""
    from smartcash.ui.setup.env_config.components.ui_components import create_env_config_ui
    from smartcash.ui.components.log_accordion import legacy as legacy_log

    # clean registry
    legacy_log._log_accordions.clear()

    # Trigger UI creation (no heavy initialization logic)
    create_env_config_ui()

    assert len(legacy_log._log_accordions) == 1, (
        f"Expected exactly 1 log accordion, got {len(legacy_log._log_accordions)}."
    )
