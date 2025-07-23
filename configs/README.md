# SmartCash Configuration Directory

This directory contains configuration files for the SmartCash application.

## Default Files

- `base_config.yaml`: Main configuration file with application settings
- `colab_config.yaml`: Configuration specific to Google Colab environments

## Adding New Configurations

1. Create a new `.yaml` file in this directory
2. Use the existing configs as templates
3. The application will automatically discover and load the new configuration

## Syncing with Google Drive

In Colab environments, configurations can be synced with Google Drive to persist between sessions.
