#!/usr/bin/env python3

import click
from pathlib import Path
import yaml
from smartcash.utils.logger import SmartCashLogger
from smartcash.handlers.dataset_cleanup import DatasetCleanupHandler

# Initialize logger
logger = SmartCashLogger("smartcash")

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

@click.group()
@click.option('--config', '-c', default='configs/base_config.yaml',
              help='Path to configuration file')
@click.pass_context
def cli(ctx, config):
    """SmartCash - Indonesian Rupiah Currency Detection System

    This CLI provides commands to run different pipelines of the SmartCash project.
    Use --help with any command to see its usage.
    """
    ctx.ensure_object(dict)
    ctx.obj['config'] = load_config(config)
    ctx.obj['logger'] = logger

@cli.command()
@click.option('--data-dir', '-d', default='data',
              help='Directory containing the dataset')
@click.option('--backup-dir', '-b', default='backup',
              help='Directory for backups')
@click.option('--augmented-only/--all-files', default=True,
              help='Clean only augmented files or all files')
@click.pass_context
def cleanup(ctx, data_dir, backup_dir, augmented_only):
    """Clean and prepare the dataset."""
    logger.info("Starting dataset cleanup pipeline")
    
    handler = DatasetCleanupHandler(
        config_path=ctx.obj['config'],
        data_dir=data_dir,
        backup_dir=backup_dir,
        logger=logger
    )
    
    stats = handler.cleanup(
        augmented_only=augmented_only,
        create_backup=True
    )
    
    logger.info(f"Cleanup complete. Stats: {stats}")

@cli.command()
@click.option('--dataset', '-d', default='local',
              type=click.Choice(['local', 'roboflow']),
              help='Dataset source to use')
@click.option('--epochs', '-e', default=100,
              help='Number of training epochs')
@click.option('--batch-size', '-b', default=16,
              help='Batch size for training')
@click.pass_context
def train(ctx, dataset, epochs, batch_size):
    """Train the YOLOv5 model."""
    logger.info(f"Starting training pipeline with {dataset} dataset")
    # TODO: Implement training pipeline
    logger.info("Training pipeline not yet implemented")

@cli.command()
@click.option('--weights', '-w', required=True,
              help='Path to model weights')
@click.option('--test-data', '-t', default='data/test',
              help='Path to test data')
@click.pass_context
def evaluate(ctx, weights, test_data):
    """Evaluate model performance."""
    logger.info("Starting evaluation pipeline")
    # TODO: Implement evaluation pipeline
    logger.info("Evaluation pipeline not yet implemented")

@cli.command()
@click.option('--weights', '-w', required=True,
              help='Path to model weights')
@click.option('--source', '-s', required=True,
              help='Path to image or video file, or 0 for webcam')
@click.option('--conf-thres', '-c', default=0.25,
              help='Confidence threshold')
@click.pass_context
def detect(ctx, weights, source, conf_thres):
    """Run detection on images, video, or webcam."""
    logger.info("Starting detection pipeline")
    # TODO: Implement detection pipeline
    logger.info("Detection pipeline not yet implemented")

if __name__ == '__main__':
    cli(obj={})