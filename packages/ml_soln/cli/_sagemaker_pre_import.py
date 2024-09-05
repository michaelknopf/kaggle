import logging
import os
from pathlib import Path

def _pre_import():
    # Prevent sagemaker SDK from creating folders by default when imported
    if 'SAGEMAKER_BASE_DIR' not in os.environ:
        sagemaker_base_dir = Path('~').expanduser() / 'sagemaker_local' / 'jobs' / 'default' / 'opt' / 'ml'
        sagemaker_base_dir.mkdir(parents=True, exist_ok=True)
        os.environ['SAGEMAKER_BASE_DIR'] = str(sagemaker_base_dir)

    # Prevent logs about loading config file on import
    sagemaker_config_logger = logging.getLogger("sagemaker.config")
    sagemaker_config_logger.setLevel(logging.WARNING)
