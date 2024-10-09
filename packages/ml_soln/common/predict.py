import logging
from datetime import datetime
from pathlib import Path
from typing import Callable

from pandas import DataFrame

from ml_soln.common.context import BaseContext

logger = logging.getLogger(__name__)

def get_model_and_paths(ctx: Callable[[], BaseContext],
                        model=None,
                        job_name: str = None):
    if not model and not job_name:
        raise ValueError("model or job_name must be provided")

    if not model:
        model = ctx().model_persistence.load_model(job_name)

    paths = ctx().paths
    if job_name:
        paths = paths.clone(job_name)

    return model, paths

def save_prediction(submission_df: DataFrame,
                    predictions_dir: Path,
                    filename=None,
                    columns=None):
    if not filename:
        filename = _default_filename()

    predictions_dir.mkdir(exist_ok=True, parents=True)
    prediction_file = predictions_dir / f'{filename}.csv'
    logger.info('Saving predictions to %s', prediction_file)
    with open(prediction_file, 'w') as f:
        submission_df.to_csv(f, index=False, columns=columns)

    return prediction_file

def _default_filename():
    return datetime.now().replace(microsecond=0).isoformat()
