import logging
from datetime import datetime
from pathlib import Path

from pandas import DataFrame

logger = logging.getLogger(__name__)


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
