from datetime import datetime
from pathlib import Path

from pandas import DataFrame


def save_submission(submission_df: DataFrame, submissions_dir: Path, filename=None):
    if not filename:
        filename = datetime.now().replace(microsecond=0).isoformat()

    submissions_dir.mkdir(exist_ok=True)
    submission_file = submissions_dir / f'{filename}.csv'
    with open(submission_file, 'w') as f:
        submission_df.to_csv(f, index=False, columns=['Id', 'SalePrice'])

    return submission_file
