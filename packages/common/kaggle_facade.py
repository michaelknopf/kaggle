import os
from zipfile import ZipFile

from common.paths import ROOT_DIR, CompetitionPaths

# this must run before importing the kaggle module
config_dir = os.environ.setdefault('KAGGLE_CONFIG_DIR', str(ROOT_DIR))
from kaggle import api, KaggleApi

HOUSE_PRICES_COMPETITION_NAME = 'house-prices-advanced-regression-techniques'

class KaggleFacade:

    def __init__(self,
                 competition: str,
                 paths: CompetitionPaths,
                 kaggle_api: KaggleApi = api):
        self.competition = competition
        self.paths = paths
        self.api = kaggle_api
        self.api.authenticate()

    def download_data(self):
        temp_zip = self.paths.DATA_DIR / f'{self.competition}.zip'
        output_folder = self.paths.DATA_DIR / 'kaggle_dataset'

        if output_folder.exists():
            return

        self.api.competition_download_files(self.competition, path=self.paths.DATA_DIR, quiet=False)

        with ZipFile(temp_zip) as f:
            f.extractall(output_folder)

        temp_zip.unlink()

    def submit_predictions(self, file_path, message=''):
        return self.api.competition_submit(competition=self.competition,
                                           file_name=file_path,
                                           message=message,
                                           quiet=False)
