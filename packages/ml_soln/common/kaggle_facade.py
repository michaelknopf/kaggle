import os
from zipfile import ZipFile

from ml_soln.common.paths import ROOT_DIR, Paths
from ml_soln.common.manifest import get_manifest

# this must run before importing the kaggle module
config_dir = os.environ.setdefault('KAGGLE_CONFIG_DIR', str(ROOT_DIR))
from kaggle import api, KaggleApi

class KaggleFacade:

    def __init__(self,
                 competition: str,
                 paths: Paths,
                 kaggle_api: KaggleApi = api):
        self.competition = get_manifest().get_competition_by_package(competition)
        self.paths = paths
        self.api = kaggle_api
        self.api.authenticate()

    def download_data(self):
        temp_zip = self.paths.input_dir / f'{self.competition.kaggle_name}.zip'
        output_folder = self.paths.input_dir / 'kaggle_dataset'

        if output_folder.exists():
            return

        self.api.competition_download_files(self.competition.kaggle_name, path=self.paths.input_dir, quiet=False)

        with ZipFile(temp_zip) as f:
            f.extractall(output_folder)

        temp_zip.unlink()

    def submit_predictions(self, file_path, message):
        if not message:
            message = ''
        return self.api.competition_submit(competition=self.competition.kaggle_name,
                                           file_name=file_path,
                                           message=message,
                                           quiet=False)
