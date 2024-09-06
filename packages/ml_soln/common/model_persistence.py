import json
import pickle
import subprocess
from dataclasses import dataclass, asdict

from keras.src.callbacks import History

from ml_soln.common.paths import Paths, paths_for_package_name
from ml_soln.common.sagemaker_utils import sm_utils


@dataclass
class ModelPersistenceMeta:
    timestamp: str
    git_commit_hash: str
    job_name: str
    note: str
    sagemaker_env: dict

class ModelPersistence:

    def __init__(self,
                 paths: Paths):
        self.paths = paths
        self.commit_hash = None
        if not sm_utils.is_sagemaker:
            self.commit_hash = self._get_git_commit_hash()

    def save_model(self,
                   model,
                   history: History = None,
                   note=''):
        meta = ModelPersistenceMeta(timestamp=sm_utils.start_time,
                                    git_commit_hash=self.commit_hash,
                                    job_name=sm_utils.job_name,
                                    note=note,
                                    sagemaker_env=dict(sm_utils.sagemaker_env))
        self.paths.output_data_dir.mkdir(exist_ok=True, parents=True)
        self.paths.model_dir.mkdir(exist_ok=True, parents=True)
        with open(self.paths.output_data_dir / f'meta.json', 'w') as f:
            json.dump(asdict(meta), f, indent=2)
        with open(self.paths.model_dir / f'model.pkl', 'wb') as f:
            pickle.dump(model, f, protocol=5)
        if history:
            with open(self.paths.output_data_dir / f'history.pkl', 'wb') as f:
                pickle.dump(model, f, protocol=5)

    def load_model(self, job_name: str):
        paths = paths_for_package_name(package_name=self.paths.package_name, job_name=job_name)
        with open(paths.model_dir, 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def _get_git_commit_hash():
        result = subprocess.run(['git', 'rev-parse', 'HEAD'],
                                capture_output=True,
                                text=True,
                                check=True)
        return result.stdout.strip()
