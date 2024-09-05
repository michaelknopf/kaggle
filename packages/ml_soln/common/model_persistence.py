import subprocess
from dataclasses import dataclass, asdict
from datetime import datetime

from keras.src.callbacks import History

from ml_soln.common.paths import Paths
from ml_soln.common.sagemaker_utils import sm_utils
import pickle
import json


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
        self.paths.job_output_dir.mkdir(exist_ok=True)
        with open(self.paths.job_output_dir / f'meta.json', 'w') as f:
            json.dump(asdict(meta), f, indent=2)
        with open(self.paths.job_output_dir / f'model.pkl', 'wb') as f:
            pickle.dump(model, f, protocol=5)
        if history:
            with open(self.paths.job_output_dir / f'history.pkl', 'wb') as f:
                pickle.dump(model, f, protocol=5)

    def load_model(self, job_name: str):
        with open(self.paths.make_job_output_dir(job_name), 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def _get_git_commit_hash():
        result = subprocess.run(['git', 'rev-parse', 'HEAD'],
                                capture_output=True,
                                text=True,
                                check=True)
        return result.stdout.strip()
