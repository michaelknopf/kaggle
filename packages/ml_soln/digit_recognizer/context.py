from dataclasses import dataclass

from ml_soln.common.model_persistence import ModelPersistence
from ml_soln.common.paths import CompetitionPaths, competition_paths_for_package_name
from ml_soln.digit_recognizer.model import Model
from ml_soln.digit_recognizer.prepare_data import DataPreparer
from ml_soln.digit_recognizer.trainer import Trainer


@dataclass
class Context:
    model_persistence: ModelPersistence
    paths: CompetitionPaths = competition_paths_for_package_name(__package__)
    data_preparer: DataPreparer = DataPreparer()
    model: Model = Model()
    trainer: Trainer = Trainer()
