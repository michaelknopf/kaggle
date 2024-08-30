from dataclasses import dataclass

from common.model_persistence import ModelPersistence
from common.paths import CompetitionPaths, competition_paths_for_package_name
from digit_recognizer.model import Model
from digit_recognizer.prepare_data import DataPreparer
from digit_recognizer.trainer import Trainer


@dataclass
class Context:
    model_persistence: ModelPersistence
    paths: CompetitionPaths = competition_paths_for_package_name(__package__)
    data_preparer: DataPreparer = DataPreparer()
    model: Model = Model()
    trainer: Trainer = Trainer()
