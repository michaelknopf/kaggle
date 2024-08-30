from common.model_persistence import ModelPersistence
from common.paths import CompetitionPaths, competition_paths_for_package_name
from digit_recognizer.context import Context
from digit_recognizer.model import Model
from digit_recognizer.prepare_data import DataPreparer
from digit_recognizer.trainer import Trainer


def _new_ctx():
    paths: CompetitionPaths = competition_paths_for_package_name(__package__)
    return Context(
        model_persistence=ModelPersistence(paths),
        paths=paths,
        data_preparer=DataPreparer(),
        model=Model(),
        trainer=Trainer(),
    )
