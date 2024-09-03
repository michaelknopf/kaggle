from ml_soln.common.model_persistence import ModelPersistence
from ml_soln.common.paths import CompetitionPaths, competition_paths_for_package_name
from ml_soln.digit_recognizer.context import Context
from ml_soln.digit_recognizer.model import Model
from ml_soln.digit_recognizer.prepare_data import DataPreparer
from ml_soln.digit_recognizer.trainer import Trainer


def _new_ctx():
    paths: CompetitionPaths = competition_paths_for_package_name(__package__)
    return Context(
        model_persistence=ModelPersistence(paths),
        paths=paths,
        data_preparer=DataPreparer(),
        model=Model(),
        trainer=Trainer(),
    )
