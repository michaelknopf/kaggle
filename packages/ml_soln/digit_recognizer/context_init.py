from ml_soln.common.model_persistence import ModelPersistence
from ml_soln.common.paths import Paths
from ml_soln.common.sagemaker_utils import sm_utils
from ml_soln.digit_recognizer.context import Context
from ml_soln.digit_recognizer.hyperparameters import HyperParams
from ml_soln.digit_recognizer.model import Model
from ml_soln.digit_recognizer.prepare_data import DataPreparer
from ml_soln.digit_recognizer.trainer import Trainer


def _new_ctx():
    paths: Paths = Paths.for_package_name(__package__)
    hyperparams = HyperParams.from_dict(sm_utils.hyperparams)
    return Context(
        model_persistence=ModelPersistence(paths),
        paths=paths,
        data_preparer=DataPreparer(),
        model=Model(),
        trainer=Trainer(),
        hyperparams=hyperparams
    )
