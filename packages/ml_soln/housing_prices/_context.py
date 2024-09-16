from dataclasses import dataclass

from ml_soln.common.model_persistence import ModelPersistence
from ml_soln.common.paths import Paths
from ml_soln.common.sagemaker_utils import sm_utils
from ml_soln.housing_prices.config import load_config, ModelConfig
from ml_soln.housing_prices.hyperparameters import HyperParams
from ml_soln.housing_prices.model import HousingPricesModel
from ml_soln.housing_prices.model_selection import HousingPricesModelSelection
from ml_soln.housing_prices.prepare_data import DataPreparer

@dataclass
class Context:
    model_persistence: ModelPersistence
    paths: Paths
    data_preparer: DataPreparer
    model_config: ModelConfig
    model: HousingPricesModel
    model_selection: HousingPricesModelSelection
    hyperparams: HyperParams

def _new_ctx():
    paths: Paths = Paths.for_package_name(__package__)
    model_config = load_config()
    return Context(
        model_persistence=ModelPersistence(paths),
        paths=paths,
        data_preparer=DataPreparer(),
        model_config=model_config,
        model=HousingPricesModel(model_config),
        model_selection=HousingPricesModelSelection(),
        hyperparams=HyperParams.from_dict(sm_utils.hyperparams)
    )
