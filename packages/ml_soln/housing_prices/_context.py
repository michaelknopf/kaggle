from functools import cached_property

from ml_soln.common.context import BaseContext
from ml_soln.common.sagemaker_utils import sm_utils
from ml_soln.housing_prices.config import load_config, ModelConfig
from ml_soln.housing_prices.hyperparameters import HyperParams
from ml_soln.housing_prices.model import HousingPricesModel
from ml_soln.housing_prices.model_selection import HousingPricesModelSelection
from ml_soln.housing_prices.prepare_data import DataPreparer

class Context(BaseContext):

    @cached_property
    def data_preparer(self):
        return DataPreparer()

    @cached_property
    def model_config(self):
        return load_config(self.paths.package_dir)

    @cached_property
    def model(self):
        return HousingPricesModel()

    @cached_property
    def model_selection(self):
        return HousingPricesModelSelection()

    @cached_property
    def hyperparams(self):
        return HyperParams.from_dict(sm_utils.hyperparams)
