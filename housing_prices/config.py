from dataclasses import dataclass
from functools import cache
from typing import List, Optional

import yaml

from housing_prices.path_anchor import HOUSING_PRICES_DIR
from housing_prices.dataclass_utils import from_dict


@cache
def load_config():
    with open(HOUSING_PRICES_DIR / 'feature_config.yml') as f:
        config_dict = yaml.safe_load(f)
    return from_dict(config_dict, ModelConfig)

@dataclass
class ModelConfig:
    features: List['FeatureConfig']

    def categorical_features(self):
        return (x for x in self.features if x.type == 'categorical')

    def ordinal_features(self):
        return (x for x in self.features if x.type == 'ordinal')

@dataclass
class FeatureConfig:
    name: str
    type: str
    categories: Optional[List[str]]
