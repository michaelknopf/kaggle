from dataclasses import dataclass, field
from functools import cache
from typing import List, Optional

import yaml

from housing_prices.paths import paths
from common.dataclass_utils import DictClassMixin


@cache
def load_config() -> 'ModelConfig':
    with open(paths.HOUSING_PRICES_DIR / 'feature_config.yml') as f:
        config_dict = yaml.safe_load(f)
    return ModelConfig.from_dict(config_dict)

@dataclass
class ModelConfig(DictClassMixin):
    features: List['FeatureConfig']

    def categorical_features(self):
        return (x for x in self.features if x.type == 'categorical')

    def ordinal_features(self):
        return (x for x in self.features if x.type == 'ordinal')

@dataclass
class FeatureConfig(DictClassMixin):
    name: str
    type: str
    categories: Optional[List[str]]
    null_rep: Optional[str]
