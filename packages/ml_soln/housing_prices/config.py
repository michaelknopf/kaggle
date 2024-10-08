from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import yaml
from ml_soln.common.dataclass_utils import DictClassMixin


def load_config(package_dir: Path) -> 'ModelConfig':
    with open(package_dir / 'feature_config.yml') as f:
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
    categories: Optional[List[str]] = None
    null_rep: Optional[str] = None
