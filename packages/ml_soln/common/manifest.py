from dataclasses import dataclass
from functools import cache
from typing import List

import yaml

from ml_soln.common.dataclass_utils import DictClassMixin


@cache
def get_manifest() -> 'Manifest':
    # avoid circular import
    from ml_soln.common.paths import ML_SOLN_DIR

    MANIFEST_PATH = ML_SOLN_DIR / 'manifest.yml'
    with open(MANIFEST_PATH) as f:
        config_dict = yaml.safe_load(f)
    return Manifest.from_dict(config_dict)

@dataclass
class Manifest(DictClassMixin):
    competitions: List['Competition']

    def get_competition_by_package(self, package):
        return next(c for c in self.competitions if c.package == package)

    def list_competitions(self):
        return sorted(c.package for c in self.competitions)

@dataclass
class Competition(DictClassMixin):
    kaggle_name: str
    package: str
