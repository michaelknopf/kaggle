import sys
from abc import ABC
from functools import cached_property

from ml_soln.common.model_persistence import ModelPersistence
from ml_soln.common.paths import Paths


class BaseContext(ABC):

    def __init__(self, package=None):
        self.package = package or self._get_package()

    @classmethod
    def _get_package(cls):
        """
        Get the package that the subclass is defined in
        """
        return sys.modules[cls.__module__].__package__

    @cached_property
    def paths(self):
        return Paths.for_package_name(self.package)

    @cached_property
    def model_persistence(self):
        return ModelPersistence(self.paths)

# TODO: create @cached_property but with lock to make initialization thread-safe
