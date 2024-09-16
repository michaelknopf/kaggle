from ml_soln.common.silence_warnings import silence_warnings
from functools import cache

silence_warnings()


@cache
def ctx():
    from ml_soln.housing_prices._context import _new_ctx
    return _new_ctx()
