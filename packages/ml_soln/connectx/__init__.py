from functools import cache


@cache
def ctx():
    from ml_soln.connectx._context import Context
    return Context()
