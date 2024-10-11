from functools import cache


@cache
def ctx():
    from ml_soln.disaster_tweets._context import Context
    return Context()
