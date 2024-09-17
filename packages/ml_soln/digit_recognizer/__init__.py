from functools import cache


@cache
def ctx():
    from ml_soln.digit_recognizer._context import Context
    return Context()
