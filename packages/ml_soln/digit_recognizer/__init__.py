from functools import cache


@cache
def ctx():
    from ml_soln.digit_recognizer._context import _new_ctx
    return _new_ctx()
