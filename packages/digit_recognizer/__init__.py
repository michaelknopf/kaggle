from functools import cache


@cache
def ctx():
    from digit_recognizer.context_init import _new_ctx
    return _new_ctx()
