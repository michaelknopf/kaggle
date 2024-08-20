from dataclasses import fields, is_dataclass, asdict
from typing import get_args, get_origin, Type, ForwardRef
import inspect

class DictClassMixin:

    def to_dict(self):
        return to_dict(self)

    @classmethod
    def from_dict(cls, d):
        return from_dict(d, cls)

def to_dict(data):
    if isinstance(data, list):
        return [to_dict(x) for x in data]
    else:
        return asdict(data)

def from_dict(data, data_class: Type):
    if not is_dataclass(data_class):
        raise ValueError(f"{data_class} is not a dataclass")
    return _convert_dataclass(data, data_class)

def _convert(x, cls, parent):
    if x is None:
        return None
    cls = _resolve_forward_ref(cls, parent)
    if is_dataclass(cls):
        return _convert_dataclass(x, cls)
    elif get_origin(cls) == list:
        return _convert_list(x, cls, parent)
    else:
        return x

def _convert_dataclass(d, cls):
    kwargs = {field.name: _convert(d.get(field.name), field.type, parent=cls)
              for field in fields(cls)}
    return cls(**kwargs)

def _convert_list(values, cls, parent):
    generic_types = get_args(cls)
    if len(generic_types) != 1:
        return values
    generic_type = generic_types[0]
    generic_type = _resolve_forward_ref(generic_type, parent)
    return [_convert(v, generic_type, parent=cls) for v in values]

def _resolve_forward_ref(cls, parent):
    if isinstance(cls, ForwardRef):
        parent_module = inspect.getmodule(parent)
        parent_globals = vars(parent_module)
        return cls._evaluate(parent_globals, locals(), frozenset())
    return cls
