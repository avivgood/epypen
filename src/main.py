import inspect

from typing_extensions import (
    Any,
    Callable,
    Union,
    Optional,
    List,
)
import functools

from src.conversions import DEFAULT_CONVERSIONS
from src.utils import (
    get_typed_signature,
    get_target_type_keyword,
    get_target_type_positional,
    convert, get_typed_annotation, TargetAnnotation, update_signature_according_to_annotations,
)


def converted(
    conversions: Union[Optional[List[Callable[[type, object], Any]]], Callable] = None,
) -> Callable[Callable, Callable]:
    if conversions is None or callable(conversions):
        selected_conversions = DEFAULT_CONVERSIONS
    else:
        selected_conversions = conversions

    def deco(func: Callable) -> Callable:
        raw_func_signature = inspect.signature(func)
        @functools.wraps(func)
        def raw_function(*args, **kwargs):
            signature = get_typed_signature(func)
            typed_args = get_target_type_positional(signature, args)
            typed_kwargs = get_target_type_keyword(signature, kwargs)
            new_args = []
            new_kwargs = {}
            for obj, typ in typed_args:
                new_args.append(convert(obj, typ, selected_conversions))
            for kv, typ in typed_kwargs:
                key, val = kv
                new_kwargs[key] = convert(val, typ, selected_conversions)
            return_val = func(*new_args, **new_kwargs)
            return_type = get_typed_annotation(signature.return_annotation, func)
            return convert(return_val, return_type, selected_conversions)
        raw_function.__signature__ = update_signature_according_to_annotations(raw_func_signature)
        return raw_function
    if callable(conversions):
        return deco(conversions)
    else:
        return deco


def to(annotation: Any) -> TargetAnnotation:
    return TargetAnnotation(annotation)
