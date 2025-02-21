import inspect
from typing_extensions import (Any, get_origin, overload, Callable, get_args, Unpack, NewType, Annotated,
                               Literal, Dict, TypeVar, ForwardRef, Tuple, TypeGuard, Union, Iterable, Optional, List)
import functools

from pydantic.version import VERSION as PYDANTIC_VERSION
PYDANTIC_VERSION_MINOR_TUPLE = tuple(int(x) for x in PYDANTIC_VERSION.split(".")[:2])
PYDANTIC_V2 = PYDANTIC_VERSION_MINOR_TUPLE[0] == 2

if PYDANTIC_V2:
    from pydantic._internal._typing_extra import eval_type_lenient
    evaluate_forwardref = eval_type_lenient
    from pydantic import parse_obj_as
else:
    from pydantic.typing import evaluate_forwardref
    from pydantic import parse_obj_as

def get_typed_annotation(annotation: Any, globalns: Dict[str, Any]) -> Any:
    if isinstance(annotation, str):
        annotation = ForwardRef(annotation)
        annotation = evaluate_forwardref(annotation, globalns, globalns)
    return annotation

def get_typed_signature(call: Callable[..., Any]) -> inspect.Signature:
    signature = inspect.signature(call)
    globalns = getattr(call, "__globals__", {})
    typed_params = [
        inspect.Parameter(
            name=param.name,
            kind=param.kind,
            default=param.default,
            annotation=get_typed_annotation(param.annotation, globalns),
        )
        for param in signature.parameters.values()
    ]
    typed_signature = inspect.Signature(typed_params)
    return typed_signature

"""
T = TypeVar("T")

_TypePredicate = Callable[[Any], bool]
_ConversionFunction = Callable[[Any], T]
_ConditionalConversionEntry = Tuple[_TypePredicate, _ConversionFunction]
_UnconditionalConversionEntry = _ConversionFunction
_ConversionEntry = Union[_ConditionalConversionEntry, _UnconditionalConversionEntry]
_Conversions: Iterable[_ConversionEntry]
"""
Unknown = NewType("Unknown", Any)


def extract_kwargs_value_type(annotation: Any, key: str) -> Any:
    """
    Given the annotation on a **kwargs parameter and a keyword name,
    return the expected type for that keyword argument’s value.

    Valid cases:
      1. A direct annotation (e.g. **kwargs: int) where the annotation is simply the type.
      2. An annotation wrapped in Annotated (per PEP 593); such wrappers are unwrapped.
      3. An annotation wrapped in Unpack (per PEP 646) where the inner type is a TypedDict
         (or a dict-like class with __annotations__).

    In case (3), if the key is present in the TypedDict’s __annotations__, that type is returned.
    Otherwise, a KeyError is raised.

    If the annotation is not Unpack-wrapped, it is assumed that every keyword's value is of the given type.
    """
    # Unwrap any Annotated[...] wrappers.
    try:
        while get_origin(annotation) is Annotated:
            # In Annotated[T, ...], T is the underlying type.
            annotation = get_args(annotation)[0]
    except Exception:
        return Unknown

    # Check for Unpack: if present, we expect the inner type to be a TypedDict.
    if get_origin(annotation) is Unpack:
        inner = get_args(annotation)[0]
        # Ensure inner is a class that qualifies as a TypedDict.
        if not (
                isinstance(inner, type)
                and issubclass(inner, dict)
                and hasattr(inner, "__annotations__")
        ):
            return Unknown
        td_annotations = inner.__annotations__
        if key in td_annotations:
            return td_annotations[key]
        else:
            return Unknown
    # Otherwise, assume a direct annotation applies to every keyword.
    return annotation

def extract_args_element_type(annotation: Any, idx_in_args: int) -> Any:
    """
    Given the annotation on a *args parameter, return the per‐element type.

    This function handles the following valid cases:
      1. A direct annotation (e.g. *args: int) where the annotation is simply the type.
      2. A homogeneous tuple annotation (e.g. *args: tuple[int, ...]).
      3. An annotation wrapped in Unpack (e.g. *args: Unpack[tuple[int, ...]]).

    Any annotation that attempts to use a heterogeneous tuple such as
      tuple[int, str, ...]
    is not considered valid for variadic parameters.
    """
    # Unwrap any Annotated[...] wrappers.
    try:
        while get_origin(annotation) is Annotated:
            # In Annotated[T, ...], T is the underlying type.
            annotation = get_args(annotation)[0]
    except Exception:
        return Unknown

    origin = get_origin(annotation)
    if origin is Unpack:
        # Unpack[...] returns a one-element tuple containing the type to unpack.
        inner = get_args(annotation)[0]
        if idx_in_args >= len(inner):
            return Unknown
        return origin[idx_in_args]
    if origin in (tuple, Tuple):
        args = get_args(annotation)
        # Expect homogeneous tuple: it must have exactly two arguments and the second one must be Ellipsis.
        if len(args) == 2 and args[1] is Ellipsis:
            return args[0]
    # Otherwise, assume the annotation itself is the per‐element type.
    return annotation
def can_be_positional(param: inspect.Parameter) -> bool:
    return param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)

def can_be_keyword(param: inspect.Parameter) -> bool:
    return param.kind in (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)

def locate_positional_param(target_params: inspect.Signature) -> Optional[inspect.Parameter]:
    for param in target_params.parameters.values():
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            return param

    return None

def locate_keyword_param(target_params: inspect.Signature) -> Optional[inspect.Parameter]:
    for param in target_params.parameters.values():
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            return param

    return None
def get_target_type_positional(
    target_params: inspect.Signature,
    original_args: tuple,
) -> Iterable[Tuple[object, type]]:
    original_args = iter(original_args)
    param = None
    for param_name, param in target_params.parameters.items():
        param: Optional[inspect.Parameter] = param
        # inspect.Parameter.annotation
        if can_be_positional(param):
            try:
                arg = next(original_args)
            except StopIteration:
                return
            yield arg, param.annotation

    rest = list(original_args)
    param = locate_positional_param(target_params)
    if param is not None:
        for idx, rest_arg in enumerate(rest):
            yield rest_arg, extract_args_element_type(param.annotation, idx)
    else:
        for rest_arg in rest:
            yield rest_arg, Unknown

def get_target_type_keyword(
        target_params: inspect.Signature, original_args: dict
) -> Iterable[Tuple[Tuple[str, object], type]]:
    remaining = dict(original_args)

    for param_name, param in target_params.parameters.items():
        if can_be_keyword(param):
            if param_name in remaining:
                val = remaining.pop(param_name)
                yield (param_name, val), param.annotation

    param = locate_keyword_param(target_params)
    if param is not None:
        for key, val in remaining:
            yield (key, val), extract_kwargs_value_type(param.annotation, key)
    else:
        for key, val in remaining:
            yield (key, val), Unknown

def convert(obj: object, to_typ: type, conversions: List[Callable[[object, type], Any]]):
    for conversion in conversions:
        try:
             return conversion(to_typ, obj)
        except Exception:
            pass

    raise NotImplemented(f"No conversion for object {obj}")

def as_is_conversion(typ, obj):
    return obj

DEFAULT_CONVERSIONS = [
    as_is_conversion,
    parse_obj_as
]


def converted(
    conversions: Union[Optional[List[Callable[[type, object], Any]]], Callable] = None
) -> Callable[Callable, Callable]:
    if conversions is None or callable(conversions):
        selected_conversions = DEFAULT_CONVERSIONS
    else:
        selected_conversions = conversions


    def deco(func: Callable) -> Callable:
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
            return_type = signature.return_annotation
            return convert(return_val, return_type, selected_conversions)

        return raw_function
    if callable(conversions):
        return deco(conversions)
    else:
        return deco


