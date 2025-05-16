import inspect
from abc import abstractclassmethod, abstractmethod
from dataclasses import dataclass
from inspect import Signature
from typing import Type

from typing_extensions import (
    Any,
    get_origin,
    Callable,
    get_args,
    Unpack,
    NewType,
    Annotated,
    ForwardRef,
    Tuple,
    Iterable,
    Optional,
    List,
    Dict,
)

from pydantic.version import VERSION as PYDANTIC_VERSION
from .exceptions import ConversionsError

Unknown = NewType("Unknown", Any)
PYDANTIC_VERSION_MINOR_TUPLE = tuple(int(x) for x in PYDANTIC_VERSION.split(".")[:2])
PYDANTIC_V2 = PYDANTIC_VERSION_MINOR_TUPLE[0] == 2

if PYDANTIC_V2:
    from pydantic._internal._typing_extra import eval_type_lenient

    evaluate_forwardref = eval_type_lenient
else:
    from pydantic.utils import evaluate_forwardref


def get_typed_annotation(annotation: Any, call: Callable[..., Any]) -> Any:
    globalns = getattr(call, "__globals__", {})
    if isinstance(annotation, str):
        annotation = ForwardRef(annotation)
        annotation = evaluate_forwardref(annotation, globalns, globalns)
    return annotation


def get_typed_signature(call: Callable[..., Any]) -> inspect.Signature:
    signature = inspect.signature(call)

    return signature.replace(
        parameters=[
            inspect.Parameter(
                name=param.name,
                kind=param.kind,
                default=param.default,
                annotation=get_origin_type_param(
                    get_typed_annotation(param.annotation, call)
                ),
            )
            for param in signature.parameters.values()
        ],
        return_annotation=get_target_type_return_value(
            get_typed_annotation(signature.return_annotation, call)
        ),
    )


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
    # Otherwise, assume the annotation itself is the per‐element type.
    return annotation


def can_be_positional(param: inspect.Parameter) -> bool:
    return param.kind in (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    )


def can_be_keyword(param: inspect.Parameter) -> bool:
    return param.kind in (
        inspect.Parameter.KEYWORD_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    )


def locate_positional_param(
        target_params: inspect.Signature,
) -> Optional[inspect.Parameter]:
    for param in target_params.parameters.values():
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            return param

    return None


def locate_keyword_param(
        target_params: inspect.Signature,
) -> Optional[inspect.Parameter]:
    for param in target_params.parameters.values():
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            return param

    return None


def convert(
        obj: object, to_typ: type, conversions: List[Callable[[object, type], Any]]
):
    errors = []
    for conversion in conversions:
        try:
            return conversion(to_typ, obj)
        except Exception as e:
            errors.append(e)

    raise ConversionsError(f"No conversion for object {obj}", errors)


class MetadataAnnotation:
    def __init__(self, annotation: Any):
        self.annotation = annotation

    @classmethod
    @abstractmethod
    def get_friendly_name(cls) -> str:
        ...


class TargetAnnotation(MetadataAnnotation):

    @classmethod
    def get_friendly_name(cls) -> str:
        return "to()"


class OriginAnnotation(MetadataAnnotation):
    @classmethod
    def get_friendly_name(cls) -> str:
        return "via()"


@dataclass
class TypeInfo:
    original_type: Type
    metadata_type: Type


def get_illegal_annotation(expected_annotation: Type[MetadataAnnotation], found_annotations: Iterable[MetadataAnnotation]) -> Iterable[MetadataAnnotation]:
    return filter(lambda annotation: isinstance(annotation, MetadataAnnotation) and not isinstance(annotation, expected_annotation), found_annotations)


def get_legal_annotation(expected_annotation: Type[MetadataAnnotation], found_annotations: Iterable[MetadataAnnotation]) -> Iterable[MetadataAnnotation]:
    return filter(lambda annotation: isinstance(annotation, expected_annotation), found_annotations)


def parse_type(typ: Type, metadata_type: Type[MetadataAnnotation]) -> TypeInfo:
    if get_origin(typ) is Annotated:
        # In Annotated[T, ...], T is the underlying type.
        annotations = get_args(typ)
        illegals = list(get_illegal_annotation(metadata_type, annotations))
        if len(illegals) != 0:
            raise TypeError(f"Expected {metadata_type.get_friendly_name()}, got {illegals[0].get_friendly_name()}")
        legals = list(get_legal_annotation(metadata_type, annotations))
        if len(annotations) > 2 and len(legals) != 0:
            raise TypeError(
                f"Annotating with {legals[0].get_friendly_name()} is only supported"
                f"with one metadata parameter (cannot do (Annotated[T1, {legals[0].get_friendly_name()}(T2), M]))"
            )
        if len(annotations) != 2:
            return TypeInfo(original_type=typ, metadata_type=typ)
        annotation = annotations[1]
        if isinstance(annotation, metadata_type):
            return TypeInfo(original_type=typ, metadata_type=annotation.annotation)

        return TypeInfo(original_type=typ, metadata_type=typ)
    else:
        return TypeInfo(original_type=typ, metadata_type=typ)



def get_original_signature(signature: Signature) -> Signature:
    return signature.replace(
        parameters=[
            inspect.Parameter(
                name=param.name,
                kind=param.kind,
                default=param.default,
                annotation=parse_type(
                    param.annotation, OriginAnnotation
                ).metadata_type,
            )
            for param in signature.parameters.values()
        ],
        return_annotation=parse_type(
            signature.return_annotation, TargetAnnotation
        ).metadata_type,
    )


def get_target_signature(signature: Signature) -> Signature:
    return signature.replace(
        parameters=[
            inspect.Parameter(
                name=param.name,
                kind=param.kind,
                default=param.default,
                annotation=parse_type(
                    param.annotation, OriginAnnotation
                ).original_type,
            )
            for param in signature.parameters.values()
        ],
        return_annotation=parse_type(
            signature.return_annotation, TargetAnnotation
        ).original_type,
    )
