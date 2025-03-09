from pydantic import BaseModel, parse_obj_as
from typing_extensions import get_origin


def as_is_conversion(typ, obj):
    return obj


def convert_object_to_any_type(typ, obj):
    # In pydantic V1, I did parse_obj_as,
    # However in Pydantic V2, parse_obj_as does not support
    # Conversions between BaseModel and dict without manually calling model_dump() or dict()
    # unlike parse_obj_as in V1 that knew to manually convert to dict
    # TypeAdapter(dict).validate_python(model) does not work either.
    # This function needs to support any time that is convertable by pydantic, just like parse_obj_as
    # in V1 did.
    original_type = get_origin(typ) or typ
    if isinstance(obj, BaseModel) and issubclass(original_type, dict):
        obj = obj.dict()

    return parse_obj_as(typ, obj)


DEFAULT_PARAMETER_CONVERSIONS = [convert_object_to_any_type, as_is_conversion]
DEFAULT_RETURN_TYPE_CONVERSIONS = [convert_object_to_any_type, as_is_conversion]
