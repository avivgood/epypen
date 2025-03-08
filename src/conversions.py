from pydantic import VERSION

if VERSION.startswith("1."):
    from pydantic import parse_obj_as
else:
    from pydantic.v1 import parse_obj_as


def as_is_conversion(typ, obj):
    return obj


DEFAULT_PARAMETER_CONVERSIONS = [parse_obj_as, as_is_conversion]
DEFAULT_RETURN_TYPE_CONVERSIONS = [parse_obj_as, as_is_conversion]
