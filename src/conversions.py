from pydantic import parse_obj_as


def as_is_conversion(typ, obj):
    return obj


DEFAULT_CONVERSIONS = [as_is_conversion, parse_obj_as]
