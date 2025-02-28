from pydantic import parse_obj_as, BaseModel


def as_is_conversion(typ, obj):
    return obj


DEFAULT_PARAMETER_CONVERSIONS = [parse_obj_as, as_is_conversion]


def convert_model_to_dict(typ, obj: BaseModel):
    return obj.dict()


DEFAULT_RETURN_TYPE_CONVERSIONS = [parse_obj_as, convert_model_to_dict, as_is_conversion]

