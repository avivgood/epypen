import inspect
from typing import Dict, OrderedDict

from typing_extensions import Iterable, List, Any
from inspect import Signature, Parameter, BoundArguments

from src.utils import can_be_keyword, can_be_positional


# hack to avoid name collisions between **kwargs and signatures
class Arguments:
    def __init__(self, args: List[Any], kwargs: Dict):
        self.args = args
        self.kwargs = kwargs


class ArgumentsSignatureAssociation:
    def __init__(self, signature: Signature):
        self.arguments = Arguments([], {})
        self.signature = signature


def signature_deepcopy(signature: inspect.Signature) -> inspect.Signature:
    return inspect.Signature(list(signature.parameters.values()), return_annotation=signature.return_annotation)


def bind_args(signature: Signature, args: OrderedDict[str, Any]):
    signature_params = dict(signature.parameters)
    args = []
    kwargs = {}
    for param_name, param_value in args.values():
            if can_be_keyword(signature_params[unified_parameter_name]):
                kwargs[unified_parameter_name] = unified_parameter_value
            else:
                args.append(unified_parameter_value)


def bind_multiple(signatures: List[Signature], arguments: Arguments) \
        -> List[BoundArguments]:
    result = []

    unified_signature = unify_signatures(signatures)
    bound = unified_signature.bind(*arguments.args, **arguments.kwargs)

    for signature in signatures:
        args = []
        kwargs = {}
        signature_params = dict(signature.parameters)
        for unified_parameter_name, unified_parameter_value in bound.arguments.values():
            if unified_parameter_name in signature_params:
                if can_be_keyword(signature_params[unified_parameter_name]):
                    kwargs[unified_parameter_name] = unified_parameter_value
                else:
                    args.append(unified_parameter_value)

        result.append(signature.bind(*args, **kwargs))

    return result

def unify_signatures(signatures: List[inspect.Signature]) -> Signature:
    if len(signatures) == 0:
        return inspect.Signature()
    params = [
        param for signature in signatures
        for param in signature.parameters.values()
    ]
    params.sort(key=lambda param: (param.kind, not param.default == Parameter.empty))
    param_kinds = [param.kind for param in params]
    # For some reason, inspect.Signature does not validate duplicate *args or **kwargs
    if param_kinds.count(inspect.Parameter.VAR_KEYWORD):
        raise ValueError("Only one instance of **kwargs allowed")
    if param_kinds.count(inspect.Parameter.VAR_POSITIONAL):0
        raise ValueError("Only one instance of *args allowed")
    return inspect.Signature(parameters=params,
                             return_annotation=signatures[0].return_annotation,
                             __validate_parameters__=True)




