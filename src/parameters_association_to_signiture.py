import inspect
from typing import Dict

from typing_extensions import Iterable, List, Any
from inspect import Signature, Parameter

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


def associate_arguments_to_signatures(signatures: Iterable[Signature], arguments: Arguments) \
        -> List[ArgumentsSignatureAssociation]:
    split_args = []
    args, kwargs = list(arguments.args), dict(arguments.kwargs)
    signatures = [signature_deepcopy(sig) for sig in signatures]
    for signature in signatures:
        asc = ArgumentsSignatureAssociation(signature)
        for param in list(signature.parameters.values()):
            for kwargk, kwargv in dict(kwargs).items():
                if kwargk == param.name and can_be_keyword(param):
                    asc.arguments.kwargs[kwargk] = kwargv
                    signature.replace(parameters=[p for p in signature.parameters.values() if p.name != param.name])
                    kwargs.pop(kwargk)
        split_args.append(asc)
    for idx, arg in enumerate(list(args)):
        for signature, asc in zip(signatures, split_args):
            for param in list(signature.parameters.values()):
                param: inspect.Parameter = param
                arg: ArgumentsSignatureAssociation = arg
                if can_be_positional(param):
                    asc.arguments.args.append(arg)
                    signature.replace(parameters=[p for p in signature.parameters.values() if p.name != param.name])
                    args.pop(idx)

    for signature, asc in zip(signatures, split_args):
        for param in list(signature.parameters.values()):
            param: inspect.Parameter = param
            arg: ArgumentsSignatureAssociation = arg
            if param.kind == inspect.Parameter.VAR_POSITIONAL:
                for idx, arg in enumerate(list(args)):
                    asc.arguments.args.append(arg)
                    signature.replace(parameters=[p for p in signature.parameters.values() if p.name != param.name])
                    args.pop(idx)
                continue

    for signature, asc in zip(signatures, split_args):
        for param in list(signature.parameters.values()):
            param: inspect.Parameter = param
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                for kwargk, kwargv in dict(kwargs).items():
                    asc.arguments.kwargs[kwargk] = kwargv
                    signature.replace(parameters=[p for p in signature.parameters.values() if p.name != param.name])
                    kwargs.pop(kwargk)
                continue

    if args != [] or kwargs != {}:
        raise ValueError("Some values or unmached.")
    return split_args


def create_unified_signature