import inspect
from inspect import Parameter
import pytest
from typing_extensions import Annotated
from src.main import converted, to, via


def test_signature_is_preserved_if_not_action_is_taken():
    def foo(x: int, y: int) -> int:
        return x + y

    wrapped_foo = converted(foo)

    assert inspect.signature(foo) == inspect.signature(wrapped_foo)


def test_signature_is_modified_by_annotations():
    @converted
    def foo(x: Annotated[int, via(str)], y: int = 3) -> Annotated[int, to(bool)]:
        return x + y

    assert inspect.signature(foo) == inspect.Signature(
        parameters=[
            Parameter("x", Parameter.POSITIONAL_OR_KEYWORD, annotation=str),
            Parameter("y", Parameter.POSITIONAL_OR_KEYWORD, annotation=int, default=3),
        ],
        return_annotation=bool,
    )


def test_target_instead_of_origin_annotation():
    with pytest.raises(TypeError):

        @converted
        def foo(x: Annotated[int, to(str)], y: int = 3) -> Annotated[int, to(bool)]:
            return x + y


def test_origin_instead_of_target_annotation():
    with pytest.raises(TypeError):

        @converted
        def foo(x: Annotated[int, via(str)], y: int = 3) -> Annotated[int, via(bool)]:
            return x + y


def test_more_data_inside_of_annotation_block_in_parameters():
    with pytest.raises(TypeError):

        @converted
        def foo(
            x: Annotated[int, via(str), int], y: int = 3
        ) -> Annotated[int, to(bool)]:
            return x + y

    with pytest.raises(TypeError):

        @converted
        def foo(
            x: Annotated[int, via(str), via(int)], y: int = 3
        ) -> Annotated[int, to(bool)]:
            return x + y

    with pytest.raises(TypeError):

        @converted
        def foo(
            x: Annotated[int, str, via(int)], y: int = 3
        ) -> Annotated[int, to(bool)]:
            return x + y

    with pytest.raises(TypeError):

        @converted
        def foo(
            x: Annotated[int, via(int)], y: Annotated[int, via(int), str] = 3
        ) -> Annotated[int, to(bool)]:
            return x + y


def test_more_data_inside_of_annotation_block_in_return_type():
    with pytest.raises(TypeError):

        @converted
        def foo(
            x: Annotated[int, via(str)], y: int = 3
        ) -> Annotated[int, to(bool), str]:
            return x + y

    with pytest.raises(TypeError):

        @converted
        def foo(
            x: Annotated[int, via(str)], y: int = 3
        ) -> Annotated[int, to(bool), to(str)]:
            return x + y

    with pytest.raises(TypeError):

        @converted
        def foo(
            x: Annotated[int, via(str)], y: int = 3
        ) -> Annotated[int, to(str), int]:
            return x + y


def test_mix_target_annotation_and_original_annotation():
    with pytest.raises(TypeError):

        @converted
        def foo(
            x: Annotated[bool, via(str), to(str)], y: int = 3
        ) -> Annotated[int, to(bool)]:
            return x + y

    with pytest.raises(TypeError):

        @converted
        def foo(
            x: Annotated[int, via(str)], y: int = 3
        ) -> Annotated[bool, via(int), to(bool)]:
            return x + y

    with pytest.raises(TypeError):

        @converted
        def foo(x: Annotated[to(int), via(str)], y: int = 3) -> Annotated[int, to(str)]:
            return x + y

    with pytest.raises(TypeError):

        @converted
        def foo(
            x: Annotated[int, via(str)], y: int = 3
        ) -> Annotated[via(int), to(str)]:
            return x + y
