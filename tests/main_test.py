import asyncio
import dataclasses
import inspect
from inspect import Parameter
from typing_extensions import Annotated, Dict
from pydantic import BaseModel
import pytest
from enum import Enum

from src.exceptions import ConversionsError
from src.main import converted, to, via


def test_support_for_all_argument_types():
    @converted
    def _test(foo1, foo2, /, foo3, foo4, *, foo5, foo6, **foo8):
        pass

    _test(1, 2, 5, foo4=6, foo5=7, foo6=10, foo8=4)


def test_results():
    @converted
    def _sum(a, b):
        return a + b

    assert _sum(1, 2) == 3


def test_converts_types():
    @converted
    def _cast(x: int, y: int) -> int:
        return str(x + y)

    assert _cast("2", "3") == 5


def test_no_needless_conversions():
    @converted
    def _cast(x: int, y: int) -> str:
        return str(x + y)

    assert _cast("2", "3") == "5"


def test_return_type_converts_according_to_annotations():
    @converted
    def _cast(x: int, y: int) -> Annotated[str, to(int)]:
        return str(x + y)

    assert _cast("2", "3") == 5


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


def test_no_param_conversions_failure():
    def always_fail(typ, obj):
        raise ValueError()

    def always_fail2(typ, oj):
        raise TypeError()

    @converted(parameter_conversions=[always_fail, always_fail2])
    def foo(x, y):
        pass

    with pytest.raises(ConversionsError) as err:
        foo(1, 2)

    errs = err.value.conversion_exceptions
    assert len(errs) == 2
    assert isinstance(errs[0], ValueError)
    assert isinstance(errs[1], TypeError)


def test_no_return_type_conversions_failure():
    def always_fail(typ, obj):
        raise ValueError()

    def always_fail2(typ, oj):
        raise TypeError()

    @converted(return_value_conversions=[always_fail, always_fail2])
    def foo(x, y):
        pass

    with pytest.raises(ConversionsError) as err:
        foo(1, 2)

    errs = err.value.conversion_exceptions
    assert len(errs) == 2
    assert isinstance(errs[0], ValueError)
    assert isinstance(errs[1], TypeError)


@pytest.mark.asyncio
async def test_async_support():
    @converted
    async def async_func(time: float):
        await asyncio.sleep(time)

    await async_func("0.01")


def test_empty_call_to_the_decorator_factory():
    @converted()
    def _cast(x: int, y: int) -> int:
        return str(x + y)

    assert _cast("2", "3") == 5


def test_instance_method_support():
    class A:
        def __init__(self, x):
            self.x = x

        @converted
        def cast(self, y: int) -> int:
            return str(self.x + y)

    a = A(2)
    assert a.cast("3") == 5


def test_class_method_support():
    class A:
        @classmethod
        @converted
        def cast(cls, x: int, y: int) -> int:
            return str(x + y)

    assert A.cast("2", "3") == 5


def test_static_method_support():
    class A:
        @staticmethod
        @converted
        def cast(x: int, y: int) -> int:
            return str(x + y)

    assert A.cast("2", "3") == 5


def test_base_model_conversion():
    class X(BaseModel):
        x: int

    class Y(BaseModel):
        y: int

    class Result(BaseModel):
        z: int

    @converted
    def add(x: X, y: Y) -> Result:
        return Result(z=x.x + y.y)

    assert add({"x": 2}, {"y": 3}) == Result(z=5)


def test_base_model_conversion_including_return_type():
    class X(BaseModel):
        x: int

    class Y(BaseModel):
        y: int

    class Result(BaseModel):
        z: int

    @converted
    def add(x: X, y: Y) -> dict:
        return Result(z=x.x + y.y)

    assert add({"x": "2"}, {"y": 3}) == {"z": 5}


def test_base_model_conversion_including_return_type_as_annotated():
    class X(BaseModel):
        x: int

    class Y(BaseModel):
        y: int

    class Result(BaseModel):
        z: int

    @converted
    def add(x: X, y: Y) -> Annotated[Result, to(dict)]:
        return Result(z=x.x + y.y)

    assert add({"x": 2}, {"y": 3}) == {"z": 5}


def test_dataclass_conversion():
    @dataclasses.dataclass
    class X:
        x: int

    @dataclasses.dataclass
    class Y:
        y: int

    @dataclasses.dataclass
    class Result:
        z: int

    @converted
    def add(x: X, y: Y):
        return Result(z=x.x + y.y)

    assert add(x=X(x=2), y=Y(y=3)).z == 5


def test_conversions_basemodel_to_dict_according_to_type_params_passed_to_dict():
    class M(BaseModel):
        x: str

    @converted
    def mock(m: Dict[str, int]):
        return m

    m = M(x="5")
    assert mock(m) == {"x": 5}


def test_enum_conversion():
    class Color(Enum):
        RED = "red"
        BLUE = "blue"

    @converted
    def process_color(color: Color) -> str:
        return color.value

    assert process_color("red") == "red"
    assert process_color(Color.BLUE) == "blue"

    @converted
    def process_color_back(color: str) -> Color:
        return Color(color)

    assert process_color_back("red") == Color.RED
    assert process_color_back(Color.BLUE.value) == Color.BLUE


def test_unannotated_return_type():
    @converted
    def unannotated_return(x: int):
        return str(x)

    # Should work without type conversion since return type is not annotated
    assert unannotated_return("5") == "5"


def test_raw_callable():
    def raw_func(x: int) -> int:
        return x + 1

    @converted
    def process_callable(func: callable) -> int:
        return func(5)

    assert process_callable(raw_func) == 6


def test_args_kwargs():
    @converted
    def process_args(*args: int, **kwargs: str) -> dict:
        return {"args": list(args), "kwargs": kwargs}

    result = process_args("1", "2", "3", a="hello", b="world")
    assert result == {"args": [1, 2, 3], "kwargs": {"a": "hello", "b": "world"}}


def test_string_annotations():
    @converted
    def string_annotated(x: "int", y: "str") -> "bool":
        return bool(int(x) + len(y))

    assert string_annotated("5", "hello") is True
    assert string_annotated("0", "") is False
