from enum import Enum
from src.main import converted


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
