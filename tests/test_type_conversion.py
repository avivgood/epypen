from typing_extensions import Annotated
from src.main import converted, to


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


def test_string_annotations():
    @converted
    def string_annotated(x: "int", y: "str") -> "bool":
        return bool(int(x) + len(y))

    assert string_annotated("5", "hello") is True
    assert string_annotated("0", "") is False
